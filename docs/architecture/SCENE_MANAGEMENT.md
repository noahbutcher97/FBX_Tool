# Scene Management Architecture

## Overview

The FBX Tool implements a reference-counted scene manager to efficiently manage FBX scene lifecycle, prevent memory leaks, and enable scene sharing between GUI, visualization, and analysis components.

## Problem Statement

**Before Scene Manager:**
- Analysis always cleaned up scenes immediately after use
- GUI and visualizer each loaded separate copies of the same FBX file
- Running analysis while visualizer was open would crash (scene deleted while in use)
- No way to overlay analysis data on visualizer
- Memory leaks in batch processing

**After Scene Manager:**
- Scenes are shared via reference counting
- Analysis can run while visualizer is open
- Clear button properly frees memory
- Smart caching prevents memory bloat
- Foundation for future overlay features

## Architecture

### Core Components

#### 1. FBXSceneManager (`fbx_tool/analysis/scene_manager.py`)

**Singleton pattern:** Only one instance manages all scenes globally.

```python
from fbx_tool.analysis.scene_manager import get_scene_manager

scene_manager = get_scene_manager()
```

**Key Features:**
- **Reference counting:** Tracks how many "owners" need each scene
- **Automatic cleanup:** Destroys scene when ref count reaches 0
- **Thread-safe:** Uses locks for concurrent access
- **Cache support:** Reuses scenes for the same file path

#### 2. FBXSceneReference (`fbx_tool/analysis/scene_manager.py`)

**Smart pointer** that automatically releases scene when done.

```python
# Manual release
scene_ref = scene_manager.get_scene("path/to/file.fbx")
scene = scene_ref.scene
# ... use scene ...
scene_ref.release()

# Context manager (auto-release)
with scene_manager.get_scene("path/to/file.fbx") as scene_ref:
    scene = scene_ref.scene
    # ... use scene ...
# Automatically released here
```

## Usage Patterns

### GUI Integration

**Location:** `fbx_tool/gui/main_window.py`

```python
class MainWindow(QMainWindow):
    def __init__(self):
        from fbx_tool.analysis.scene_manager import get_scene_manager
        self.scene_manager = get_scene_manager()
        self.active_scene_refs = {}  # {filepath: FBXSceneReference}

    def launch_visualizer(self):
        filepath = self.fbx_files[0]

        # Get or create scene reference
        if filepath not in self.active_scene_refs:
            self.active_scene_refs[filepath] = self.scene_manager.get_scene(filepath)

        # Launch visualizer - it gets its own reference
        self.viewer_window = launch_skeleton_viewer(
            self.active_scene_refs[filepath].scene,
            fbx_files=self.fbx_files,
            scene_manager=self.scene_manager
        )

    def clearSelectedFiles(self):
        """Clear button - releases all scene references."""
        for filepath, scene_ref in self.active_scene_refs.items():
            scene_ref.release()
        self.active_scene_refs.clear()
        # Memory freed!
```

### Visualizer Integration with Smart Caching

**Location:** `fbx_tool/visualization/opengl_viewer.py`

**Smart Caching Strategy:**
- Keep only current ± 1 files in memory
- Prevents memory bloat when switching through large batches
- Still gets cache hits if GUI holds references

```python
class FBXSkeletonViewer(QOpenGLWidget):
    def __init__(self, scene, fbx_files=None, scene_manager=None):
        from fbx_tool.analysis.scene_manager import get_scene_manager
        self.scene_manager = scene_manager or get_scene_manager()

        self.fbx_files = fbx_files or []
        self.current_file_index = 0
        self.scene_refs = {}  # {index: FBXSceneReference}

        # Get reference for initial scene
        if self.fbx_files:
            self.scene_refs[0] = self.scene_manager.get_scene(self.fbx_files[0])

    def _switch_to_file(self, index):
        """Switch files with smart caching (current ± 1)."""
        # Determine which files to keep cached
        files_to_keep = {index}
        if index > 0:
            files_to_keep.add(index - 1)  # Previous file
        if index < len(self.fbx_files) - 1:
            files_to_keep.add(index + 1)  # Next file

        # Release files not in keep set
        for file_index in list(self.scene_refs.keys()):
            if file_index not in files_to_keep:
                self.scene_refs[file_index].release()
                del self.scene_refs[file_index]

        # Get scene reference (cache hit if GUI has it)
        if index not in self.scene_refs:
            self.scene_refs[index] = self.scene_manager.get_scene(self.fbx_files[index])

        self.scene = self.scene_refs[index].scene
        self.update()

    def closeEvent(self, event):
        """Release all scene references on close."""
        for index, scene_ref in self.scene_refs.items():
            scene_ref.release()
        self.scene_refs.clear()
        event.accept()
```

### Analysis Worker Integration

**Location:** `fbx_tool/gui/main_window.py` (AnalysisWorker class)

```python
class AnalysisWorker(QThread):
    def run(self):
        from fbx_tool.analysis.scene_manager import get_scene_manager
        scene_manager = get_scene_manager()
        scene_ref = None

        try:
            # Get scene from scene manager (cache hit if GUI has it!)
            scene_ref = scene_manager.get_scene(self.fbx_file)
            scene = scene_ref.scene

            # Run analysis...
            metadata = get_scene_metadata(scene)
            # ... more analysis ...

        finally:
            # CRITICAL: Release even on error
            if scene_ref is not None:
                scene_ref.release()
                self.progress.emit("\n🧹 Scene reference released")
```

## Workflow Examples

### Example 1: Load → Analyze → Clear

```
1. User loads file.fbx in GUI
   └─> GUI calls scene_manager.get_scene("file.fbx")
   └─> Ref count = 1

2. User runs analysis
   └─> Worker calls scene_manager.get_scene("file.fbx")
   └─> Cache HIT! Ref count = 2
   └─> Worker finishes, releases ref
   └─> Ref count = 1 (GUI still holds it)

3. User clicks "Clear"
   └─> GUI releases ref
   └─> Ref count = 0
   └─> Scene automatically destroyed! Memory freed!
```

### Example 2: Visualizer + Analysis Simultaneously

```
1. User loads file.fbx, opens visualizer
   └─> GUI: ref count = 1
   └─> Visualizer: ref count = 2

2. User runs analysis while visualizer open
   └─> Worker: ref count = 3
   └─> All three components share SAME scene object!
   └─> Worker finishes: ref count = 2

3. User closes visualizer
   └─> Visualizer releases: ref count = 1
   └─> Scene stays alive (GUI still has it)

4. User clicks "Clear"
   └─> GUI releases: ref count = 0
   └─> Scene destroyed
```

### Example 3: Batch Processing with Smart Caching

```
User loads 100 FBX files, switches through visualizer:

Without smart caching:
└─> All 100 scenes stay in memory = MEMORY BLOAT

With smart caching (current ± 1):
└─> Only 3 scenes in memory at any time
└─> Files 47, 48, 49 cached when viewing file 48
└─> File 47 released when switch to file 50

BUT: If GUI holds refs to all 100, they stay cached
└─> This is intentional! For fast batch analysis
```

## Implementation Details

### Reference Counting

**SceneEntry (internal):**
```python
class SceneEntry:
    """Tracks a cached scene with its ref count."""
    def __init__(self, scene, manager, filepath):
        self.scene = scene
        self.manager = manager
        self.filepath = filepath
        self.ref_count = 0
```

**get_scene():**
```python
def get_scene(self, filepath: str, force_reload: bool = False):
    with self._cache_lock:
        # Check cache
        if filepath in self._scenes and not force_reload:
            entry = self._scenes[filepath]
            entry.ref_count += 1
            print(f"📦 Cache HIT: {filepath} (refs: {entry.ref_count})")
            return FBXSceneReference(...)

        # Load new scene
        scene, manager = load_fbx(filepath)
        entry = SceneEntry(scene, manager, filepath)
        entry.ref_count = 1
        self._scenes[filepath] = entry
        return FBXSceneReference(...)
```

**_release_scene():**
```python
def _release_scene(self, filepath: str):
    with self._cache_lock:
        entry = self._scenes[filepath]
        entry.ref_count -= 1

        # Cleanup if no more references
        if entry.ref_count <= 0:
            print(f"🧹 Cleaning up scene (0 refs): {filepath}")
            cleanup_fbx_scene(entry.scene, entry.manager)
            del self._scenes[filepath]
```

### Thread Safety

All scene manager methods use `self._cache_lock` to prevent race conditions when multiple threads access scenes concurrently (e.g., GUI thread + worker thread).

## Testing

### Unit Tests

**Location:** `tests/unit/test_scene_manager.py`

**Coverage:** 22 tests, 83.33% coverage

**Test Classes:**
- `TestFBXSceneReference` - Smart pointer behavior
- `TestSceneManager` - Core functionality
- `TestSceneManagerThreadSafety` - Concurrent access
- `TestSceneManagerIntegration` - Common workflows
- `TestVisualizerSmartCaching` - Smart caching logic

### Integration Tests

**Location:** `tests/integration/test_analysis_pipeline.py`

**Coverage:** 6 tests for scene manager workflows

**Test Classes:**
- `TestAnalysisWorkerSceneManager` - Worker integration
- `TestGUIWorkflowWithSceneManager` - Complete GUI workflows

**Key Tests:**
- Worker gets cache hit when GUI holds ref
- Worker releases scene on error
- Load → Analyze → Clear workflow
- Visualizer and analysis share scene
- Batch processing with no memory accumulation

## Future Enhancements

### Planned Features

1. **Overlay Analysis on Visualizer**
   - Scene manager enables this architecture
   - Analysis data can be shared via scene reference
   - Visualizer can render analysis results in real-time

2. **Acceleration/Jerk Caching**
   - Cache computed derivatives in trajectory
   - Prevent redundant computation
   - Build on scene manager caching pattern

3. **Smart Preloading**
   - Preload next ± 2 files in background
   - Use scene manager to manage preload refs
   - Seamless file switching

## Debugging

### Cache Statistics

```python
stats = scene_manager.get_cache_stats()
print(stats)
# Output:
# {
#     'cached_scenes': 3,
#     'scenes': [
#         {'filepath': 'file1.fbx', 'ref_count': 2},
#         {'filepath': 'file2.fbx', 'ref_count': 1},
#         {'filepath': 'file3.fbx', 'ref_count': 1}
#     ]
# }
```

### Debug Logging

Scene manager prints helpful debug messages:
- `📦 Scene cache HIT: file.fbx (refs: 2)` - Cache hit
- `📂 Loading scene: file.fbx` - New load
- `📉 Scene reference released: file.fbx (refs: 1)` - Release
- `🧹 Cleaning up scene (0 refs): file.fbx` - Destruction

## Best Practices

### ✅ DO:
- Always use context managers when possible (`with scene_ref:`)
- Release references in `finally` blocks for error safety
- Use scene manager for all FBX loading (don't call `load_fbx()` directly)
- Keep references in instance variables for long-lived objects

### ❌ DON'T:
- Don't store scenes directly (store FBXSceneReference instead)
- Don't call `cleanup_fbx_scene()` directly (scene manager handles it)
- Don't forget to release references (causes memory leaks)
- Don't force reload without good reason (breaks caching)

## Performance Impact

### Before Scene Manager:
- Walking animation (100 MB): Loaded 3 times (GUI + Visualizer + Analysis) = 300 MB
- Batch of 10 files: 10 loads × 100 MB = 1 GB

### After Scene Manager:
- Walking animation: Loaded once, shared = 100 MB
- Batch of 10 files with smart caching: ~300 MB (3 files cached at a time)
- Batch of 10 files with GUI refs: All cached for fast analysis

**Memory savings: 66-90% depending on workflow**

## Related Documentation

- `fbx_tool/analysis/scene_manager.py` - Implementation
- `fbx_tool/analysis/fbx_loader.py` - load_fbx() and cleanup_fbx_scene()
- `tests/unit/test_scene_manager.py` - Unit tests
- `tests/integration/test_analysis_pipeline.py` - Integration tests
