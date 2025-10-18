"""
FBX Scene Manager with Reference Counting

Manages FBX scene lifecycle to prevent memory leaks while allowing
scene sharing between analysis and visualization.

Key features:
- Reference counting: tracks how many "owners" need a scene
- Automatic cleanup: destroys scene when ref count reaches 0
- Thread-safe: uses locks for concurrent access
- Cache support: reuses scenes for the same file path

Usage:
    # Get a scene (increments ref count)
    scene_ref = scene_manager.get_scene("path/to/file.fbx")

    # Use the scene
    scene = scene_ref.scene
    # ... do work ...

    # Release when done (decrements ref count, auto-cleans if 0)
    scene_ref.release()

    # Or use context manager (auto-releases)
    with scene_manager.get_scene("path/to/file.fbx") as scene_ref:
        scene = scene_ref.scene
        # ... do work ...
    # Automatically released here
"""

import threading
from typing import Dict, Optional

from fbx_tool.analysis.fbx_loader import cleanup_fbx_scene, load_fbx


class FBXSceneReference:
    """
    Reference to an FBX scene with automatic cleanup.

    Acts as a smart pointer that tracks when the scene is no longer needed.
    """

    def __init__(self, scene, manager, filepath, scene_manager):
        self.scene = scene
        self.manager = manager
        self.filepath = filepath
        self._scene_manager = scene_manager
        self._released = False

    def release(self):
        """Release this reference. Scene cleaned up when all references released."""
        if not self._released:
            self._scene_manager._release_scene(self.filepath)
            self._released = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto release."""
        self.release()
        return False

    def __del__(self):
        """Destructor - ensure cleanup even if release() not called."""
        if not self._released:
            self.release()


class SceneEntry:
    """Internal: tracks a cached scene with its ref count."""

    def __init__(self, scene, manager, filepath):
        self.scene = scene
        self.manager = manager
        self.filepath = filepath
        self.ref_count = 0


class FBXSceneManager:
    """
    Global scene manager with reference counting and automatic cleanup.

    Singleton pattern - use get_instance() to access.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize scene cache and locks."""
        self._scenes: Dict[str, SceneEntry] = {}
        self._cache_lock = threading.Lock()

    def get_scene(self, filepath: str, force_reload: bool = False) -> FBXSceneReference:
        """
        Get a scene reference (loads if needed, increments ref count).

        Args:
            filepath: Path to FBX file
            force_reload: If True, reload even if cached

        Returns:
            FBXSceneReference: Reference to the scene (call .release() when done)
        """
        with self._cache_lock:
            # Check if already cached and not forcing reload
            if filepath in self._scenes and not force_reload:
                entry = self._scenes[filepath]
                entry.ref_count += 1
                print(f"📦 Scene cache HIT: {filepath} (refs: {entry.ref_count})")
                return FBXSceneReference(entry.scene, entry.manager, filepath, self)

            # Need to load
            if filepath in self._scenes and force_reload:
                # Force reload - cleanup old version first
                print(f"🔄 Force reloading scene: {filepath}")
                old_entry = self._scenes[filepath]
                if old_entry.ref_count > 0:
                    print(f"⚠️  Warning: Force reloading scene with {old_entry.ref_count} active references!")
                cleanup_fbx_scene(old_entry.scene, old_entry.manager)
                del self._scenes[filepath]

            # Load new scene
            print(f"📂 Loading scene: {filepath}")
            scene, manager = load_fbx(filepath)

            # Cache it
            entry = SceneEntry(scene, manager, filepath)
            entry.ref_count = 1
            self._scenes[filepath] = entry

            print(f"✓ Scene loaded and cached: {filepath} (refs: 1)")
            return FBXSceneReference(scene, manager, filepath, self)

    def _release_scene(self, filepath: str):
        """
        Internal: Release a scene reference (decrements ref count, cleans up if 0).

        Args:
            filepath: Path to FBX file
        """
        with self._cache_lock:
            if filepath not in self._scenes:
                print(f"⚠️  Warning: Attempted to release non-cached scene: {filepath}")
                return

            entry = self._scenes[filepath]
            entry.ref_count -= 1

            print(f"📉 Scene reference released: {filepath} (refs: {entry.ref_count})")

            # Cleanup if no more references
            if entry.ref_count <= 0:
                print(f"🧹 Cleaning up scene (0 refs): {filepath}")
                cleanup_fbx_scene(entry.scene, entry.manager)
                del self._scenes[filepath]

    def cleanup_all(self):
        """
        Force cleanup of all cached scenes.

        WARNING: Only call on application shutdown or when certain no scenes are in use!
        """
        with self._cache_lock:
            print(f"🧹 Cleaning up all cached scenes ({len(self._scenes)} total)")
            for filepath, entry in list(self._scenes.items()):
                if entry.ref_count > 0:
                    print(f"⚠️  Warning: Cleaning up scene with {entry.ref_count} active references: {filepath}")
                cleanup_fbx_scene(entry.scene, entry.manager)
            self._scenes.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics for debugging."""
        with self._cache_lock:
            stats = {"cached_scenes": len(self._scenes), "scenes": []}
            for filepath, entry in self._scenes.items():
                stats["scenes"].append({"filepath": filepath, "ref_count": entry.ref_count})
            return stats


# Global singleton instance
def get_scene_manager() -> FBXSceneManager:
    """Get the global scene manager instance."""
    return FBXSceneManager.get_instance()
