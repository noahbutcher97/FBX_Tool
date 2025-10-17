# 3D Skeleton Viewer - Quick Reference Guide

## Overview
The interactive OpenGL viewer provides real-time animation playback with full control over camera, playback, and display options.

## Getting Started

1. **Load FBX File** - Use GUI "Choose File(s)" or drag & drop
2. **Run Analysis** - Click "Run All Analyses"
3. **Open 3D Viewer** - Click "3D Visualization" button
4. **Play Animation** - Click "Play" or press Space

## Controls Reference

### 🖱️ Mouse Controls

| Action | Control |
|--------|---------|
| Rotate Camera (Orbit) | Left Click + Drag |
| Pan Camera | Right Click + Drag |
| Zoom In/Out | Mouse Wheel |

### ⌨️ Keyboard Shortcuts

#### Animation Control
| Key | Action |
|-----|--------|
| `Space` | Play/Pause animation |
| `←` | Previous frame |
| `→` | Next frame |
| `Home` | Jump to first frame |
| `End` | Jump to last frame |

#### Camera Presets
| Key | View |
|-----|------|
| `R` | Reset to default view |
| `F` | Front view (facing camera) |
| `S` | Side view (profile) |
| `T` | Top view (bird's eye) |

#### Display Toggles
| Key | Toggle |
|-----|--------|
| `G` | Grid on/off |
| `A` | Coordinate axes on/off |
| `W` | Wireframe mode on/off |

### 🎛️ UI Controls

#### Playback Controls
- **Play/Pause Button** - Start/stop animation
- **Frame Slider** - Scrub through timeline
- **Speed Slider** - Adjust playback speed (0.1x - 4.0x)
- **Frame Counter** - Shows current/total frames

#### Display Options (Checkboxes)
- **Grid (G)** - Ground reference grid
- **Axes (A)** - RGB coordinate axes (Red=X, Green=Y, Blue=Z)
- **Wireframe (W)** - Skeleton wireframe rendering

#### Camera Presets (Buttons)
- **Front (F)** - Character facing camera
- **Side (S)** - Profile/lateral view
- **Top (T)** - Overhead/bird's eye view
- **Reset (R)** - Default perspective view

## Visual Reference

### Coordinate Axes Colors
- 🔴 **Red** - X-axis (left/right)
- 🟢 **Green** - Y-axis (up/down)
- 🔵 **Blue** - Z-axis (forward/back)

### Skeleton Rendering
- **Cyan Lines** - Bone connections
- **Red Spheres** - Joint positions

## Tips & Tricks

### Camera Navigation
1. **Reset Often** - Press `R` if you get lost
2. **Use Presets** - Quick standard views with `F`, `S`, `T`
3. **Smooth Orbiting** - Small mouse movements for precise control
4. **Pan to Follow** - Right-click drag to keep character centered

### Animation Analysis
1. **Slow Motion** - Reduce speed to 0.1x-0.5x for detailed observation
2. **Frame-by-Frame** - Use arrow keys to step through poses
3. **Loop Playback** - Animation automatically loops when playing
4. **Speed Up** - Increase to 2x-4x for quick previews

### Display Options
1. **Grid Reference** - Helpful for judging foot placement and height
2. **Axes Orientation** - Verify coordinate system and character facing
3. **Wireframe Mode** - See all bones without occlusion
4. **Toggle All Off** - Clean view by pressing G, A to disable overlays

## Troubleshooting

### Animation Not Playing
- **Check Frame Count** - Verify total_frames > 0
- **Check Speed** - Ensure speed slider not at minimum
- **Press Play** - Animation won't advance when paused

### Character Appears Static
- **Verify Animation** - Check that FBX contains keyframes
- **Stack Selection** - Tool should auto-select "mixamo.com" stack
- **Frame Range** - Use arrow keys to manually test frame changes

### Camera Issues
- **Too Close/Far** - Use mouse wheel or press `R` to reset
- **Upside Down** - Press `R` to restore default orientation
- **Lost Character** - Press `R` then zoom out with mouse wheel

### Performance
- **Slow Playback** - Reduce speed multiplier
- **Choppy Animation** - Try wireframe mode (less rendering)
- **Large Files** - May need to reduce display quality

## Advanced Features

### Custom Camera Angles
1. Rotate to desired angle with left-click drag
2. Adjust elevation for high/low angles
3. Pan to center subject
4. Save mental note or take screenshot

### Analysis Workflow
1. Load animation → Run analysis
2. Open 3D viewer
3. Slow motion playback (0.1x-0.5x)
4. Use frame-by-frame (arrow keys) for key poses
5. Switch camera views (F/S/T) for different perspectives
6. Toggle wireframe to see all bones

### Export Workflow
1. Analyze animation
2. Verify results in 3D viewer
3. Export CSV data from output folder
4. Use dopesheet/joint data in external tools

## Keyboard Shortcut Summary

```
ANIMATION              CAMERA              DISPLAY
Space - Play/Pause     R - Reset           G - Grid
←/→   - Prev/Next     F - Front           A - Axes
Home  - First         S - Side            W - Wireframe
End   - Last          T - Top
```

## Need Help?

- **Documentation**: See [README.md](../README.md)
- **Installation**: See [INSTALL.md](../INSTALL.md)
- **Changes**: See [CHANGELOG.md](../CHANGELOG.md)
- **Issues**: https://github.com/noahbutcher97/FBX_Tool/issues

---

**FBX Tool v1.1.0** - Professional FBX Animation Analysis
