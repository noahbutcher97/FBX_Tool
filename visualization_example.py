"""
Visualization Examples

Demonstrates how to use the skeleton visualization features.
"""

import sys
from analysis_modules.fbx_loader import load_fbx_scene
from skeleton_visualizer import SkeletonVisualizer, visualize_skeleton_frame, create_skeleton_animation
from opengl_viewer import launch_skeleton_viewer


def example_matplotlib_single_frame(fbx_path):
    """
    Example: Visualize a single frame with matplotlib.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Matplotlib Single Frame")
    print("="*60)

    # Load FBX
    scene = load_fbx_scene(fbx_path)

    # Visualize frame 0
    visualize_skeleton_frame(scene, frame_idx=0, save_path="skeleton_frame_0.png")

    print("✓ Saved: skeleton_frame_0.png")


def example_matplotlib_animation(fbx_path):
    """
    Example: Create animation with matplotlib.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Matplotlib Animation")
    print("="*60)

    # Load FBX
    scene = load_fbx_scene(fbx_path)

    # Create animation (first 100 frames at 30fps)
    create_skeleton_animation(
        scene,
        output_path="skeleton_animation.mp4",
        start_frame=0,
        end_frame=100,
        fps=30
    )

    print("✓ Saved: skeleton_animation.mp4")


def example_matplotlib_custom(fbx_path):
    """
    Example: Custom visualization with SkeletonVisualizer class.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Matplotlib Custom Visualization")
    print("="*60)

    # Load FBX
    scene = load_fbx_scene(fbx_path)

    # Create visualizer
    visualizer = SkeletonVisualizer(scene)

    # Customize colors
    visualizer.bone_color = 'blue'
    visualizer.joint_color = 'yellow'
    visualizer.joint_size = 100
    visualizer.bone_width = 3

    # Visualize multiple frames
    for frame in [0, 30, 60, 90]:
        visualizer.visualize_frame(frame, save_path=f"custom_frame_{frame}.png")
        print(f"✓ Saved: custom_frame_{frame}.png")


def example_opengl_viewer(fbx_path):
    """
    Example: Interactive OpenGL viewer.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Interactive OpenGL Viewer")
    print("="*60)
    print("Controls:")
    print("  - Left Mouse: Rotate camera")
    print("  - Right Mouse: Pan camera")
    print("  - Mouse Wheel: Zoom")
    print("  - Play/Pause: Animation playback")
    print("  - Slider: Scrub frames")
    print("="*60)

    # Load FBX
    scene = load_fbx_scene(fbx_path)

    # Launch viewer (opens interactive window)
    launch_skeleton_viewer(scene)


def example_batch_visualization(fbx_path, output_dir="visualizations"):
    """
    Example: Batch export frames at key intervals.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Frame Export")
    print("="*60)

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load FBX
    scene = load_fbx_scene(fbx_path)

    # Create visualizer
    visualizer = SkeletonVisualizer(scene)

    # Get total frames
    total_frames = len(next(iter(visualizer.bone_transforms.values())))

    # Export every 10th frame
    for frame in range(0, total_frames, 10):
        save_path = os.path.join(output_dir, f"frame_{frame:04d}.png")
        visualizer.visualize_frame(frame, save_path)
        print(f"✓ Exported frame {frame}")

    print(f"\n✓ All frames saved to: {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualization_example.py <fbx_file> [example_number]")
        print("\nExamples:")
        print("  1 - Single frame (matplotlib)")
        print("  2 - Animation (matplotlib)")
        print("  3 - Custom colors (matplotlib)")
        print("  4 - Interactive viewer (OpenGL)")
        print("  5 - Batch export")
        print("\nDefault: Runs example 4 (OpenGL viewer)")
        sys.exit(1)

    fbx_path = sys.argv[1]
    example = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if example == 1:
        example_matplotlib_single_frame(fbx_path)
    elif example == 2:
        example_matplotlib_animation(fbx_path)
    elif example == 3:
        example_matplotlib_custom(fbx_path)
    elif example == 4:
        example_opengl_viewer(fbx_path)
    elif example == 5:
        example_batch_visualization(fbx_path)
    else:
        print(f"Unknown example: {example}")
