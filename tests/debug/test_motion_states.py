"""Quick test script for motion state detection debug logging"""
import sys
import os
import io

# Redirect stdout/stderr to handle Unicode properly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fbx_tool.analysis.scene_manager import FBXSceneManager
from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
from fbx_tool.analysis.motion_transition_detection import analyze_motion_transitions

fbx_path = r"C:\Users\posne\Downloads\Mixamo\Mixamo\Running\Run Forward Arc Left.fbx"
output_dir = r"output\Run Forward Arc Left"

print("=" * 80)
print(f"Testing motion state detection: {fbx_path}")
print("=" * 80)

# Load scene
scene_manager = FBXSceneManager()
scene = scene_manager.get_scene(fbx_path)

if scene:
    print("\n[1/2] Root Motion Analysis")
    print("-" * 80)
    root_motion_result = analyze_root_motion(scene, output_dir)

    print("\n[2/2] Motion Transition Detection")
    print("-" * 80)
    motion_result = analyze_motion_transitions(scene, output_dir)

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

    # Cleanup
    scene_manager.clear_all_scenes()
else:
    print("❌ Failed to load scene")
