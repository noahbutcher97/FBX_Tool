#!/usr/bin/env python
"""
Quick pipeline test script to verify end-to-end analysis works.
"""

from fbx_tool.analysis import directional_change_detection, root_motion_analysis
from fbx_tool.analysis.scene_manager import get_scene_manager


def test_analysis_pipeline():
    """Test basic analysis pipeline."""
    mgr = get_scene_manager()
    filepath = "./assets/Test/FBX/Change Direction.fbx"

    print(f"\n[TEST] Testing analysis pipeline with: {filepath}")

    with mgr.get_scene(filepath) as scene_ref:
        print("\n[1] Running root motion analysis...")
        result = root_motion_analysis.analyze_root_motion(scene_ref.scene, output_dir="output/test_run")

        print(f"[OK] Root Motion Analysis Complete!")
        print(f"   Root bone: {result['root_bone_name']}")
        print(f"   Total distance: {result['total_distance']:.2f} units")
        print(f"   Mean velocity: {result['mean_velocity']:.2f} units/s")
        print(f"   Dominant direction: {result['dominant_direction']}")

        print("\n[2] Running directional change detection...")
        result2 = directional_change_detection.analyze_directional_changes(
            scene_ref.scene, output_dir="output/test_run"
        )

        print(f"[OK] Directional Change Detection Complete!")
        print(f"   Transitions detected: {result2['total_transitions']}")
        print(f"   Turning events: {result2['total_turning_events']}")

        print(f"\n[OUTPUT] Output files written to: output/test_run/")
        print("\n[SUCCESS] Pipeline test PASSED!")
        return True


if __name__ == "__main__":
    try:
        test_analysis_pipeline()
    except Exception as e:
        print(f"\n[FAIL] Pipeline test FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
