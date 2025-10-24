"""
Integration test for comprehensive FBX analysis pipeline.
Tests all analysis modules including constraint violation detection.
"""

import os
from pathlib import Path

import pytest

from fbx_tool.analysis.chain_analysis import analyze_chains
from fbx_tool.analysis.constraint_violation_detection import analyze_constraint_violations
from fbx_tool.analysis.dopesheet_export import export_dopesheet
from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.foot_contact_analysis import analyze_foot_contacts
from fbx_tool.analysis.gait_analysis import analyze_gait
from fbx_tool.analysis.joint_analysis import analyze_joints
from fbx_tool.analysis.pose_validity_analysis import analyze_pose_validity
from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
from fbx_tool.analysis.scene_manager import get_scene_manager
from fbx_tool.analysis.utils import ensure_output_dir
from fbx_tool.analysis.velocity_analysis import analyze_velocity


@pytest.mark.integration
class TestFullAnalysisPipeline:
    """Integration tests for complete analysis pipeline."""

    def test_comprehensive_analysis_pipeline(self, temp_output_dir):
        """Test all analysis modules work together end-to-end."""
        # Check if test assets exist
        test_fbx = Path("assets/Test/FBX/Female Walk.fbx")
        if not test_fbx.exists():
            pytest.skip("Test FBX file not available")

        fbx_file = str(test_fbx)
        output_dir = temp_output_dir

        results = {}
        scene_manager = get_scene_manager()

        with scene_manager.get_scene(fbx_file) as scene_ref:
            scene = scene_ref.scene

            # Test 1: Metadata extraction
            metadata = get_scene_metadata(scene)
            assert "duration" in metadata
            assert "frame_rate" in metadata
            assert "bone_count" in metadata
            results["metadata"] = metadata

            # Test 2: Dopesheet export
            dopesheet_path = os.path.join(output_dir, "dopesheet.csv")
            export_dopesheet(scene, dopesheet_path)
            assert Path(dopesheet_path).exists()
            results["dopesheet"] = {"path": dopesheet_path}

            # Test 3: Joint analysis
            joint_conf = analyze_joints(scene, output_dir=output_dir)
            assert isinstance(joint_conf, dict)
            results["joints"] = joint_conf

            # Test 4: Chain analysis
            chain_conf = analyze_chains(scene, output_dir=output_dir)
            assert isinstance(chain_conf, dict)
            results["chains"] = chain_conf

            # Test 5: Velocity analysis
            velocity_data = analyze_velocity(scene, output_dir=output_dir)
            assert isinstance(velocity_data, dict)
            results["velocity"] = velocity_data

            # Test 6: Foot contact analysis
            foot_contacts = analyze_foot_contacts(scene, output_dir=output_dir)
            assert isinstance(foot_contacts, dict)
            results["foot_contacts"] = foot_contacts

            # Test 7: Root motion analysis
            root_motion = analyze_root_motion(scene, output_dir=output_dir)
            assert isinstance(root_motion, dict)
            results["root_motion"] = root_motion

            # Test 8: Gait analysis
            stride_segments, gait_summary = analyze_gait(scene, output_dir=output_dir)
            assert isinstance(stride_segments, list)
            assert isinstance(gait_summary, dict)
            results["gait"] = {"segments": stride_segments, "summary": gait_summary}

            # Test 9: Pose validity analysis
            pose_validity = analyze_pose_validity(scene, output_dir=output_dir)
            assert isinstance(pose_validity, dict)
            assert "overall_validity_score" in pose_validity
            results["pose_validity"] = pose_validity

            # Test 10: Constraint violation detection (NEW!)
            constraints = analyze_constraint_violations(scene, output_dir=output_dir)
            assert isinstance(constraints, dict)
            assert "overall_constraint_score" in constraints
            assert "total_chains" in constraints
            assert "ik_violations" in constraints
            assert "hierarchy_violations" in constraints
            assert "curve_discontinuities" in constraints
            results["constraints"] = constraints

        # Verify all modules ran successfully
        assert len(results) == 10, "Not all analysis modules completed"

        # Verify output files exist
        expected_files = [
            "dopesheet.csv",
            "chain_confidence.csv",
            "constraint_summary.csv",
            "ik_chain_violations.csv",
            "hierarchy_violations.csv",
            "curve_discontinuities.csv",
        ]

        for filename in expected_files:
            filepath = Path(output_dir) / filename
            assert filepath.exists(), f"Expected output file not created: {filename}"

    def test_constraint_violation_integration(self, temp_output_dir):
        """Test constraint violation detection specifically."""
        test_fbx = Path("assets/Test/FBX/Female Walk.fbx")
        if not test_fbx.exists():
            pytest.skip("Test FBX file not available")

        scene_manager = get_scene_manager()

        with scene_manager.get_scene(str(test_fbx)) as scene_ref:
            scene = scene_ref.scene

            # Run constraint analysis
            results = analyze_constraint_violations(scene, output_dir=temp_output_dir)

            # Verify structure
            assert "total_chains" in results
            assert "ik_violations" in results
            assert "hierarchy_violations" in results
            assert "curve_discontinuities" in results
            assert "overall_constraint_score" in results

            # Verify score is valid
            assert 0.0 <= results["overall_constraint_score"] <= 1.0

            # Verify CSV files were created
            csv_files = [
                "constraint_summary.csv",
                "ik_chain_violations.csv",
                "hierarchy_violations.csv",
                "curve_discontinuities.csv",
            ]

            for csv_file in csv_files:
                assert (Path(temp_output_dir) / csv_file).exists()
