"""Integration test for the comprehensive FBX analysis pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fbx_tool.analysis.chain_analysis import analyze_chains
from fbx_tool.analysis.constraint_violation_detection import analyze_constraint_violations
from fbx_tool.analysis.directional_change_detection import analyze_directional_changes
from fbx_tool.analysis.dopesheet_export import export_dopesheet
from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.foot_contact_analysis import analyze_foot_contacts
from fbx_tool.analysis.gait_analysis import analyze_gait
from fbx_tool.analysis.joint_analysis import analyze_joints
from fbx_tool.analysis.motion_classification import generate_motion_summary
from fbx_tool.analysis.motion_transition_detection import analyze_motion_transitions
from fbx_tool.analysis.pose_validity_analysis import analyze_pose_validity
from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
from fbx_tool.analysis.scene_manager import get_scene_manager
from fbx_tool.analysis.temporal_segmentation import analyze_temporal_segmentation
from fbx_tool.analysis.utils import ensure_output_dir
from fbx_tool.analysis.velocity_analysis import analyze_velocity


@pytest.mark.integration
class TestFullAnalysisPipeline:
    """Integration tests for complete analysis pipeline."""

    def test_cli_module_import_has_no_console_side_effects(self):
        """Importing the CLI module should not rewrite process stdout or stderr."""
        script_path = Path("examples/run_analysis.py").resolve()
        repo_root = script_path.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        probe = (
            "import importlib, sys; "
            "stdout = sys.stdout; stderr = sys.stderr; "
            "module = importlib.import_module('examples.run_analysis'); "
            "assert hasattr(module, 'run_analysis'); "
            "raise SystemExit(0 if sys.stdout is stdout and sys.stderr is stderr else 1)"
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        assert result.returncode == 0, result.stderr or result.stdout

    def test_cli_pipeline_writes_advanced_analysis_outputs(self, tmp_path):
        """CLI pipeline should generate the same advanced analysis outputs exposed by the GUI."""
        test_fbx = Path("assets/Test/FBX/Female Walk.fbx").resolve()
        if not test_fbx.exists():
            pytest.skip("Test FBX file not available")

        script_path = Path("examples/run_analysis.py").resolve()
        repo_root = script_path.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            [sys.executable, str(script_path), str(test_fbx)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "[11/14] Detecting directional changes" in result.stdout
        assert "[12/14] Detecting motion transitions" in result.stdout
        assert "[13/14] Creating temporal segmentation" in result.stdout
        assert "[14/14] Generating motion summary" in result.stdout
        assert "Modules completed: 14/14" in result.stdout
        assert "module(s) failed" not in result.stdout

        output_dir = tmp_path / "output" / test_fbx.stem
        assert not (output_dir / "errors.log").exists()
        expected_outputs = [
            "movement_segments.csv",
            "motion_states.csv",
            "temporal_segments.csv",
            "segment_hierarchy.csv",
            "motion_summary.txt",
            "animation_metadata.json",
            "segment_descriptions.csv",
            "motion_classification.json",
        ]
        for filename in expected_outputs:
            assert (output_dir / filename).exists(), f"Expected advanced output not created: {filename}"

        metadata = json.loads((output_dir / "animation_metadata.json").read_text())
        classification = json.loads((output_dir / "motion_classification.json").read_text())
        assert metadata["statistics"]["segment_count"] > 0
        assert metadata["classification"] == classification

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

            # Test 11: Directional change detection
            directional_changes = analyze_directional_changes(scene, output_dir=output_dir)
            assert isinstance(directional_changes, dict)
            assert "movement_segments" in directional_changes
            assert "turning_events" in directional_changes
            results["directional_changes"] = directional_changes

            # Test 12: Motion transition detection
            motion_transitions = analyze_motion_transitions(scene, output_dir=output_dir)
            assert isinstance(motion_transitions, dict)
            assert "motion_states" in motion_transitions
            assert "transitions" in motion_transitions
            results["motion_transitions"] = motion_transitions

            # Test 13: Temporal segmentation
            motion_states = motion_transitions["motion_states"]
            movement_segments = directional_changes["movement_segments"]
            frame_rate = metadata.get("frame_rate", 30.0)
            duration = metadata.get("duration", 0)
            total_frames = int(duration * frame_rate) + 1 if duration else 1
            if not motion_states:
                motion_states = [
                    {
                        "start_frame": 0,
                        "end_frame": total_frames - 1,
                        "duration_frames": total_frames,
                        "duration_seconds": duration,
                        "motion_state": "continuous",
                    }
                ]
            if not movement_segments:
                movement_segments = [
                    {
                        "start_frame": 0,
                        "end_frame": total_frames - 1,
                        "direction": "unknown",
                    }
                ]

            temporal_segmentation = analyze_temporal_segmentation(
                motion_states,
                movement_segments,
                directional_changes["turning_events"],
                frame_rate,
                output_dir=output_dir,
            )
            assert isinstance(temporal_segmentation, dict)
            assert temporal_segmentation["segments_count"] > 0
            results["temporal_segmentation"] = temporal_segmentation

            # Test 14: Motion summary
            motion_summary = generate_motion_summary(
                segments=temporal_segmentation["segments"],
                transitions=temporal_segmentation["transitions"],
                turning_events=directional_changes["turning_events"],
                root_motion_summary=root_motion,
                gait_summary=gait_summary,
                output_dir=output_dir,
            )
            assert isinstance(motion_summary, dict)
            assert "classification" in motion_summary
            results["motion_summary"] = motion_summary

        # Verify all modules ran successfully
        assert len(results) == 14, "Not all analysis modules completed"

        # Verify output files exist
        expected_files = [
            "dopesheet.csv",
            "chain_confidence.csv",
            "constraint_summary.csv",
            "ik_chain_violations.csv",
            "hierarchy_violations.csv",
            "curve_discontinuities.csv",
            "movement_segments.csv",
            "motion_states.csv",
            "temporal_segments.csv",
            "segment_hierarchy.csv",
            "motion_summary.txt",
            "animation_metadata.json",
            "segment_descriptions.csv",
            "motion_classification.json",
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
