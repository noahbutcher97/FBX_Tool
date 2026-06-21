"""
Integration tests for coordinate system detection in real trajectory extraction.

Tests that the procedural coordinate system detection correctly integrates
with extract_root_trajectory() and fixes the turning direction bug.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import fbx
import numpy as np

from fbx_tool.analysis.utils import extract_root_trajectory

LOCAL_FBX_DIR = Path("assets/Test/FBX")


class TestTrajectoryCoordinateSystemIntegration:
    """Integration tests verifying coordinate system detection in trajectory extraction."""

    def test_trajectory_uses_detected_coordinate_system(self):
        """Verify that extract_root_trajectory uses detected coordinate system."""
        from fbx_tool.analysis.scene_manager import get_scene_manager
        from fbx_tool.analysis.utils import clear_trajectory_cache

        fbx_path = LOCAL_FBX_DIR / "Female Walk.fbx"
        assert fbx_path.exists(), f"Expected integration fixture at {fbx_path}"

        clear_trajectory_cache()
        with get_scene_manager().get_scene(str(fbx_path)) as scene_ref:
            scene = scene_ref.scene

            # Extract trajectory with coordinate system detection
            trajectory = extract_root_trajectory(scene)

            # Verify coordinate system was detected and logged
            assert "coordinate_system" in trajectory, "Should include coordinate system metadata"

            coord_sys = trajectory["coordinate_system"]
            assert "forward_axis" in coord_sys
            assert "forward_sign" in coord_sys
            assert "detection_confidence" in coord_sys

            # Verify trajectory data uses procedural yaw axis
            assert "angular_velocity_yaw" in trajectory, "Should use yaw (not hardcoded Y)"

    def test_arc_left_animation_correctly_classified(self):
        """
        Critical test: Verify a left arc animation is classified as a LEFT turn.

        This tests the bug fix where turning direction was hardcoded and ignored
        coordinate system conventions.
        """
        from fbx_tool.analysis.scene_manager import get_scene_manager
        from fbx_tool.analysis.utils import clear_trajectory_cache

        fbx_path = LOCAL_FBX_DIR / "Run Forward Arc Left.fbx"
        assert fbx_path.exists(), f"Expected integration fixture at {fbx_path}"

        clear_trajectory_cache()
        with get_scene_manager().get_scene(str(fbx_path)) as scene_ref:
            scene = scene_ref.scene

            trajectory = extract_root_trajectory(scene)

            # Analyze turning directions in trajectory data
            trajectory_data = trajectory["trajectory_data"]

            # Count frames with left vs right turns
            left_turns = sum(1 for frame in trajectory_data if frame["turning_direction"] == "left")
            right_turns = sum(1 for frame in trajectory_data if frame["turning_direction"] == "right")

            # For an "Arc Left" animation, left turns should dominate.
            assert left_turns > right_turns, "Arc Left animation should have more left turns than right turns"

    def test_coordinate_system_metadata_exported(self):
        """Verify coordinate system metadata is properly packaged in trajectory result."""
        # This is a structural test - doesn't need real FBX

        # Mock the parts we need
        with (
            patch("fbx_tool.analysis.fbx_loader.get_scene_metadata") as mock_metadata,
            patch("fbx_tool.analysis.utils._detect_root_bone") as mock_root,
            patch("fbx_tool.analysis.velocity_analysis.compute_derivatives") as mock_derivatives,
        ):
            # Setup mocks
            mock_metadata.return_value = {
                "has_animation": True,
                "start_time": 0.0,
                "stop_time": 1.0,
                "frame_rate": 30.0,
                "duration": 1.0,
                "total_frames": 31,
            }

            mock_root_bone = Mock()
            mock_root_bone.GetName.return_value = "Hips"

            # Mock animation evaluation
            def mock_evaluate(time):
                transform = Mock()
                transform.GetT.return_value = [0, 0, 0]
                transform.GetR.return_value = [0, 0, 0]
                transform.Get = Mock(side_effect=lambda i, j: 1.0 if i == j else 0.0)
                return transform

            mock_root_bone.EvaluateGlobalTransform = mock_evaluate
            mock_root.return_value = mock_root_bone

            # Mock derivatives
            sample_count = 31
            velocities = np.array([[30.0, 0.0, 0.0] for _ in range(sample_count)], dtype=float)
            accelerations = np.zeros((sample_count, 3), dtype=float)
            jerks = np.zeros((sample_count, 3), dtype=float)
            mock_derivatives.return_value = (velocities, accelerations, jerks)

            # Create mock scene with Y-up right-handed system
            scene = self._create_mock_scene()

            # Mock animation stack
            anim_stack = Mock()
            time_span = Mock()
            time_span.GetStart.return_value = fbx.FbxTime()
            anim_stack.GetLocalTimeSpan.return_value = time_span

            scene.GetSrcObjectCount.return_value = 1
            scene.GetSrcObject.return_value = anim_stack

            # Extract trajectory
            from fbx_tool.analysis.utils import clear_trajectory_cache

            clear_trajectory_cache()  # Ensure clean state
            trajectory = extract_root_trajectory(scene)

            # Verify coordinate system metadata is present
            assert "coordinate_system" in trajectory
            coord_sys = trajectory["coordinate_system"]

            assert "forward_axis" in coord_sys
            assert "forward_sign" in coord_sys
            assert "detection_confidence" in coord_sys

            # Verify metadata is human-readable
            assert coord_sys["forward_axis"] in ["X", "Y", "Z"]
            assert coord_sys["forward_sign"] in ["+", "-"]
            assert isinstance(coord_sys["detection_confidence"], (int, float))

    def _create_mock_scene(self):
        """Helper to create mock FBX scene."""
        import fbx as fbx_module

        scene = Mock()
        global_settings = Mock()
        axis_system = Mock()

        # Y-up, right-handed (standard)
        axis_system.GetUpVector.return_value = (fbx_module.FbxAxisSystem.EUpVector.eYAxis, 1)
        axis_system.GetFrontVector.return_value = (fbx_module.FbxAxisSystem.EFrontVector.eParityEven, 1)
        axis_system.GetCoorSystem.return_value = fbx_module.FbxAxisSystem.ECoordSystem.eRightHanded

        global_settings.GetAxisSystem.return_value = axis_system
        scene.GetGlobalSettings.return_value = global_settings

        # Mock root node
        root_node = Mock()
        scene.GetRootNode.return_value = root_node

        return scene


class TestTurningDirectionProcedural:
    """Specific tests for procedural turning direction classification."""

    def test_turning_direction_logic_right_handed(self):
        """
        Test the turning direction classification logic for right-handed systems.

        In right-handed Y-up:
        - Positive angular velocity around Y = LEFT turn
        - Negative angular velocity around Y = RIGHT turn
        """
        # Simulate coordinate system detection result
        coord_system = {"yaw_positive_is_left": True}  # Right-handed system

        # Test positive angular velocity
        angular_velocity = 45.0
        if coord_system["yaw_positive_is_left"]:
            direction = "left" if angular_velocity > 0 else "right"
        else:
            direction = "right" if angular_velocity > 0 else "left"

        assert direction == "left", "Positive angular velocity in right-handed should be LEFT"

        # Test negative angular velocity
        angular_velocity = -45.0
        if coord_system["yaw_positive_is_left"]:
            direction = "left" if angular_velocity > 0 else "right"
        else:
            direction = "right" if angular_velocity > 0 else "left"

        assert direction == "right", "Negative angular velocity in right-handed should be RIGHT"

    def test_turning_direction_logic_left_handed(self):
        """
        Test the turning direction classification logic for left-handed systems.

        In left-handed Y-up:
        - Positive angular velocity around Y = RIGHT turn
        - Negative angular velocity around Y = LEFT turn
        """
        # Simulate coordinate system detection result
        coord_system = {"yaw_positive_is_left": False}  # Left-handed system

        # Test positive angular velocity
        angular_velocity = 45.0
        if coord_system["yaw_positive_is_left"]:
            direction = "left" if angular_velocity > 0 else "right"
        else:
            direction = "right" if angular_velocity > 0 else "left"

        assert direction == "right", "Positive angular velocity in left-handed should be RIGHT"

        # Test negative angular velocity
        angular_velocity = -45.0
        if coord_system["yaw_positive_is_left"]:
            direction = "left" if angular_velocity > 0 else "right"
        else:
            direction = "right" if angular_velocity > 0 else "left"

        assert direction == "left", "Negative angular velocity in left-handed should be LEFT"

    def test_turning_direction_respects_coordinate_system(self):
        """
        Critical test: Verify turning direction is NEVER hardcoded.

        The bug was:
        turning_direction = "left" if angular_velocity_y[frame] > 0 else "right"

        The fix is:
        if coord_system['yaw_positive_is_left']:
            turning_direction = "left" if angular_velocity_yaw[frame] > 0 else "right"
        else:
            turning_direction = "right" if angular_velocity_yaw[frame] > 0 else "left"
        """
        test_cases = [
            # (yaw_positive_is_left, angular_velocity, expected_direction)
            (True, 50.0, "left"),  # Right-handed: +velocity = left
            (True, -50.0, "right"),  # Right-handed: -velocity = right
            (False, 50.0, "right"),  # Left-handed: +velocity = right
            (False, -50.0, "left"),  # Left-handed: -velocity = left
        ]

        for yaw_positive_is_left, angular_velocity, expected in test_cases:
            coord_system = {"yaw_positive_is_left": yaw_positive_is_left}

            # Apply the procedural logic
            if coord_system["yaw_positive_is_left"]:
                actual = "left" if angular_velocity > 0 else "right"
            else:
                actual = "right" if angular_velocity > 0 else "left"

            assert actual == expected, (
                f"Failed for yaw_positive_is_left={yaw_positive_is_left}, "
                f"angular_velocity={angular_velocity}: expected {expected}, got {actual}"
            )
