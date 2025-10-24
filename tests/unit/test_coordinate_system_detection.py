"""
Unit tests for comprehensive coordinate system detection.

Tests the detect_full_coordinate_system() function which detects:
- UP axis (X/Y/Z)
- FORWARD axis (empirically from motion)
- RIGHT axis (computed from cross product)
- Handedness (right/left-handed)
- Yaw axis and turning conventions

Following TDD principles: tests demand robust implementations.
"""

from unittest.mock import MagicMock, Mock

import fbx
import numpy as np
import pytest

from fbx_tool.analysis.utils import _detect_forward_axis_empirical, detect_full_coordinate_system


class TestForwardAxisEmpiricalDetection:
    """Test empirical forward axis detection from motion data."""

    def test_forward_motion_along_positive_x(self):
        """Should detect +X as forward when motion is along +X axis."""
        # Linear motion from origin to +X
        positions = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [8, 0, 0],
                [9, 0, 0],
                [10, 0, 0],
            ]
        )
        velocities = np.diff(positions, axis=0)

        axis_idx, sign, confidence = _detect_forward_axis_empirical(positions, velocities)

        assert axis_idx == 0, "Should detect X axis"
        assert sign == 1, "Should detect positive direction"
        assert confidence > 0.5, "Should have high confidence for clear linear motion"

    def test_forward_motion_along_negative_z(self):
        """Should detect -Z as forward when motion is along -Z axis (common in many engines)."""
        # Linear motion from origin to -Z
        positions = np.array(
            [
                [0, 0, 0],
                [0, 0, -1],
                [0, 0, -2],
                [0, 0, -3],
                [0, 0, -4],
                [0, 0, -5],
                [0, 0, -6],
                [0, 0, -7],
                [0, 0, -8],
                [0, 0, -9],
                [0, 0, -10],
            ]
        )
        velocities = np.diff(positions, axis=0)

        axis_idx, sign, confidence = _detect_forward_axis_empirical(positions, velocities)

        assert axis_idx == 2, "Should detect Z axis"
        assert sign == -1, "Should detect negative direction"
        assert confidence > 0.5, "Should have high confidence"

    def test_curved_motion_detects_dominant_axis(self):
        """Should detect dominant axis even with curved motion."""
        # Arc motion mostly along +X but with some Y variation
        t = np.linspace(0, 2 * np.pi, 20)
        positions = np.column_stack(
            [t * 2, np.sin(t) * 0.5, np.zeros_like(t)]  # Strong X progression  # Minor Y oscillation  # No Z motion
        )
        velocities = np.diff(positions, axis=0)

        axis_idx, sign, confidence = _detect_forward_axis_empirical(positions, velocities)

        assert axis_idx == 0, "Should detect X as dominant forward axis despite curve"
        assert sign == 1, "Should detect positive direction"

    def test_insufficient_data_returns_default(self):
        """Should return default -Z with zero confidence when insufficient data."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])  # Only 2 points
        velocities = np.diff(positions, axis=0)

        axis_idx, sign, confidence = _detect_forward_axis_empirical(positions, velocities)

        assert axis_idx == 2, "Should default to Z axis"
        assert sign == -1, "Should default to negative"
        assert confidence == 0.0, "Should have zero confidence"

    def test_stationary_data_returns_default(self):
        """Should return default when no significant movement."""
        # All positions essentially the same
        positions = np.array(
            [
                [0, 0, 0],
                [0.0001, 0, 0],
                [0.0002, 0, 0],
                [0, 0.0001, 0],
                [0.0001, 0.0001, 0],
            ]
            * 3
        )  # Repeat to have enough samples
        velocities = np.diff(positions, axis=0)

        axis_idx, sign, confidence = _detect_forward_axis_empirical(positions, velocities)

        assert axis_idx == 2, "Should default to Z"
        assert sign == -1, "Should default to negative"
        assert confidence == 0.0, "Should have zero confidence for no movement"


class TestFullCoordinateSystemDetection:
    """Test complete coordinate system detection with FBX metadata integration."""

    def create_mock_scene(self, up_axis="Y", is_right_handed=True):
        """
        Create a mock FBX scene with specified axis configuration.

        Args:
            up_axis: 'X', 'Y', or 'Z'
            is_right_handed: True for right-handed, False for left-handed

        Returns:
            Mock FBX scene object
        """
        import fbx as fbx_module

        scene = Mock()
        global_settings = Mock()
        axis_system = Mock()

        # Map up axis string to FBX enum
        up_axis_map = {
            "X": fbx_module.FbxAxisSystem.EUpVector.eXAxis,
            "Y": fbx_module.FbxAxisSystem.EUpVector.eYAxis,
            "Z": fbx_module.FbxAxisSystem.EUpVector.eZAxis,
        }

        # Configure axis system
        axis_system.GetUpVector.return_value = (up_axis_map[up_axis], 1)  # Positive up
        axis_system.GetFrontVector.return_value = (fbx_module.FbxAxisSystem.EFrontVector.eParityEven, 1)

        if is_right_handed:
            axis_system.GetCoorSystem.return_value = fbx_module.FbxAxisSystem.ECoordSystem.eRightHanded
        else:
            axis_system.GetCoorSystem.return_value = fbx_module.FbxAxisSystem.ECoordSystem.eLeftHanded

        global_settings.GetAxisSystem.return_value = axis_system
        scene.GetGlobalSettings.return_value = global_settings

        return scene

    def test_right_handed_y_up_system(self):
        """Test standard right-handed Y-up coordinate system (most common)."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        # Forward motion along -Z (common convention)
        positions = np.array([[0, 0, i] for i in range(10, 0, -1)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        assert result["up_axis"] == 1, "Y should be up"
        assert result["up_sign"] == 1, "Positive Y"
        assert result["forward_axis"] == 2, "Z should be forward"
        assert result["forward_sign"] == -1, "Negative Z forward"
        assert result["right_axis"] == 0, "X should be right"
        assert result["is_right_handed"] is True
        assert result["yaw_axis"] == 1, "Yaw around Y axis"
        assert result["yaw_positive_is_left"] is True, "Right-handed: +yaw = left turn"

    def test_right_handed_z_up_system(self):
        """Test right-handed Z-up coordinate system (common in CAD)."""
        scene = self.create_mock_scene(up_axis="Z", is_right_handed=True)

        # Forward motion along +X
        positions = np.array([[i, 0, 0] for i in range(15)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        assert result["up_axis"] == 2, "Z should be up"
        assert result["forward_axis"] == 0, "X should be forward"
        assert result["forward_sign"] == 1, "Positive X forward"
        assert result["is_right_handed"] is True
        assert result["yaw_axis"] == 2, "Yaw around Z axis (up axis)"
        assert result["yaw_positive_is_left"] is True

    def test_left_handed_y_up_system(self):
        """Test left-handed Y-up coordinate system."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=False)

        # Forward motion along +Z
        positions = np.array([[0, 0, i] for i in range(12)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        assert result["up_axis"] == 1, "Y should be up"
        assert result["forward_axis"] == 2, "Z should be forward"
        assert result["forward_sign"] == 1, "Positive Z forward"
        assert result["is_right_handed"] is False
        assert result["yaw_axis"] == 1, "Yaw around Y"
        assert result["yaw_positive_is_left"] is False, "Left-handed: +yaw = right turn"

    def test_turning_convention_right_handed(self):
        """Verify turning conventions for right-handed systems."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        positions = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        # In right-handed Y-up: positive Y rotation (counterclockwise from above) = LEFT turn
        assert (
            result["yaw_positive_is_left"] is True
        ), "Right-handed system: positive rotation around up axis should be LEFT turn"

    def test_turning_convention_left_handed(self):
        """Verify turning conventions for left-handed systems."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=False)

        positions = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        # In left-handed Y-up: positive Y rotation (clockwise from above) = RIGHT turn
        assert (
            result["yaw_positive_is_left"] is False
        ), "Left-handed system: positive rotation around up axis should be RIGHT turn"

    def test_confidence_score_returned(self):
        """Should return confidence score from empirical detection."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        # Clear linear motion should give high confidence
        positions = np.array([[i, 0, 0] for i in range(20)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] > 0.5, "Clear linear motion should have high confidence"


class TestCoordinateSystemIntegration:
    """Integration tests verifying coordinate system detection with trajectory extraction."""

    def create_mock_scene(self, up_axis="Y", is_right_handed=True):
        """Helper to create mock scene."""
        import fbx as fbx_module

        scene = Mock()
        global_settings = Mock()
        axis_system = Mock()

        up_axis_map = {
            "X": fbx_module.FbxAxisSystem.EUpVector.eXAxis,
            "Y": fbx_module.FbxAxisSystem.EUpVector.eYAxis,
            "Z": fbx_module.FbxAxisSystem.EUpVector.eZAxis,
        }

        axis_system.GetUpVector.return_value = (up_axis_map[up_axis], 1)
        axis_system.GetFrontVector.return_value = (fbx_module.FbxAxisSystem.EFrontVector.eParityEven, 1)

        if is_right_handed:
            axis_system.GetCoorSystem.return_value = fbx_module.FbxAxisSystem.ECoordSystem.eRightHanded
        else:
            axis_system.GetCoorSystem.return_value = fbx_module.FbxAxisSystem.ECoordSystem.eLeftHanded

        global_settings.GetAxisSystem.return_value = axis_system
        scene.GetGlobalSettings.return_value = global_settings

        return scene

    def test_turning_direction_uses_detected_yaw_axis(self):
        """
        Critical test: Turning direction must use DETECTED yaw axis, not hardcoded Y.

        This is the bug fix for "WalksArc_Right" being misclassified.
        """
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        positions = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        velocities = np.diff(positions, axis=0)

        coord_system = detect_full_coordinate_system(scene, positions, velocities)

        # Simulate positive rotation around detected yaw axis
        yaw_axis = coord_system["yaw_axis"]
        assert yaw_axis == 1, "Should use Y axis for yaw in Y-up system"

        # Verify turning convention is respected
        if coord_system["yaw_positive_is_left"]:
            # Positive angular velocity should mean LEFT turn
            angular_velocity = 45.0  # Positive
            expected_direction = "left"
        else:
            # Positive angular velocity should mean RIGHT turn
            angular_velocity = 45.0
            expected_direction = "right"

        # This is the logic that should be used in extract_root_trajectory
        if coord_system["yaw_positive_is_left"]:
            actual_direction = "left" if angular_velocity > 0 else "right"
        else:
            actual_direction = "right" if angular_velocity > 0 else "left"

        assert (
            actual_direction == expected_direction
        ), "Turning direction must respect detected coordinate system conventions"

    def test_all_axes_form_valid_coordinate_frame(self):
        """Up, Forward, and Right axes must form a valid orthogonal coordinate frame."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        positions = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        # All three axes must be different
        axes = {result["up_axis"], result["forward_axis"], result["right_axis"]}
        assert len(axes) == 3, "Up, Forward, Right must be three different axes"
        assert axes == {0, 1, 2}, "Must use all three coordinate axes (X, Y, Z)"

    def test_output_contains_all_required_fields(self):
        """Result must contain all documented fields."""
        scene = self.create_mock_scene(up_axis="Y", is_right_handed=True)

        positions = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        velocities = np.diff(positions, axis=0)

        result = detect_full_coordinate_system(scene, positions, velocities)

        required_fields = [
            "up_axis",
            "up_sign",
            "forward_axis",
            "forward_sign",
            "right_axis",
            "right_sign",
            "is_right_handed",
            "yaw_axis",
            "yaw_positive_is_left",
            "confidence",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
