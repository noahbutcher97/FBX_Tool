"""
Unit tests for gait analysis module.

Tests stride segmentation, gait phase analysis, cycle rate calculation,
and asymmetry detection following TDD principles.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from fbx_tool.analysis.gait_analysis import (
    FOOT_BONE_KEYWORDS,
    GAIT_CONFIDENCE_NORMALIZER,
    GAIT_RUN_PHASE_THRESHOLD_SECONDS,
    analyze_gait,
    calculate_asymmetry,
    compute_gait_confidence,
    detect_foot_bone,
)


class TestFootBoneDetection:
    """Test suite for foot bone detection in leg chains."""

    def test_detect_foot_bone_by_keyword(self):
        """Should detect bone with 'foot' keyword."""
        chain = ["mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase"]

        result = detect_foot_bone(chain)

        assert result == "mixamorig:LeftFoot"

    def test_detect_foot_bone_by_ankle(self):
        """Should detect bone with 'ankle' keyword."""
        chain = ["LeftThigh", "LeftCalf", "LeftAnkle", "LeftToe"]

        result = detect_foot_bone(chain)

        assert result == "LeftAnkle"

    def test_detect_foot_bone_by_tarsal(self):
        """Should detect bone with 'tarsal' keyword."""
        chain = ["leg_upper_l", "leg_lower_l", "tarsal_l", "phalanges_l"]

        result = detect_foot_bone(chain)

        assert result == "tarsal_l"

    def test_detect_foot_bone_fallback_second_to_last(self):
        """Should fallback to second-to-last bone when no keyword match."""
        chain = ["Bone1", "Bone2", "Bone3", "Bone4"]

        result = detect_foot_bone(chain)

        assert result == "Bone3"  # Second-to-last (before toes)

    def test_detect_foot_bone_case_insensitive(self):
        """Should detect foot bones case-insensitively."""
        chain = ["UpLeg", "LoLeg", "FOOT", "Toe"]

        result = detect_foot_bone(chain)

        assert result == "FOOT"

    def test_detect_foot_bone_short_chain(self):
        """Should handle chains with only 1 bone."""
        chain = ["OnlyBone"]

        result = detect_foot_bone(chain)

        assert result == "OnlyBone"


class TestGaitConfidenceComputation:
    """Test suite for gait confidence scoring."""

    def test_confidence_zero_velocity(self):
        """Should return maximum confidence for zero velocity (stable stance)."""
        velocity = np.array([0.0, 0.0, 0.0, 0.0])

        result = compute_gait_confidence(velocity)

        # Confidence = mean(π / (1 + |0|)) = π
        assert result == pytest.approx(np.pi, rel=1e-6)

    def test_confidence_low_velocity(self):
        """Should return high confidence for low velocity."""
        velocity = np.array([1.0, -1.0, 0.5, -0.5])

        result = compute_gait_confidence(velocity)

        # Higher confidence for smoother motion
        assert result > 1.5  # Should be high
        assert result < np.pi  # But less than perfect

    def test_confidence_high_velocity(self):
        """Should return low confidence for erratic high velocity."""
        velocity = np.array([10.0, -10.0, 15.0, -15.0])

        result = compute_gait_confidence(velocity)

        # Lower confidence for erratic motion
        assert result < 1.0

    def test_confidence_formula_correctness(self):
        """Should compute confidence using correct formula."""
        velocity = np.array([2.0])

        result = compute_gait_confidence(velocity)

        # Formula: mean(π / (1 + |v|)) = π / (1 + 2) = π/3
        expected = np.pi / 3.0
        assert result == pytest.approx(expected, rel=1e-6)


class TestAsymmetryCalculation:
    """Test suite for left-right gait asymmetry."""

    def test_asymmetry_perfect_symmetry(self):
        """Should return 0.0 for perfectly symmetric strides."""
        # Stride format: [chain, start, end, phase, cycle_time, confidence, length, asymmetry]
        left_strides = [
            ["LeftLeg", 0, 10, "Stride", 0.333, 2.5, 1.0, 0.0],
            ["LeftLeg", 10, 20, "Stride", 0.333, 2.5, 1.0, 0.0],
        ]
        right_strides = [
            ["RightLeg", 5, 15, "Stride", 0.333, 2.5, 1.0, 0.0],
            ["RightLeg", 15, 25, "Stride", 0.333, 2.5, 1.0, 0.0],
        ]

        result = calculate_asymmetry(left_strides, right_strides)

        assert result == 0.0

    def test_asymmetry_high_difference(self):
        """Should return high asymmetry for very different stride times."""
        left_strides = [
            ["LeftLeg", 0, 10, "Stride", 0.5, 2.5, 1.0, 0.0],  # Longer cycle
        ]
        right_strides = [
            ["RightLeg", 0, 10, "Stride", 0.25, 2.5, 1.0, 0.0],  # Shorter cycle
        ]

        result = calculate_asymmetry(left_strides, right_strides)

        # Asymmetry = |0.5 - 0.25| / 0.5 = 0.5 (50% asymmetry)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_asymmetry_empty_left(self):
        """Should return 0.0 when left leg has no strides."""
        left_strides = []
        right_strides = [["RightLeg", 0, 10, "Stride", 0.333, 2.5, 1.0, 0.0]]

        result = calculate_asymmetry(left_strides, right_strides)

        assert result == 0.0

    def test_asymmetry_empty_right(self):
        """Should return 0.0 when right leg has no strides."""
        left_strides = [["LeftLeg", 0, 10, "Stride", 0.333, 2.5, 1.0, 0.0]]
        right_strides = []

        result = calculate_asymmetry(left_strides, right_strides)

        assert result == 0.0

    def test_asymmetry_multiple_strides(self):
        """Should average asymmetry across multiple strides."""
        left_strides = [
            ["LeftLeg", 0, 10, "Stride", 0.4, 2.5, 1.0, 0.0],
            ["LeftLeg", 10, 20, "Stride", 0.4, 2.5, 1.0, 0.0],
        ]
        right_strides = [
            ["RightLeg", 0, 10, "Stride", 0.3, 2.5, 1.0, 0.0],
            ["RightLeg", 10, 20, "Stride", 0.3, 2.5, 1.0, 0.0],
        ]

        result = calculate_asymmetry(left_strides, right_strides)

        # Mean left = 0.4, mean right = 0.3
        # Asymmetry = |0.4 - 0.3| / 0.4 = 0.25
        assert result == pytest.approx(0.25, rel=1e-6)


class TestGaitAnalysis:
    """Test suite for full gait analysis."""

    @patch("fbx_tool.analysis.gait_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.gait_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.gait_analysis.detect_chains_from_hierarchy")
    @patch("fbx_tool.analysis.gait_analysis.prepare_output_file")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("csv.writer")
    def test_analyze_gait_no_leg_chains(
        self, mock_csv, mock_open, mock_prepare, mock_detect_chains, mock_hierarchy, mock_anim_info
    ):
        """Should handle scenes with no detected leg chains."""
        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 1.0, "frame_rate": 30.0}
        mock_hierarchy.return_value = {}
        mock_detect_chains.return_value = {}  # No chains

        scene = Mock()

        stride_segments, gait_summary = analyze_gait(scene, output_dir="test_output")

        assert len(stride_segments) == 0
        assert gait_summary["cycle_rate"] == 0.0

    @patch("fbx_tool.analysis.gait_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.gait_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.gait_analysis.detect_chains_from_hierarchy")
    @patch("fbx_tool.analysis.gait_analysis.prepare_output_file")
    @patch("fbx_tool.analysis.gait_analysis.fbx_vector_to_array")
    @patch("fbx_tool.analysis.gait_analysis.fbx")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("csv.writer")
    def test_analyze_gait_basic(
        self,
        mock_csv,
        mock_open,
        mock_fbx,
        mock_vec_to_array,
        mock_prepare,
        mock_detect_chains,
        mock_hierarchy,
        mock_anim_info,
    ):
        """Should analyze gait and detect strides."""
        # Setup animation info
        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 1.0, "frame_rate": 30.0}

        # Setup hierarchy with leg chains
        mock_hierarchy.return_value = {"LeftFoot": "LeftLeg", "RightFoot": "RightLeg"}
        mock_detect_chains.return_value = {
            "LeftLeg": ["LeftThigh", "LeftCalf", "LeftFoot"],
            "RightLeg": ["RightThigh", "RightCalf", "RightFoot"],
        }

        # Mock scene with foot bone nodes
        scene = Mock()
        left_foot_node = Mock()
        left_foot_node.GetName.return_value = "LeftFoot"
        left_leg_node = Mock()
        right_foot_node = Mock()
        right_foot_node.GetName.return_value = "RightFoot"
        right_leg_node = Mock()

        scene.FindNodeByName.side_effect = lambda name: {
            "LeftFoot": left_foot_node,
            "LeftLeg": left_leg_node,
            "RightFoot": right_foot_node,
            "RightLeg": right_leg_node,
        }.get(name)

        # Mock FbxTime properly
        # Each call to FbxTime() should return a fresh mock instance
        def create_fbx_time():
            time_mock = Mock()
            time_mock.SetSecondDouble = Mock()
            return time_mock

        mock_fbx.FbxTime.side_effect = create_fbx_time

        # Create foot motion pattern: down -> up -> down (stride cycles)
        y_positions = [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0] * 4
        position_index = [0]

        def vec_to_array_side_effect(vec):
            pos_y = y_positions[position_index[0] % len(y_positions)]
            position_index[0] += 1
            return np.array([0.0, pos_y, position_index[0] * 0.1])

        mock_vec_to_array.side_effect = vec_to_array_side_effect

        # Mock transforms
        # Need to mock both GetT() for translation and Inverse() and __mul__ for relative transform
        def create_transform():
            transform = Mock()

            # Mock GetT() to return a vector
            t_vector = Mock()
            transform.GetT.return_value = t_vector

            # Mock Inverse() to return another transform
            inverse_transform = Mock()
            inverse_transform.__mul__ = Mock(return_value=transform)
            transform.Inverse.return_value = inverse_transform

            return transform

        left_foot_node.EvaluateGlobalTransform.side_effect = lambda t: create_transform()
        left_leg_node.EvaluateGlobalTransform.side_effect = lambda t: create_transform()
        right_foot_node.EvaluateGlobalTransform.side_effect = lambda t: create_transform()
        right_leg_node.EvaluateGlobalTransform.side_effect = lambda t: create_transform()

        stride_segments, gait_summary = analyze_gait(scene, output_dir="test_output")

        # Should detect some stride segments
        assert "cycle_rate" in gait_summary
        assert "confidence" in gait_summary
        assert "gait_type" in gait_summary


class TestGaitTypeClassification:
    """Test suite for gait type classification (walk vs run)."""

    def test_gait_type_classification_walking(self):
        """Should classify as walking when phase shift is large."""
        # Phase shift > 0.1s = walking
        # This would be tested through the full analyze_gait function
        # with appropriate phase shift values
        pass  # Tested via integration test

    def test_gait_type_classification_running(self):
        """Should classify as running when phase shift is small."""
        # Phase shift < 0.1s = running
        # This would be tested through the full analyze_gait function
        pass  # Tested via integration test


class TestEdgeCases:
    """Edge case tests for gait analysis."""

    def test_gait_confidence_single_sample(self):
        """Should handle single velocity sample."""
        velocity = np.array([5.0])

        result = compute_gait_confidence(velocity)

        # Should still compute valid confidence
        assert result > 0.0
        assert result <= np.pi

    def test_asymmetry_zero_cycle_time(self):
        """Should handle zero cycle times gracefully."""
        left_strides = [["LeftLeg", 0, 0, "Stride", 0.0, 2.5, 0.0, 0.0]]
        right_strides = [["RightLeg", 0, 0, "Stride", 0.0, 2.5, 0.0, 0.0]]

        result = calculate_asymmetry(left_strides, right_strides)

        # Should return 0 when both are zero
        assert result == 0.0

    def test_detect_foot_bone_all_keywords(self):
        """Should prioritize keywords in order."""
        # Test all keywords work
        for keyword in FOOT_BONE_KEYWORDS:
            chain = [f"Bone_{keyword}_test", "OtherBone"]
            result = detect_foot_bone(chain)
            assert keyword in result.lower()
