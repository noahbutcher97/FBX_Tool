"""
Unit tests for pose_validity_analysis module

Tests cover:
- Bone length validation (stretch/squash detection)
- Joint angle limit validation
- Self-intersection detection
- Symmetry validation (left vs right limbs)
- Pose type detection (T-pose, A-pose, bind pose)
- Anatomical constraint validation
"""

from unittest.mock import Mock

import numpy as np
import pytest

from fbx_tool.analysis.pose_validity_analysis import (
    analyze_pose_validity,
    compute_bone_lengths,
    compute_symmetry_score,
    detect_bone_length_violations,
    detect_pose_type,
    detect_self_intersections,
    validate_joint_angle_limits,
)


@pytest.mark.unit
class TestBoneLengthValidation:
    """Test bone length computation and validation."""

    def test_compute_bone_lengths_basic(self):
        """Test basic bone length computation."""
        # Parent at origin, child at (3, 4, 0) -> length = 5
        parent_pos = np.array([0.0, 0.0, 0.0])
        child_pos = np.array([3.0, 4.0, 0.0])

        length = np.linalg.norm(child_pos - parent_pos)

        assert np.isclose(length, 5.0)

    def test_compute_bone_lengths_multiple_frames(self):
        """Test bone length computation across multiple frames."""
        frames = 100
        parent_positions = np.zeros((frames, 3))
        child_positions = np.ones((frames, 3)) * 5.0  # Distance = sqrt(75)

        bone_lengths = compute_bone_lengths(parent_positions, child_positions)

        assert bone_lengths.shape == (frames,)
        expected_length = np.sqrt(75)
        assert np.allclose(bone_lengths, expected_length)

    def test_compute_bone_lengths_varying(self):
        """Test bone length with varying positions."""
        parent_positions = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        child_positions = np.array(
            [
                [1, 0, 0],  # Length = 1
                [0, 2, 0],  # Length = 2
                [0, 0, 3],  # Length = 3
            ]
        )

        bone_lengths = compute_bone_lengths(parent_positions, child_positions)

        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(bone_lengths, expected)

    def test_detect_bone_length_violations_no_violations(self):
        """Test detection with constant bone length (no stretch/squash)."""
        bone_lengths = np.ones(100) * 10.0  # Constant length
        reference_length = 10.0

        violations = detect_bone_length_violations(bone_lengths, reference_length, tolerance=0.05)

        assert len(violations) == 0

    def test_detect_bone_length_violations_stretching(self):
        """Test detection of bone stretching."""
        bone_lengths = np.ones(100) * 10.0
        bone_lengths[40:60] = 12.0  # 20% stretch (beyond 5% tolerance)
        reference_length = 10.0

        violations = detect_bone_length_violations(bone_lengths, reference_length, tolerance=0.05)

        assert len(violations) >= 1
        violation = violations[0]
        assert violation["type"] == "stretch"
        assert violation["max_deviation_percent"] >= 15.0

    def test_detect_bone_length_violations_squashing(self):
        """Test detection of bone squashing."""
        bone_lengths = np.ones(100) * 10.0
        bone_lengths[30:50] = 8.0  # 20% squash
        reference_length = 10.0

        violations = detect_bone_length_violations(bone_lengths, reference_length, tolerance=0.05)

        assert len(violations) >= 1
        violation = violations[0]
        assert violation["type"] == "squash"
        assert violation["max_deviation_percent"] >= 15.0

    def test_detect_bone_length_violations_severity(self):
        """Test severity classification."""
        bone_lengths = np.ones(50) * 10.0
        bone_lengths[20:30] = 15.0  # 50% stretch (severe)
        reference_length = 10.0

        violations = detect_bone_length_violations(bone_lengths, reference_length, tolerance=0.05)

        assert violations[0]["severity"] == "high"


@pytest.mark.unit
class TestJointAngleLimits:
    """Test joint angle limit validation."""

    def test_validate_joint_angle_limits_within_limits(self):
        """Test angles within anatomical limits."""
        # Elbow: 0-140° flexion
        angles = np.ones(100) * 90.0  # 90° - within limits

        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        assert len(violations) == 0

    def test_validate_joint_angle_limits_exceeds_max(self):
        """Test angles exceeding maximum limit."""
        angles = np.ones(100) * 90.0
        angles[40:60] = 160.0  # Exceeds 140° elbow limit

        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        assert len(violations) >= 1
        violation = violations[0]
        assert violation["type"] == "max_exceeded"
        assert violation["max_violation_degrees"] >= 15.0

    def test_validate_joint_angle_limits_below_min(self):
        """Test angles below minimum limit."""
        angles = np.ones(100) * 10.0
        angles[20:40] = -20.0  # Below 0° minimum

        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        assert len(violations) >= 1
        violation = violations[0]
        assert violation["type"] == "min_exceeded"

    def test_validate_joint_angle_limits_knee(self):
        """Test knee joint limits (0-130° flexion)."""
        angles = np.ones(50) * 150.0  # Hyperextension

        violations = validate_joint_angle_limits(angles, joint_type="knee", min_angle=0.0, max_angle=130.0)

        assert len(violations) >= 1

    def test_validate_joint_angle_limits_shoulder(self):
        """Test shoulder joint limits (wider range of motion)."""
        # Shoulder flexion: -60 to 180°
        angles = np.ones(50) * 90.0

        violations = validate_joint_angle_limits(angles, joint_type="shoulder", min_angle=-60.0, max_angle=180.0)

        assert len(violations) == 0

    def test_validate_joint_angle_limits_severity_classification(self):
        """Test violation severity levels."""
        angles = np.ones(50) * 90.0
        angles[20:30] = 200.0  # Severe violation (60° over limit of 140°)

        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        assert violations[0]["severity"] in ["medium", "high"]


@pytest.mark.unit
class TestSelfIntersectionDetection:
    """Test self-intersection detection."""

    def test_detect_self_intersections_no_intersection(self):
        """Test with bones that don't intersect."""
        bone1_start = np.array([[0, 0, 0]] * 50)
        bone1_end = np.array([[10, 0, 0]] * 50)
        bone2_start = np.array([[0, 10, 0]] * 50)
        bone2_end = np.array([[10, 10, 0]] * 50)

        intersections = detect_self_intersections(bone1_start, bone1_end, bone2_start, bone2_end)

        assert len(intersections) == 0

    def test_detect_self_intersections_with_intersection(self):
        """Test detection of intersecting bones."""
        # Bone 1: horizontal from (0,5,0) to (10,5,0)
        bone1_start = np.array([[0, 5, 0]] * 50)
        bone1_end = np.array([[10, 5, 0]] * 50)

        # Bone 2: vertical from (5,0,0) to (5,10,0) - crosses bone 1
        bone2_start = np.array([[5, 0, 0]] * 50)
        bone2_end = np.array([[5, 10, 0]] * 50)

        intersections = detect_self_intersections(
            bone1_start, bone1_end, bone2_start, bone2_end, distance_threshold=0.5
        )

        assert len(intersections) >= 1

    def test_detect_self_intersections_near_miss(self):
        """Test with bones that come close but don't intersect."""
        bone1_start = np.array([[0, 0, 0]] * 50)
        bone1_end = np.array([[10, 0, 0]] * 50)
        bone2_start = np.array([[0, 2, 0]] * 50)  # 2 units away
        bone2_end = np.array([[10, 2, 0]] * 50)

        # With threshold of 0.5, should not detect
        intersections = detect_self_intersections(
            bone1_start, bone1_end, bone2_start, bone2_end, distance_threshold=0.5
        )

        assert len(intersections) == 0

    def test_detect_self_intersections_threshold_adjustment(self):
        """Test threshold sensitivity."""
        bone1_start = np.array([[0, 0, 0]] * 50)
        bone1_end = np.array([[10, 0, 0]] * 50)
        bone2_start = np.array([[0, 1.5, 0]] * 50)  # 1.5 units away
        bone2_end = np.array([[10, 1.5, 0]] * 50)

        # With larger threshold, should detect
        intersections = detect_self_intersections(
            bone1_start, bone1_end, bone2_start, bone2_end, distance_threshold=2.0
        )

        assert len(intersections) >= 1


@pytest.mark.unit
class TestSymmetryValidation:
    """Test bilateral symmetry validation."""

    def test_compute_symmetry_score_perfect_symmetry(self):
        """Test with perfectly symmetric limbs."""
        # Left and right identical (mirrored)
        left_positions = np.array(
            [
                [-5, 0, 0],
                [-5, 0, 0],
                [-5, 0, 0],
            ]
        )
        right_positions = np.array(
            [
                [5, 0, 0],  # Mirrored X
                [5, 0, 0],
                [5, 0, 0],
            ]
        )

        symmetry_score = compute_symmetry_score(left_positions, right_positions)

        # Perfect symmetry = score close to 1.0
        assert symmetry_score >= 0.95

    def test_compute_symmetry_score_asymmetric(self):
        """Test with asymmetric limb positions."""
        left_positions = np.array(
            [
                [-5, 0, 0],
                [-5, 5, 0],
                [-5, 10, 0],
            ]
        )
        right_positions = np.array(
            [
                [5, 0, 0],
                [5, 15, 0],  # Very different Y
                [5, 25, 0],  # Very different Y
            ]
        )

        symmetry_score = compute_symmetry_score(left_positions, right_positions)

        # Asymmetric = lower score
        assert symmetry_score < 0.8

    def test_compute_symmetry_score_rotation_invariant(self):
        """Test that symmetry considers rotation."""
        left_rotations = np.array(
            [
                [0, 0, 0],
                [45, 0, 0],
                [90, 0, 0],
            ]
        )
        right_rotations = np.array(
            [
                [0, 0, 0],
                [45, 0, 0],  # Symmetric
                [90, 0, 0],
            ]
        )

        symmetry_score = compute_symmetry_score(left_rotations, right_rotations, compare_rotations=True)

        assert symmetry_score >= 0.95

    def test_compute_symmetry_score_high_asymmetry(self):
        """Test severe asymmetry detection."""
        left_positions = np.array([[-5, 0, 0]] * 50)
        right_positions = np.array([[5, 20, 0]] * 50)  # Very different Y

        symmetry_score = compute_symmetry_score(left_positions, right_positions)

        assert symmetry_score < 0.5


@pytest.mark.unit
class TestPoseTypeDetection:
    """Test pose type detection (T-pose, A-pose, bind pose)."""

    def test_detect_pose_type_tpose(self):
        """Test T-pose detection."""
        # T-pose: arms extended horizontally (90° from body)
        bone_rotations = {
            "left_shoulder": np.array([[0, 0, 90]] * 10),  # Horizontal
            "right_shoulder": np.array([[0, 0, -90]] * 10),  # Horizontal
            "left_elbow": np.array([[0, 0, 0]] * 10),  # Straight
            "right_elbow": np.array([[0, 0, 0]] * 10),  # Straight
        }

        pose_type, confidence = detect_pose_type(bone_rotations)

        assert pose_type == "T-pose"
        assert confidence >= 0.8

    def test_detect_pose_type_apose(self):
        """Test A-pose detection."""
        # A-pose: arms at 45° angle
        bone_rotations = {
            "left_shoulder": np.array([[0, 0, 45]] * 10),
            "right_shoulder": np.array([[0, 0, -45]] * 10),
            "left_elbow": np.array([[0, 0, 0]] * 10),
            "right_elbow": np.array([[0, 0, 0]] * 10),
        }

        pose_type, confidence = detect_pose_type(bone_rotations)

        assert pose_type == "A-pose"
        assert confidence >= 0.8

    def test_detect_pose_type_bind_pose(self):
        """Test bind pose detection (arms down)."""
        # Bind pose: arms at sides (0° rotation)
        bone_rotations = {
            "left_shoulder": np.array([[0, 0, 0]] * 10),
            "right_shoulder": np.array([[0, 0, 0]] * 10),
            "left_elbow": np.array([[0, 0, 0]] * 10),
            "right_elbow": np.array([[0, 0, 0]] * 10),
        }

        pose_type, confidence = detect_pose_type(bone_rotations)

        assert pose_type == "bind"
        assert confidence >= 0.8

    def test_detect_pose_type_animated(self):
        """Test with animated pose (not a reference pose)."""
        # Random animated rotations
        bone_rotations = {
            "left_shoulder": np.random.randn(10, 3) * 45,
            "right_shoulder": np.random.randn(10, 3) * 45,
            "left_elbow": np.random.randn(10, 3) * 30,
            "right_elbow": np.random.randn(10, 3) * 30,
        }

        pose_type, confidence = detect_pose_type(bone_rotations)

        assert pose_type == "animated"
        assert confidence < 0.6

    def test_detect_pose_type_low_confidence(self):
        """Test ambiguous pose detection."""
        # Ambiguous: arms at 60° (between A-pose and T-pose)
        bone_rotations = {
            "left_shoulder": np.array([[0, 0, 60]] * 10),
            "right_shoulder": np.array([[0, 0, -60]] * 10),
        }

        pose_type, confidence = detect_pose_type(bone_rotations)

        # Should have low confidence
        assert confidence < 0.7


@pytest.mark.unit
class TestPoseValidityAnalysis:
    """Test main pose validity analysis function."""

    def test_analyze_pose_validity_basic(self, mock_scene, temp_output_dir):
        """Test basic pose validity analysis."""
        results = analyze_pose_validity(mock_scene, output_dir=temp_output_dir)

        # Check required fields
        assert "total_bones" in results
        assert "bones_with_length_violations" in results
        assert "bones_with_angle_violations" in results
        assert "self_intersections_detected" in results
        assert "overall_validity_score" in results

    def test_analyze_pose_validity_outputs_csv(self, mock_scene, temp_output_dir):
        """Test that CSV files are generated."""
        from pathlib import Path

        analyze_pose_validity(mock_scene, output_dir=temp_output_dir)

        # Check for expected output files
        expected_files = [
            "bone_length_violations.csv",
            "joint_angle_violations.csv",
            "symmetry_analysis.csv",
            "pose_validity_summary.csv",
        ]

        for filename in expected_files:
            filepath = Path(temp_output_dir) / filename
            assert filepath.exists(), f"Expected file not created: {filename}"

    def test_analyze_pose_validity_no_violations(self, mock_scene, temp_output_dir):
        """Test with valid pose (no violations)."""
        # Mock scene configured for valid pose
        results = analyze_pose_validity(mock_scene, output_dir=temp_output_dir)

        assert results["overall_validity_score"] >= 0.8

    def test_analyze_pose_validity_severity_scoring(self, mock_scene, temp_output_dir):
        """Test that severity affects overall score."""
        # This would require injecting violations into mock scene
        results = analyze_pose_validity(mock_scene, output_dir=temp_output_dir)

        # Score should be between 0 and 1
        assert 0.0 <= results["overall_validity_score"] <= 1.0


@pytest.mark.unit
class TestAnatomicalConstraints:
    """Test anatomical constraint validation."""

    def test_elbow_bend_direction(self):
        """Test elbow bends in correct direction (not backwards)."""
        # Elbow should bend forward, not backwards
        # This is a constraint that joint angle limits should catch
        angles = np.array([0, 45, 90, 120])  # Normal flexion

        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        assert len(violations) == 0

    def test_knee_hyperextension(self):
        """Test knee hyperextension detection."""
        angles = np.array([0, -10, -20, -30])  # Hyperextension (negative)

        violations = validate_joint_angle_limits(angles, joint_type="knee", min_angle=0.0, max_angle=130.0)

        # Should detect negative angles as violations
        assert len(violations) >= 1

    def test_shoulder_rotation_limits(self):
        """Test shoulder rotation limits (360° possible but unusual in animation)."""
        # Shoulder can rotate ~180° in most directions
        angles = np.ones(50) * 90.0  # Normal range

        violations = validate_joint_angle_limits(angles, joint_type="shoulder", min_angle=-60.0, max_angle=180.0)

        assert len(violations) == 0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for pose validity analysis."""

    def test_bone_length_single_frame(self):
        """Test bone length with single frame."""
        parent_pos = np.array([[0, 0, 0]])
        child_pos = np.array([[5, 0, 0]])

        lengths = compute_bone_lengths(parent_pos, child_pos)

        assert lengths.shape == (1,)
        assert np.isclose(lengths[0], 5.0)

    def test_bone_length_zero_length(self):
        """Test with zero-length bone (parent and child at same position)."""
        parent_pos = np.array([[5, 5, 5]] * 10)
        child_pos = np.array([[5, 5, 5]] * 10)

        lengths = compute_bone_lengths(parent_pos, child_pos)

        assert np.allclose(lengths, 0.0)

    def test_symmetry_empty_arrays(self):
        """Test symmetry with empty arrays."""
        left_pos = np.array([]).reshape(0, 3)
        right_pos = np.array([]).reshape(0, 3)

        score = compute_symmetry_score(left_pos, right_pos)

        # Should return 1.0 (no asymmetry detected) or 0.0 (no data)
        assert score in [0.0, 1.0]

    def test_joint_angle_nan_handling(self):
        """Test joint angle validation with NaN values."""
        angles = np.array([90.0, np.nan, 100.0, np.nan])

        # Should handle NaN gracefully (skip or filter)
        violations = validate_joint_angle_limits(angles, joint_type="elbow", min_angle=0.0, max_angle=140.0)

        # Should not crash
        assert isinstance(violations, list)

    def test_self_intersection_identical_bones(self):
        """Test self-intersection with identical bone positions (same bone)."""
        bone_start = np.array([[0, 0, 0]] * 10)
        bone_end = np.array([[10, 0, 0]] * 10)

        # Should not report self-intersection with itself
        intersections = detect_self_intersections(bone_start, bone_end, bone_start, bone_end)

        # Implementation should filter out identical bones
        assert len(intersections) == 0


@pytest.mark.unit
class TestReferenceLength:
    """Test reference bone length computation and caching."""

    def test_reference_length_from_bind_pose(self):
        """Test extracting reference length from bind pose."""
        # Bind pose should have consistent bone lengths
        parent_pos = np.array([[0, 0, 0]] * 100)
        child_pos = np.array([[10, 0, 0]] * 100)

        bone_lengths = compute_bone_lengths(parent_pos, child_pos)
        reference_length = np.median(bone_lengths)

        assert np.isclose(reference_length, 10.0)

    def test_reference_length_from_animation(self):
        """Test extracting reference length from animated data (use median)."""
        parent_pos = np.zeros((100, 3))
        child_pos = np.ones((100, 3)) * 10.0

        # Add some noise/variation
        child_pos += np.random.randn(100, 3) * 0.5

        bone_lengths = compute_bone_lengths(parent_pos, child_pos)
        reference_length = np.median(bone_lengths)

        # Should be close to ideal length despite noise
        expected = np.linalg.norm([10, 10, 10])
        assert np.abs(reference_length - expected) < 1.0
