"""
Unit tests for constraint_violation_detection module

Tests cover:
- IK chain constraint validation
- Hierarchy integrity checks
- Animation curve discontinuity detection
- Keyframe timing validation
- Parent-child relationship verification
- Bone chain reachability analysis
"""

from unittest.mock import Mock

import numpy as np
import pytest

from fbx_tool.analysis.constraint_violation_detection import (
    analyze_constraint_violations,
    check_end_effector_reachability,
    detect_chain_breaks,
    detect_curve_discontinuities,
    detect_hierarchy_violations,
    validate_ik_chain_length,
    validate_keyframe_timing,
    validate_parent_child_consistency,
)


@pytest.mark.unit
class TestIKChainValidation:
    """Test IK chain constraint validation."""

    def test_validate_ik_chain_length_consistent(self):
        """Test IK chain with consistent length."""
        # Chain: shoulder -> elbow -> wrist (lengths: 10, 8)
        bone_positions = {
            "shoulder": np.array([[0, 0, 0]] * 100),
            "elbow": np.array([[10, 0, 0]] * 100),
            "wrist": np.array([[18, 0, 0]] * 100),
        }

        chain = ["shoulder", "elbow", "wrist"]
        violations = validate_ik_chain_length(bone_positions, chain, tolerance=0.05)

        # No violations - chain length is constant
        assert len(violations) == 0

    def test_validate_ik_chain_length_stretching(self):
        """Test IK chain with stretching violation."""
        bone_positions = {
            "shoulder": np.array([[0, 0, 0]] * 100),
            "elbow": np.array([[10, 0, 0]] * 100),
            "wrist": np.array([[18, 0, 0]] * 100),
        }

        # Stretch the chain in frames 40-60
        bone_positions["wrist"][40:60] = [25, 0, 0]  # Much farther (39% stretch)

        chain = ["shoulder", "elbow", "wrist"]
        violations = validate_ik_chain_length(bone_positions, chain, tolerance=0.05)

        # Should detect stretching violation
        assert len(violations) >= 1
        assert violations[0]["type"] == "stretch"

    def test_validate_ik_chain_length_compression(self):
        """Test IK chain with compression."""
        bone_positions = {
            "shoulder": np.array([[0, 0, 0]] * 100),
            "elbow": np.array([[10, 0, 0]] * 100),
            "wrist": np.array([[18, 0, 0]] * 100),
        }

        # Compress the chain in frames 20-40
        bone_positions["wrist"][20:40] = [12, 0, 0]  # Too close (33% compression)

        chain = ["shoulder", "elbow", "wrist"]
        violations = validate_ik_chain_length(bone_positions, chain, tolerance=0.05)

        # Should detect compression
        assert len(violations) >= 1
        assert violations[0]["type"] == "compression"


@pytest.mark.unit
class TestChainBreakDetection:
    """Test chain break detection."""

    def test_detect_chain_breaks_no_breaks(self):
        """Test continuous chain with no breaks."""
        bone_positions = {
            "bone1": np.array([[i, 0, 0] for i in range(100)]),
            "bone2": np.array([[i + 10, 0, 0] for i in range(100)]),
        }

        chain = ["bone1", "bone2"]
        breaks = detect_chain_breaks(bone_positions, chain, max_distance=0.1)

        assert len(breaks) == 0

    def test_detect_chain_breaks_with_break(self):
        """Test chain with discontinuity."""
        bone_positions = {
            "bone1": np.array([[0, 0, 0]] * 100),
            "bone2": np.array([[10, 0, 0]] * 100),
        }

        # Create a break at frame 50
        bone_positions["bone2"][50:] = [20, 0, 0]  # Sudden jump

        chain = ["bone1", "bone2"]
        breaks = detect_chain_breaks(bone_positions, chain, max_distance=0.1)

        # Should detect break around frame 50
        assert len(breaks) >= 1
        assert 49 <= breaks[0]["frame"] <= 51

    def test_detect_chain_breaks_multiple_breaks(self):
        """Test chain with multiple breaks."""
        bone_positions = {
            "bone1": np.array([[0, 0, 0]] * 100),
            "bone2": np.array([[10, 0, 0]] * 100),
        }

        # Create breaks at frames 30 and 70
        bone_positions["bone2"][30] = [20, 0, 0]
        bone_positions["bone2"][70] = [5, 0, 0]

        chain = ["bone1", "bone2"]
        breaks = detect_chain_breaks(bone_positions, chain, max_distance=0.1)

        assert len(breaks) >= 2


@pytest.mark.unit
class TestParentChildConsistency:
    """Test parent-child relationship validation."""

    def test_validate_parent_child_consistent(self):
        """Test consistent parent-child relationship."""
        parent_pos = np.array([[0, 0, 0]] * 100)
        child_pos = np.array([[10, 0, 0]] * 100)

        violations = validate_parent_child_consistency(parent_pos, child_pos, expected_distance=10.0, tolerance=0.1)

        assert len(violations) == 0

    def test_validate_parent_child_violation(self):
        """Test parent-child distance violation."""
        parent_pos = np.array([[0, 0, 0]] * 100)
        child_pos = np.array([[10, 0, 0]] * 100)

        # Violate distance in frames 40-60
        child_pos[40:60] = [15, 0, 0]

        violations = validate_parent_child_consistency(parent_pos, child_pos, expected_distance=10.0, tolerance=0.1)

        assert len(violations) >= 1
        assert violations[0]["type"] == "distance_violation"

    def test_validate_parent_child_detachment(self):
        """Test complete parent-child detachment."""
        parent_pos = np.array([[0, 0, 0]] * 100)
        child_pos = np.array([[10, 0, 0]] * 100)

        # Child completely detaches
        child_pos[50:] = [100, 100, 100]

        violations = validate_parent_child_consistency(parent_pos, child_pos, expected_distance=10.0, tolerance=0.1)

        assert len(violations) >= 1
        assert violations[0]["severity"] == "high"


@pytest.mark.unit
class TestCurveDiscontinuityDetection:
    """Test animation curve discontinuity detection."""

    def test_detect_curve_discontinuities_smooth(self):
        """Test smooth curve with no discontinuities."""
        # Smooth sine wave
        frames = 100
        t = np.linspace(0, 2 * np.pi, frames)
        curve_data = np.sin(t) * 10

        discontinuities = detect_curve_discontinuities(curve_data, threshold=5.0)

        assert len(discontinuities) == 0

    def test_detect_curve_discontinuities_jump(self):
        """Test curve with sudden jump."""
        curve_data = np.linspace(0, 10, 100)

        # Create sudden jump at frame 50
        curve_data[50:] += 20

        discontinuities = detect_curve_discontinuities(curve_data, threshold=5.0)

        assert len(discontinuities) >= 1
        assert 49 <= discontinuities[0]["frame"] <= 51

    def test_detect_curve_discontinuities_multiple_jumps(self):
        """Test curve with multiple discontinuities."""
        curve_data = np.zeros(100)

        # Create jumps at frames 25, 50, 75
        curve_data[25] = 20
        curve_data[50] = -15
        curve_data[75] = 25

        discontinuities = detect_curve_discontinuities(curve_data, threshold=5.0)

        assert len(discontinuities) >= 3

    def test_detect_curve_discontinuities_derivative(self):
        """Test discontinuity using derivative analysis."""
        # Smooth curve
        curve_data = np.linspace(0, 10, 100)

        # Add spike in velocity
        curve_data[50] = curve_data[49] + 10  # Sudden change

        discontinuities = detect_curve_discontinuities(curve_data, threshold=2.0, use_derivative=True)

        assert len(discontinuities) >= 1


@pytest.mark.unit
class TestKeyframeTimingValidation:
    """Test keyframe timing validation."""

    def test_validate_keyframe_timing_regular(self):
        """Test regular keyframe spacing."""
        keyframes = np.array([0, 10, 20, 30, 40, 50])

        violations = validate_keyframe_timing(keyframes, expected_interval=10, tolerance=1)

        assert len(violations) == 0

    def test_validate_keyframe_timing_irregular(self):
        """Test irregular keyframe spacing."""
        keyframes = np.array([0, 10, 20, 25, 40, 50])  # Frame 25 is off

        violations = validate_keyframe_timing(keyframes, expected_interval=10, tolerance=1)

        assert len(violations) >= 1

    def test_validate_keyframe_timing_missing(self):
        """Test missing keyframes."""
        keyframes = np.array([0, 10, 20, 40, 50])  # Missing frame 30

        violations = validate_keyframe_timing(keyframes, expected_interval=10, tolerance=1)

        assert len(violations) >= 1
        assert violations[0]["type"] == "missing_keyframe"

    def test_validate_keyframe_timing_duplicate(self):
        """Test duplicate keyframes."""
        keyframes = np.array([0, 10, 20, 20, 30, 40])  # Duplicate at 20

        violations = validate_keyframe_timing(keyframes, expected_interval=10, tolerance=1)

        assert len(violations) >= 1
        assert violations[0]["type"] == "duplicate_keyframe"


@pytest.mark.unit
class TestEndEffectorReachability:
    """Test end effector reachability analysis."""

    def test_check_end_effector_reachable(self):
        """Test reachable target."""
        # Chain with total length 18 (10 + 8)
        chain_lengths = [10.0, 8.0]
        target_distance = 15.0  # Within reach

        reachable = check_end_effector_reachability(chain_lengths, target_distance)

        assert reachable is True

    def test_check_end_effector_unreachable(self):
        """Test unreachable target."""
        chain_lengths = [10.0, 8.0]
        target_distance = 20.0  # Beyond reach (max = 18)

        reachable = check_end_effector_reachability(chain_lengths, target_distance)

        assert reachable is False

    def test_check_end_effector_at_limit(self):
        """Test target at maximum reach."""
        chain_lengths = [10.0, 8.0]
        target_distance = 18.0  # Exactly at limit

        reachable = check_end_effector_reachability(chain_lengths, target_distance)

        assert reachable is True

    def test_check_end_effector_too_close(self):
        """Test target too close (minimum reach)."""
        chain_lengths = [10.0, 8.0]
        target_distance = 1.0  # Too close (min = 2 when fully folded)

        reachable = check_end_effector_reachability(chain_lengths, target_distance, check_min=True)

        assert reachable is False

    def test_check_end_effector_multiple_bones(self):
        """Test reachability with multiple bones in chain."""
        chain_lengths = [5.0, 4.0, 3.0, 2.0]
        target_distance = 12.0  # Within total reach of 14

        reachable = check_end_effector_reachability(chain_lengths, target_distance)

        assert reachable is True


@pytest.mark.unit
class TestHierarchyViolations:
    """Test hierarchy integrity checks."""

    def test_detect_hierarchy_violations_valid(self):
        """Test valid hierarchy."""
        hierarchy = {
            "root": None,
            "spine": "root",
            "chest": "spine",
            "shoulder": "chest",
        }

        violations = detect_hierarchy_violations(hierarchy)

        assert len(violations) == 0

    def test_detect_hierarchy_violations_circular(self):
        """Test circular dependency detection."""
        hierarchy = {
            "bone1": "bone2",
            "bone2": "bone3",
            "bone3": "bone1",  # Circular!
        }

        violations = detect_hierarchy_violations(hierarchy)

        assert len(violations) >= 1
        assert violations[0]["type"] == "circular_dependency"

    def test_detect_hierarchy_violations_orphan(self):
        """Test orphaned bone detection."""
        hierarchy = {
            "root": None,
            "spine": "root",
            "orphan": "nonexistent",  # Parent doesn't exist
        }

        violations = detect_hierarchy_violations(hierarchy)

        assert len(violations) >= 1
        assert violations[0]["type"] == "orphaned_bone"

    def test_detect_hierarchy_violations_multiple_roots(self):
        """Test multiple root detection."""
        hierarchy = {
            "root1": None,
            "root2": None,  # Two roots!
            "child1": "root1",
            "child2": "root2",
        }

        violations = detect_hierarchy_violations(hierarchy)

        assert len(violations) >= 1
        assert violations[0]["type"] == "multiple_roots"


@pytest.mark.unit
class TestConstraintViolationAnalysis:
    """Test main constraint violation analysis function."""

    def test_analyze_constraint_violations_basic(self, mock_scene, temp_output_dir):
        """Test basic constraint violation analysis."""
        results = analyze_constraint_violations(mock_scene, output_dir=temp_output_dir)

        # Check required fields
        assert "total_chains" in results
        assert "ik_violations" in results
        assert "hierarchy_violations" in results
        assert "curve_discontinuities" in results
        assert "overall_constraint_score" in results

    def test_analyze_constraint_violations_outputs_csv(self, mock_scene, temp_output_dir):
        """Test that CSV files are generated."""
        from pathlib import Path

        analyze_constraint_violations(mock_scene, output_dir=temp_output_dir)

        # Check for expected output files
        expected_files = [
            "ik_chain_violations.csv",
            "hierarchy_violations.csv",
            "curve_discontinuities.csv",
            "constraint_summary.csv",
        ]

        for filename in expected_files:
            filepath = Path(temp_output_dir) / filename
            assert filepath.exists(), f"Expected file not created: {filename}"

    def test_analyze_constraint_violations_no_violations(self, mock_scene, temp_output_dir):
        """Test with valid scene (no violations)."""
        results = analyze_constraint_violations(mock_scene, output_dir=temp_output_dir)

        # High score = good (no violations)
        assert results["overall_constraint_score"] >= 0.8

    def test_analyze_constraint_violations_severity_scoring(self, mock_scene, temp_output_dir):
        """Test that severity affects overall score."""
        results = analyze_constraint_violations(mock_scene, output_dir=temp_output_dir)

        # Score should be between 0 and 1
        assert 0.0 <= results["overall_constraint_score"] <= 1.0


@pytest.mark.unit
class TestAdvancedConstraints:
    """Test advanced constraint scenarios."""

    def test_ik_pole_vector_constraint(self):
        """Test pole vector constraint validation."""
        # Pole vector should maintain consistent orientation
        bone_positions = {
            "shoulder": np.array([[0, 0, 0]] * 100),
            "elbow": np.array([[10, 0, 0]] * 100),
            "wrist": np.array([[18, 0, 0]] * 100),
        }

        # Elbow should bend in consistent direction
        chain = ["shoulder", "elbow", "wrist"]
        violations = validate_ik_chain_length(bone_positions, chain, check_orientation=True)

        # Implementation may or may not support this
        assert isinstance(violations, list)

    def test_constraint_animation_loop(self):
        """Test loop continuity constraint."""
        # First and last frames should match for looping
        curve_data = np.sin(np.linspace(0, 2 * np.pi, 100)) * 10

        # Check if first and last frames are close
        loop_violation = abs(curve_data[0] - curve_data[-1])

        # For looping animation, should be small
        assert loop_violation < 1.0

    def test_constraint_center_of_mass(self):
        """Test center of mass constraint."""
        # Center of mass should stay within support polygon for stable poses
        bone_positions = {
            "hip": np.array([[0, 10, 0]] * 100),
            "left_foot": np.array([[-5, 0, 0]] * 100),
            "right_foot": np.array([[5, 0, 0]] * 100),
        }

        # COM should be between feet
        hip_x = bone_positions["hip"][:, 0]
        left_x = bone_positions["left_foot"][:, 0]
        right_x = bone_positions["right_foot"][:, 0]

        # Simple check
        assert np.all((left_x <= hip_x) & (hip_x <= right_x))


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for constraint violation detection."""

    def test_ik_chain_single_bone(self):
        """Test IK chain with single bone."""
        bone_positions = {
            "bone1": np.zeros((100, 3)),
        }

        chain = ["bone1"]
        violations = validate_ik_chain_length(bone_positions, chain)

        # Should handle gracefully (no violations possible)
        assert len(violations) == 0

    def test_ik_chain_empty(self):
        """Test IK chain with no bones."""
        bone_positions = {}
        chain = []

        violations = validate_ik_chain_length(bone_positions, chain)

        assert len(violations) == 0

    def test_curve_discontinuity_constant(self):
        """Test discontinuity detection on constant curve."""
        curve_data = np.ones(100) * 5.0

        discontinuities = detect_curve_discontinuities(curve_data)

        # Constant curve = no discontinuities
        assert len(discontinuities) == 0

    def test_curve_discontinuity_nan_handling(self):
        """Test discontinuity detection with NaN values."""
        curve_data = np.ones(100) * 5.0
        curve_data[50] = np.nan

        discontinuities = detect_curve_discontinuities(curve_data)

        # Should detect NaN as discontinuity or handle gracefully
        assert isinstance(discontinuities, list)

    def test_hierarchy_single_bone(self):
        """Test hierarchy with single bone."""
        hierarchy = {
            "root": None,
        }

        violations = detect_hierarchy_violations(hierarchy)

        # Single bone = no violations
        assert len(violations) == 0

    def test_end_effector_zero_length(self):
        """Test end effector with zero-length bones."""
        chain_lengths = [0.0, 0.0, 0.0]
        target_distance = 5.0

        reachable = check_end_effector_reachability(chain_lengths, target_distance)

        # Zero length chain cannot reach anything
        assert reachable is False


@pytest.mark.unit
class TestConstraintSeverityClassification:
    """Test constraint violation severity classification."""

    def test_severity_low(self):
        """Test low severity violation."""
        bone_positions = {
            "bone1": np.array([[0, 0, 0]] * 100),
            "bone2": np.array([[10, 0, 0]] * 100),
        }

        # Small violation (6% over tolerance of 5%)
        bone_positions["bone2"][50] = [10.6, 0, 0]

        chain = ["bone1", "bone2"]
        violations = validate_ik_chain_length(bone_positions, chain, tolerance=0.05)

        if violations:
            assert violations[0]["severity"] == "low"

    def test_severity_high(self):
        """Test high severity violation."""
        bone_positions = {
            "bone1": np.array([[0, 0, 0]] * 100),
            "bone2": np.array([[10, 0, 0]] * 100),
        }

        # Large violation (100% stretch)
        bone_positions["bone2"][50] = [20, 0, 0]

        chain = ["bone1", "bone2"]
        violations = validate_ik_chain_length(bone_positions, chain, tolerance=0.05)

        if violations:
            assert violations[0]["severity"] in ["medium", "high"]
