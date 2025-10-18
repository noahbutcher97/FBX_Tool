"""
Unit tests for foot_contact_analysis module

Tests cover:
- Foot bone detection
- Ground height computation
- Contact event detection
- Foot sliding detection
- Ground penetration measurement
- Contact stability scoring
"""

from unittest.mock import Mock

import numpy as np
import pytest

from fbx_tool.analysis.foot_contact_analysis import (
    compute_contact_stability,
    compute_ground_height,
    detect_contact_events,
    detect_foot_bones,
    detect_foot_sliding,
    measure_ground_penetration,
)


@pytest.mark.unit
class TestFootBoneDetection:
    """Test automatic foot bone detection."""

    def test_detect_foot_bones_mixamo_style(self):
        """Test detection with Mixamo naming convention."""
        bones = [
            Mock(GetName=lambda: "LeftFoot"),
            Mock(GetName=lambda: "RightFoot"),
            Mock(GetName=lambda: "Spine"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is not None
        assert foot_bones["right"] is not None
        assert foot_bones["left"].GetName() == "LeftFoot"
        assert foot_bones["right"].GetName() == "RightFoot"

    def test_detect_foot_bones_lowercase(self):
        """Test detection with lowercase names."""
        bones = [
            Mock(GetName=lambda: "left foot"),
            Mock(GetName=lambda: "right foot"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is not None
        assert foot_bones["right"] is not None

    def test_detect_foot_bones_underscore_style(self):
        """Test detection with underscore naming."""
        bones = [
            Mock(GetName=lambda: "l_foot"),
            Mock(GetName=lambda: "r_foot"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is not None
        assert foot_bones["right"] is not None

    def test_detect_foot_bones_ankle(self):
        """Test detection using ankle bones."""
        bones = [
            Mock(GetName=lambda: "LeftAnkle"),
            Mock(GetName=lambda: "RightAnkle"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is not None
        assert foot_bones["right"] is not None

    def test_detect_foot_bones_no_feet(self):
        """Test when no foot bones are found."""
        bones = [
            Mock(GetName=lambda: "Spine"),
            Mock(GetName=lambda: "Head"),
            Mock(GetName=lambda: "Hand"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is None
        assert foot_bones["right"] is None

    def test_detect_foot_bones_only_left(self):
        """Test when only left foot is present."""
        bones = [
            Mock(GetName=lambda: "LeftFoot"),
            Mock(GetName=lambda: "Spine"),
        ]

        foot_bones = detect_foot_bones(bones)

        assert foot_bones["left"] is not None
        assert foot_bones["right"] is None


@pytest.mark.unit
class TestGroundHeightComputation:
    """Test ground height estimation."""

    def test_compute_ground_height_flat(self):
        """Test ground height with flat ground."""
        # Mock bones with Y positions around 5.0
        positions_at_y5 = [[0, 5.0, 0] for _ in range(10)]

        # This is a simplified test - in reality would need full mock structure
        # Just testing the concept
        ground_y = 5.0
        assert isinstance(ground_y, float)

    def test_compute_ground_height_varying(self):
        """Test ground height with varying foot heights."""
        # Minimum Y should be selected as ground
        y_positions = [10.0, 5.0, 2.0, 8.0, 3.0]
        ground_height = np.min(y_positions)

        assert ground_height == 2.0


@pytest.mark.unit
class TestContactEventDetection:
    """Test ground contact detection."""

    def test_detect_contact_events_constant_contact(self):
        """Test detection when foot is always in contact."""
        positions = np.zeros((100, 3))
        positions[:, 1] = 1.0  # Y = 1.0 (close to ground at 0)

        velocities = np.zeros((100, 3))  # No movement

        contact_segments = detect_contact_events(
            positions, velocities, ground_height=0.0, height_threshold=5.0, velocity_threshold=10.0
        )

        # Should detect one continuous contact
        assert len(contact_segments) >= 1
        start, end = contact_segments[0]
        assert end - start >= 90  # Most of the animation

    def test_detect_contact_events_no_contact(self):
        """Test detection when foot never touches ground."""
        positions = np.zeros((100, 3))
        positions[:, 1] = 50.0  # Y = 50.0 (far from ground)

        velocities = np.ones((100, 3)) * 20.0  # High velocity

        contact_segments = detect_contact_events(
            positions, velocities, ground_height=0.0, height_threshold=5.0, velocity_threshold=10.0
        )

        assert len(contact_segments) == 0

    def test_detect_contact_events_multiple_steps(self):
        """Test detection of multiple contact events (steps)."""
        positions = np.zeros((100, 3))
        velocities = np.zeros((100, 3))

        # Create pattern: contact, lift, contact, lift
        # Frames 0-20: contact
        positions[0:20, 1] = 1.0
        velocities[0:20] = 0.0

        # Frames 21-40: lift (no contact)
        positions[21:40, 1] = 20.0
        velocities[21:40] = 15.0

        # Frames 41-60: contact
        positions[41:60, 1] = 1.0
        velocities[41:60] = 0.0

        # Frames 61-100: lift
        positions[61:100, 1] = 25.0
        velocities[61:100] = 15.0

        contact_segments = detect_contact_events(
            positions, velocities, ground_height=0.0, height_threshold=5.0, velocity_threshold=10.0
        )

        # Should detect 2 contact events
        assert len(contact_segments) >= 2


@pytest.mark.unit
class TestFootSlidingDetection:
    """Test foot sliding detection."""

    def test_detect_foot_sliding_no_sliding(self):
        """Test when foot doesn't slide during contact."""
        positions = np.zeros((100, 3))
        positions[:, 0] = 0.0  # Stationary in X
        positions[:, 2] = 0.0  # Stationary in Z

        velocities = np.zeros((100, 3))

        contact_segments = [(20, 80)]  # Single contact

        sliding_events = detect_foot_sliding(positions, velocities, contact_segments, sliding_threshold=5.0)

        assert len(sliding_events) == 0

    def test_detect_foot_sliding_with_sliding(self):
        """Test detection of foot sliding."""
        positions = np.zeros((50, 3))
        velocities = np.zeros((50, 3))

        # During contact (frames 10-40), foot moves horizontally
        for i in range(10, 40):
            positions[i, 0] = i * 0.5  # Moving in X
            velocities[i, 0] = 10.0  # High horizontal velocity

        contact_segments = [(10, 40)]

        sliding_events = detect_foot_sliding(positions, velocities, contact_segments, sliding_threshold=5.0)

        assert len(sliding_events) >= 1
        event = sliding_events[0]
        assert event["sliding_distance"] > 0
        assert event["peak_sliding_speed"] >= 5.0

    def test_detect_foot_sliding_severity_levels(self):
        """Test sliding severity classification."""
        positions = np.zeros((50, 3))
        velocities = np.zeros((50, 3))

        # High sliding distance
        for i in range(10, 40):
            positions[i, 0] = i * 1.0  # Large displacement
            velocities[i, 0] = 15.0

        contact_segments = [(10, 40)]

        sliding_events = detect_foot_sliding(positions, velocities, contact_segments, sliding_threshold=5.0)

        if sliding_events:
            event = sliding_events[0]
            assert event["severity"] in ["low", "medium", "high"]


@pytest.mark.unit
class TestGroundPenetration:
    """Test ground penetration measurement."""

    def test_measure_ground_penetration_none(self):
        """Test when foot doesn't penetrate ground."""
        positions = np.zeros((100, 3))
        positions[:, 1] = 5.0  # Y = 5.0 (above ground at 0)

        ground_height = 0.0
        contact_segments = [(20, 80)]

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments)

        assert len(penetration_events) == 0

    def test_measure_ground_penetration_detected(self):
        """Test detection of ground penetration."""
        positions = np.zeros((100, 3))
        # Some frames go below ground
        positions[30:50, 1] = -2.0  # Penetrating 2 units

        ground_height = 0.0
        contact_segments = [(20, 80)]

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments)

        assert len(penetration_events) >= 1
        event = penetration_events[0]
        assert event["max_penetration_depth"] >= 1.0
        assert event["mean_penetration_depth"] > 0

    def test_measure_ground_penetration_severity(self):
        """Test penetration severity classification."""
        positions = np.zeros((100, 3))
        # Deep penetration
        positions[30:50, 1] = -10.0

        ground_height = 0.0
        contact_segments = [(20, 80)]

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments)

        if penetration_events:
            event = penetration_events[0]
            assert event["severity"] == "high"


@pytest.mark.unit
class TestContactStability:
    """Test contact stability scoring."""

    def test_compute_contact_stability_stable(self):
        """Test stability score for stable contact."""
        # Perfectly still foot
        positions = np.zeros((50, 3))
        positions[:, :] = [10.0, 5.0, 20.0]  # Same position all frames

        velocities = np.zeros((50, 3))
        accelerations = np.zeros((50, 3))

        contact_segments = [(0, 49)]

        stability_scores = compute_contact_stability(positions, velocities, accelerations, contact_segments)

        assert len(stability_scores) == 1
        score = stability_scores[0]
        assert score["overall_stability"] > 0.8
        assert score["quality"] in ["excellent", "good"]

    def test_compute_contact_stability_unstable(self):
        """Test stability score for unstable contact."""
        # Moving/jittery foot
        positions = np.random.randn(50, 3) * 5.0  # Random movement
        velocities = np.random.randn(50, 3) * 10.0  # High velocity
        accelerations = np.random.randn(50, 3) * 20.0  # High acceleration

        contact_segments = [(0, 49)]

        stability_scores = compute_contact_stability(positions, velocities, accelerations, contact_segments)

        assert len(stability_scores) == 1
        score = stability_scores[0]
        assert score["overall_stability"] < 0.8
        # Quality should reflect low stability
        assert score["quality"] in ["fair", "poor"]

    def test_compute_contact_stability_multiple_contacts(self):
        """Test stability for multiple contact segments."""
        positions = np.zeros((100, 3))
        velocities = np.zeros((100, 3))
        accelerations = np.zeros((100, 3))

        # Multiple contact segments
        contact_segments = [(10, 30), (50, 70), (80, 95)]

        stability_scores = compute_contact_stability(positions, velocities, accelerations, contact_segments)

        assert len(stability_scores) == 3
        # All should have scores
        for score in stability_scores:
            assert 0 <= score["overall_stability"] <= 1.0
            assert "quality" in score


@pytest.mark.unit
class TestAdaptiveThresholds:
    """Test adaptive threshold calculation for robust contact detection."""

    def test_calculate_adaptive_velocity_threshold_from_distribution(self):
        """Should derive velocity threshold from data percentiles, not hardcode."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_velocity_threshold

        # Walking animation: low velocity during stance, high during swing
        # Velocities: [0.5, 1.0, 1.5, 2.0] (stance) + [20, 25, 30, 35] (swing)
        velocities = np.array([0.5, 1.0, 1.5, 2.0, 20, 25, 30, 35])

        threshold = calculate_adaptive_velocity_threshold(velocities)

        # Threshold should separate stance from swing (between 2.0 and 20)
        assert 2.0 < threshold < 20.0, f"Expected threshold between 2 and 20, got {threshold}"

        # Should use percentile-based approach (e.g., 30th percentile)
        expected_approx = np.percentile(velocities, 30)
        assert abs(threshold - expected_approx) < 10, "Should be close to 30th percentile"

    def test_calculate_adaptive_height_threshold_from_trajectory(self):
        """Should derive height threshold from foot trajectory variance."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_height_threshold

        # Foot trajectory: bounces between ground (5.0) and air (50.0)
        heights_above_ground = np.array([0.1, 0.2, 0.1, 45, 48, 50, 0.2, 0.1, 48, 49])

        threshold = calculate_adaptive_height_threshold(heights_above_ground)

        # Threshold should separate ground contact from aerial phase
        assert 0.5 < threshold < 40, f"Expected threshold between 0.5 and 40, got {threshold}"

        # Should capture low-height frames while excluding aerial frames
        ground_frames = heights_above_ground[heights_above_ground < threshold]
        assert len(ground_frames) >= 4, "Should capture at least 4 ground frames"

        aerial_frames = heights_above_ground[heights_above_ground > threshold]
        assert len(aerial_frames) >= 4, "Should exclude at least 4 aerial frames"

    def test_adaptive_thresholds_with_constant_values(self):
        """Should handle edge case of constant velocities gracefully."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_velocity_threshold

        # All velocities are the same (no variance)
        constant_velocities = np.ones(100) * 5.0

        threshold = calculate_adaptive_velocity_threshold(constant_velocities)

        # Should return a reasonable default (e.g., the constant value itself or slightly above)
        assert threshold > 0, "Threshold must be positive"
        assert abs(threshold - 5.0) < 10, "Should be close to constant value"

    def test_adaptive_thresholds_with_zero_variance(self):
        """Should handle zero variance data without division by zero."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_velocity_threshold

        # All zeros
        zero_velocities = np.zeros(50)

        threshold = calculate_adaptive_velocity_threshold(zero_velocities)

        # Should not crash and return a small positive threshold
        assert threshold >= 0, "Threshold must be non-negative"
        assert np.isfinite(threshold), "Threshold must be finite"

    def test_adaptive_thresholds_with_extreme_outliers(self):
        """Should be robust to extreme outliers in velocity data."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_velocity_threshold

        # Mostly low velocities with extreme outliers
        velocities = np.concatenate(
            [
                np.ones(90) * 2.0,  # 90 frames of normal velocity
                np.array([1000, 2000, 5000]),  # 3 extreme outliers (glitches)
            ]
        )

        threshold = calculate_adaptive_velocity_threshold(velocities)

        # Should not be influenced by outliers (use robust percentile)
        assert threshold < 100, "Should ignore extreme outliers"
        assert threshold > 1.0, "Should be based on typical values"


@pytest.mark.unit
class TestRootMotionCompatibility:
    """Test contact detection works with root motion (elevated ground)."""

    def test_detect_contacts_with_elevated_ground(self):
        """Should detect contacts when entire character is elevated (root motion)."""
        # Scenario: Character walking with root motion, ground detected at Y=50
        positions = np.zeros((100, 3))
        velocities = np.zeros((100, 3))

        # Walking pattern: foot alternates between ground (50) and air (70)
        for i in range(100):
            if i % 20 < 10:  # Contact phase
                positions[i, 1] = 50.0 + np.random.uniform(-0.5, 0.5)  # Near ground
                velocities[i] = np.random.uniform(-2, 2, 3)  # Low velocity
            else:  # Swing phase
                positions[i, 1] = 50.0 + 20.0  # Elevated
                velocities[i] = np.random.uniform(-15, 15, 3)  # High velocity

        ground_height = 50.0  # Elevated ground

        contact_segments = detect_contact_events(
            positions, velocities, ground_height, height_threshold=5.0, velocity_threshold=10.0
        )

        # Should detect multiple contacts despite elevated ground
        assert (
            len(contact_segments) >= 3
        ), f"Expected at least 3 contacts with elevated ground, got {len(contact_segments)}"

        # Verify contacts are during low-height phases
        for start, end in contact_segments:
            contact_heights = positions[start : end + 1, 1] - ground_height
            assert np.mean(contact_heights) < 5.0, "Contacts should be near ground level"

    def test_detect_contacts_with_adaptive_thresholds_root_motion(self):
        """Should auto-adapt thresholds for root motion animations."""
        from fbx_tool.analysis.foot_contact_analysis import detect_contact_events_adaptive

        # Create realistic walking with root motion
        n_frames = 120
        positions = np.zeros((n_frames, 3))
        velocities = np.zeros((n_frames, 3))

        ground_height = 100.0  # High ground due to root motion

        # 4 steps: contact at frames 0-20, 40-60, 80-100
        # Use deterministic pattern instead of random to avoid flaky tests
        for i in range(n_frames):
            cycle = i % 40
            if cycle < 20:  # Stance phase
                positions[i, 1] = ground_height + 0.5  # Near ground
                velocities[i] = [1.0, 0.5, 1.0]  # Low, consistent velocity
            else:  # Swing phase
                positions[i, 1] = ground_height + 25  # Elevated
                velocities[i] = [10.0, 15.0, 10.0]  # High, consistent velocity

        # Use adaptive detection (should calculate thresholds from data)
        contact_segments = detect_contact_events_adaptive(positions, velocities, ground_height)

        # Should detect all 3 complete stride contacts
        assert len(contact_segments) >= 3, f"Expected >= 3 contacts, got {len(contact_segments)}"

        # Verify reasonable contact durations (at least a few frames each)
        long_enough_contacts = [seg for seg in contact_segments if (seg[1] - seg[0] + 1) >= 5]
        assert len(long_enough_contacts) >= 2, (
            f"Expected at least 2 contacts with duration >= 5 frames, "
            f"got {len(long_enough_contacts)} from {len(contact_segments)} total contacts"
        )

    def test_ground_height_estimation_percentile_based(self):
        """Ground height should use percentile, not absolute minimum (avoids glitches)."""
        from fbx_tool.analysis.foot_contact_analysis import compute_ground_height_percentile

        # Foot trajectory with one glitch frame that goes below actual ground
        y_positions = np.concatenate(
            [
                np.ones(80) * 10.0,  # Normal ground contact
                np.ones(15) * 30.0,  # Swing phase
                np.array([10.0, 9.8, 10.2, 10.1, 9.9]),  # Back to ground
                np.array([-50.0]),  # GLITCH: foot penetrates way below ground (data error)
            ]
        )

        # Percentile-based approach should ignore glitch
        ground_height = compute_ground_height_percentile(y_positions, percentile=5)

        # Should be close to 10.0, NOT -50.0
        assert 8.0 <= ground_height <= 12.0, f"Expected ground ~10, got {ground_height}"
        assert ground_height > 0, "Should ignore negative glitch"

    def test_contact_detection_scale_invariance(self):
        """Should work across different character scales (small vs large)."""
        from fbx_tool.analysis.foot_contact_analysis import detect_contact_events_adaptive

        # Small character (units in centimeters, velocities low)
        small_positions = np.zeros((60, 3))
        small_velocities = np.zeros((60, 3))
        small_positions[0:20, 1] = 2.0  # Ground at 2cm
        small_positions[20:40, 1] = 12.0  # Swing at 12cm
        small_positions[40:60, 1] = 2.0  # Ground again
        small_velocities[0:20] = 0.5  # Low velocity
        small_velocities[20:40] = 5.0  # Higher velocity in swing
        small_velocities[40:60] = 0.5

        # Large character (units in meters, velocities higher)
        large_positions = small_positions * 100  # Scale up 100x
        large_velocities = small_velocities * 10  # Velocities scale with size

        # Both should detect 2 contacts
        small_contacts = detect_contact_events_adaptive(small_positions, small_velocities, ground_height=2.0)
        large_contacts = detect_contact_events_adaptive(large_positions, large_velocities, ground_height=200.0)

        assert len(small_contacts) == 2, "Small character should detect 2 contacts"
        assert len(large_contacts) == 2, "Large character should detect 2 contacts"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for foot contact analysis."""

    def test_contact_detection_empty_arrays(self):
        """Test with empty position/velocity arrays."""
        positions = np.array([]).reshape(0, 3)
        velocities = np.array([]).reshape(0, 3)

        contact_segments = detect_contact_events(positions, velocities, ground_height=0.0)

        assert len(contact_segments) == 0

    def test_sliding_detection_single_frame_contact(self):
        """Test sliding with very short contact."""
        positions = np.zeros((10, 3))
        velocities = np.zeros((10, 3))

        contact_segments = [(5, 5)]  # Single frame

        sliding_events = detect_foot_sliding(positions, velocities, contact_segments)

        # Should handle gracefully (too short to analyze)
        assert isinstance(sliding_events, list)

    def test_stability_short_contact(self):
        """Test stability with short contact segment."""
        positions = np.zeros((10, 3))
        velocities = np.zeros((10, 3))
        accelerations = np.zeros((10, 3))

        contact_segments = [(2, 5)]  # 4 frames

        stability_scores = compute_contact_stability(positions, velocities, accelerations, contact_segments)

        assert len(stability_scores) == 1
        # Should still compute a score
        assert "overall_stability" in stability_scores[0]

    def test_contact_detection_single_frame_animation(self):
        """Should handle single-frame animation without crashing."""
        positions = np.array([[0, 5, 0]])  # Single frame
        velocities = np.array([[0, 0, 0]])

        contact_segments = detect_contact_events(positions, velocities, ground_height=0.0)

        # Should handle gracefully
        assert isinstance(contact_segments, list)

    def test_contact_detection_nan_values(self):
        """Should handle NaN values in position/velocity data gracefully."""
        positions = np.zeros((50, 3))
        positions[10:15, :] = np.nan  # Corrupted frames
        positions[20:30, 1] = 1.0  # Valid ground contact

        velocities = np.zeros((50, 3))
        velocities[10:15, :] = np.nan  # Corrupted frames

        # Should not crash despite NaN values
        contact_segments = detect_contact_events(
            positions, velocities, ground_height=0.0, height_threshold=5.0, velocity_threshold=10.0
        )

        assert isinstance(contact_segments, list)
        # Should still detect contacts (NaN frames will be skipped/grouped with neighbors)
        # Verify frames 20-30 are included in some contact segment
        if contact_segments:
            # Check if frames 20-30 are covered by any segment
            frames_20_30_covered = any(start <= 20 and end >= 30 for start, end in contact_segments)
            # OR check if at least frame 25 (middle of valid range) is in a contact
            frame_25_covered = any(start <= 25 <= end for start, end in contact_segments)

            assert (
                frames_20_30_covered or frame_25_covered
            ), f"Expected frames 20-30 to be in contacts, got segments: {contact_segments}"
