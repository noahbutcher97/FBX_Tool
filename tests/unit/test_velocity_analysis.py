"""
Unit tests for velocity_analysis module

Tests cover:
- Derivative computation (velocity, acceleration, jerk)
- Angular derivative computation
- Magnitude calculations
- Spike detection
- Jitter scoring
- Smoothness scoring
- Frozen frame detection
- Directional jitter analysis
- Smoothing parameter recommendations
"""

import numpy as np
import pytest

from fbx_tool.analysis.velocity_analysis import (
    compute_angular_derivatives,
    compute_derivatives,
    compute_directional_jitter,
    compute_jitter_score,
    compute_magnitudes,
    compute_smoothing_parameters,
    compute_smoothness_score,
    detect_frozen_frames,
    detect_spikes,
)


@pytest.mark.unit
class TestComputeDerivatives:
    """Test derivative computation functions."""

    def test_compute_derivatives_basic(self, sample_positions, sample_frame_rate):
        """Test basic derivative computation."""
        velocity, acceleration, jerk = compute_derivatives(sample_positions, sample_frame_rate)

        # Check shapes
        assert velocity.shape == sample_positions.shape
        assert acceleration.shape == sample_positions.shape
        assert jerk.shape == sample_positions.shape

        # Check types
        assert isinstance(velocity, np.ndarray)
        assert isinstance(acceleration, np.ndarray)
        assert isinstance(jerk, np.ndarray)

    def test_compute_derivatives_constant_position(self, sample_frame_rate):
        """Test derivatives of stationary bone (all zeros)."""
        positions = np.zeros((100, 3))
        velocity, acceleration, jerk = compute_derivatives(positions, sample_frame_rate)

        # All derivatives should be near zero
        assert np.allclose(velocity, 0, atol=1e-6)
        assert np.allclose(acceleration, 0, atol=1e-6)
        assert np.allclose(jerk, 0, atol=1e-6)

    def test_compute_derivatives_linear_motion(self, sample_frame_rate):
        """Test derivatives of constant velocity motion."""
        frames = 100
        t = np.arange(frames) / sample_frame_rate

        # Constant velocity motion in X
        positions = np.zeros((frames, 3))
        positions[:, 0] = t * 5.0  # 5 units/second

        velocity, acceleration, jerk = compute_derivatives(positions, sample_frame_rate)

        # Velocity should be constant ~5
        assert np.allclose(velocity[:, 0], 5.0, atol=0.5)

        # Acceleration and jerk should be near zero
        assert np.allclose(acceleration, 0, atol=0.5)
        assert np.allclose(jerk, 0, atol=1.0)

    def test_compute_angular_derivatives_basic(self, sample_rotations, sample_frame_rate):
        """Test angular derivative computation."""
        angular_vel, angular_acc, angular_jerk = compute_angular_derivatives(sample_rotations, sample_frame_rate)

        # Check shapes
        assert angular_vel.shape == sample_rotations.shape
        assert angular_acc.shape == sample_rotations.shape
        assert angular_jerk.shape == sample_rotations.shape

    def test_compute_angular_derivatives_wrapping(self, sample_frame_rate):
        """Test angle unwrapping (359° -> 1° = +2°, not -358°)."""
        rotations = np.array([[0, 0, 0], [350, 0, 0], [359, 0, 0], [1, 0, 0], [10, 0, 0]])  # Wrapped from 359° to 1°

        angular_vel, _, _ = compute_angular_derivatives(rotations, sample_frame_rate)

        # After unwrapping, velocity should be positive
        # (not a huge negative spike)
        assert angular_vel[3, 0] > -50  # Should not be -358°/frame


@pytest.mark.unit
class TestMagnitudeCalculations:
    """Test magnitude computation."""

    def test_compute_magnitudes_basic(self):
        """Test basic magnitude computation."""
        vectors = np.array(
            [
                [3, 4, 0],  # Magnitude = 5
                [1, 0, 0],  # Magnitude = 1
                [0, 0, 0],  # Magnitude = 0
            ]
        )

        magnitudes = compute_magnitudes(vectors)

        expected = np.array([5.0, 1.0, 0.0])
        assert np.allclose(magnitudes, expected)

    def test_compute_magnitudes_3d(self):
        """Test magnitude with 3D vectors."""
        vectors = np.array(
            [
                [1, 1, 1],  # sqrt(3)
                [2, 2, 2],  # 2*sqrt(3)
            ]
        )

        magnitudes = compute_magnitudes(vectors)

        expected = np.array([np.sqrt(3), 2 * np.sqrt(3)])
        assert np.allclose(magnitudes, expected)


@pytest.mark.unit
class TestSpikeDetection:
    """Test spike detection."""

    def test_detect_spikes_no_spikes(self):
        """Test spike detection with no spikes."""
        values = np.ones(100) * 5.0  # Constant values
        spike_indices = detect_spikes(values, threshold_multiplier=3.0)

        assert len(spike_indices) == 0

    def test_detect_spikes_with_spike(self):
        """Test spike detection with obvious spike."""
        values = np.ones(100) * 5.0
        values[50] = 100.0  # Large spike

        spike_indices = detect_spikes(values, threshold_multiplier=3.0)

        assert len(spike_indices) >= 1
        assert 50 in spike_indices

    def test_detect_spikes_multiple(self):
        """Test detection of multiple spikes."""
        values = np.ones(100) * 5.0
        values[25] = 50.0
        values[50] = 60.0
        values[75] = 55.0

        spike_indices = detect_spikes(values, threshold_multiplier=2.0)

        assert len(spike_indices) >= 3


@pytest.mark.unit
class TestJitterScoring:
    """Test jitter scoring."""

    def test_compute_jitter_score_smooth(self):
        """Test jitter score on smooth data."""
        values = np.linspace(0, 10, 100)  # Perfectly smooth
        jitter = compute_jitter_score(values, window_size=5)

        # Smooth data should have low jitter
        assert jitter < 0.1

    def test_compute_jitter_score_noisy(self):
        """Test jitter score on noisy data."""
        values = np.random.randn(100) * 10  # High noise
        jitter = compute_jitter_score(values, window_size=5)

        # Noisy data should have high jitter
        assert jitter > 1.0

    def test_compute_jitter_score_too_few_values(self):
        """Test jitter score with insufficient data."""
        values = np.array([1.0, 2.0])
        jitter = compute_jitter_score(values, window_size=5)

        # Should return 0 for insufficient data
        assert jitter == 0.0

    def test_compute_directional_jitter(self, sample_velocities):
        """Test directional jitter analysis."""
        jitter_scores = compute_directional_jitter(sample_velocities, window_size=5)

        # Should have jitter for x, y, z, and magnitude
        assert "x" in jitter_scores
        assert "y" in jitter_scores
        assert "z" in jitter_scores
        assert "magnitude" in jitter_scores

        # All should be non-negative
        assert all(score >= 0 for score in jitter_scores.values())


@pytest.mark.unit
class TestSmoothnessScoring:
    """Test smoothness scoring."""

    def test_compute_smoothness_score_smooth(self):
        """Test smoothness on low jerk."""
        jerk_magnitude = np.ones(100) * 0.1  # Low jerk
        smoothness = compute_smoothness_score(jerk_magnitude)

        # Low jerk = high smoothness
        assert smoothness > 0.7

    def test_compute_smoothness_score_jerky(self):
        """Test smoothness on high jerk."""
        jerk_magnitude = np.ones(100) * 100.0  # High jerk
        smoothness = compute_smoothness_score(jerk_magnitude)

        # High jerk = low smoothness
        assert smoothness < 0.3

    def test_compute_smoothness_score_zero_jerk(self):
        """Test smoothness with zero jerk."""
        jerk_magnitude = np.zeros(100)
        smoothness = compute_smoothness_score(jerk_magnitude)

        # Zero jerk = perfect smoothness
        assert smoothness == 1.0


@pytest.mark.unit
class TestFrozenFrameDetection:
    """Test frozen frame detection."""

    def test_detect_frozen_frames_none(self):
        """Test with no frozen frames."""
        velocity_magnitude = np.ones(100) * 10.0  # Always moving
        frozen_segments = detect_frozen_frames(velocity_magnitude, threshold=0.001)

        assert len(frozen_segments) == 0

    def test_detect_frozen_frames_single_segment(self):
        """Test detection of single frozen segment."""
        velocity_magnitude = np.ones(100) * 10.0
        velocity_magnitude[40:60] = 0.0  # Frozen frames 40-59

        frozen_segments = detect_frozen_frames(velocity_magnitude, threshold=0.001)

        assert len(frozen_segments) >= 1
        # Should detect the frozen segment
        start, end = frozen_segments[0]
        assert start >= 40 and end <= 59

    def test_detect_frozen_frames_multiple_segments(self):
        """Test detection of multiple frozen segments."""
        velocity_magnitude = np.ones(100) * 10.0
        velocity_magnitude[10:20] = 0.0
        velocity_magnitude[50:60] = 0.0
        velocity_magnitude[80:90] = 0.0

        frozen_segments = detect_frozen_frames(velocity_magnitude, threshold=0.001)

        assert len(frozen_segments) >= 3

    def test_detect_frozen_frames_end_of_animation(self):
        """Test detection when animation ends frozen."""
        velocity_magnitude = np.ones(100) * 10.0
        velocity_magnitude[80:] = 0.0  # Frozen at end

        frozen_segments = detect_frozen_frames(velocity_magnitude, threshold=0.001)

        assert len(frozen_segments) >= 1
        _, end = frozen_segments[-1]
        assert end == 99  # Should capture to end


@pytest.mark.unit
class TestSmoothingParameters:
    """Test smoothing parameter recommendations."""

    def test_compute_smoothing_parameters_high_jitter(self, sample_frame_rate):
        """Test smoothing recommendations for high jitter."""
        jitter_score = 5.0  # High jitter
        smoothness_score = 0.2  # Low smoothness

        params = compute_smoothing_parameters(jitter_score, smoothness_score, sample_frame_rate)

        assert params["intensity"] == "high"
        assert params["kernel_size"] >= 7
        assert params["gaussian_sigma"] >= 2.0

    def test_compute_smoothing_parameters_medium_jitter(self, sample_frame_rate):
        """Test smoothing recommendations for medium jitter."""
        jitter_score = 0.5  # Medium jitter
        smoothness_score = 0.5  # Medium smoothness

        params = compute_smoothing_parameters(jitter_score, smoothness_score, sample_frame_rate)

        assert params["intensity"] == "medium"
        assert params["kernel_size"] == 5
        assert params["gaussian_sigma"] == 1.0

    def test_compute_smoothing_parameters_low_jitter(self, sample_frame_rate):
        """Test smoothing recommendations for low jitter."""
        jitter_score = 0.05  # Low jitter
        smoothness_score = 0.9  # High smoothness

        params = compute_smoothing_parameters(jitter_score, smoothness_score, sample_frame_rate)

        assert params["intensity"] == "none"
        assert params["kernel_size"] == 3

    def test_compute_smoothing_parameters_includes_all_fields(self, sample_frame_rate):
        """Test that all expected fields are present."""
        params = compute_smoothing_parameters(1.0, 0.5, sample_frame_rate)

        required_fields = [
            "intensity",
            "kernel_size",
            "gaussian_sigma",
            "cutoff_frequency_hz",
            "butterworth_order",
            "savgol_window",
            "savgol_polyorder",
        ]

        for field in required_fields:
            assert field in params


@pytest.mark.unit
class TestAdaptiveThresholds:
    """Test adaptive threshold computation for proceduralization.

    These tests ensure thresholds are data-driven, not hardcoded.
    Following TDD: tests demand robust implementations.
    """

    def test_compute_adaptive_jitter_thresholds_diverse_data(self):
        """Test adaptive jitter thresholds with diverse bone jitter scores.

        Should classify jitter levels based on data distribution, not hardcoded values.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_jitter_thresholds

        # Simulate jitter scores from multiple bones
        # Some smooth (low jitter), some moderate, some high
        jitter_scores = np.array(
            [
                0.01,
                0.02,
                0.03,
                0.05,
                0.08,  # Low jitter (5 bones)
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,  # Medium jitter (5 bones)
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,  # High jitter (5 bones)
            ]
        )

        thresholds = compute_adaptive_jitter_thresholds(jitter_scores)

        # Should return two thresholds
        assert "jitter_high_threshold" in thresholds
        assert "jitter_medium_threshold" in thresholds

        # Thresholds should be data-driven (percentile-based)
        # Medium threshold should be around 33rd percentile
        # High threshold should be around 67th percentile
        expected_medium = np.percentile(jitter_scores, 33)
        expected_high = np.percentile(jitter_scores, 67)

        assert abs(thresholds["jitter_medium_threshold"] - expected_medium) < 0.1
        assert abs(thresholds["jitter_high_threshold"] - expected_high) < 0.1

        # High threshold must be greater than medium
        assert thresholds["jitter_high_threshold"] > thresholds["jitter_medium_threshold"]

        # Both must be positive
        assert thresholds["jitter_high_threshold"] > 0
        assert thresholds["jitter_medium_threshold"] > 0

    def test_compute_adaptive_jitter_thresholds_all_smooth(self):
        """Test adaptive jitter thresholds when all bones are smooth (edge case).

        Should handle uniform low jitter gracefully.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_jitter_thresholds

        # All bones very smooth
        jitter_scores = np.array([0.01, 0.02, 0.015, 0.018, 0.012, 0.020, 0.013])

        thresholds = compute_adaptive_jitter_thresholds(jitter_scores)

        # Should still return thresholds
        assert "jitter_high_threshold" in thresholds
        assert "jitter_medium_threshold" in thresholds

        # Thresholds should be based on data (even if all low)
        assert thresholds["jitter_high_threshold"] > 0
        assert thresholds["jitter_medium_threshold"] > 0
        assert thresholds["jitter_high_threshold"] > thresholds["jitter_medium_threshold"]

    def test_compute_adaptive_jitter_thresholds_all_high(self):
        """Test adaptive jitter thresholds when all bones are noisy (edge case).

        Should handle uniform high jitter gracefully.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_jitter_thresholds

        # All bones very noisy
        jitter_scores = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        thresholds = compute_adaptive_jitter_thresholds(jitter_scores)

        # Should still return reasonable thresholds
        assert thresholds["jitter_high_threshold"] > thresholds["jitter_medium_threshold"]
        assert thresholds["jitter_medium_threshold"] > 0

    def test_compute_adaptive_jitter_thresholds_single_bone(self):
        """Test adaptive jitter thresholds with only one bone (minimum data).

        Should return sensible fallback thresholds.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_jitter_thresholds

        jitter_scores = np.array([0.5])

        thresholds = compute_adaptive_jitter_thresholds(jitter_scores)

        # Should have fallback behavior
        assert "jitter_high_threshold" in thresholds
        assert "jitter_medium_threshold" in thresholds
        assert thresholds["jitter_high_threshold"] > 0
        assert thresholds["jitter_medium_threshold"] > 0

    def test_compute_adaptive_coherence_thresholds_diverse_data(self):
        """Test adaptive coherence thresholds with diverse chain coherence scores.

        Should classify coherence levels based on data distribution.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_coherence_thresholds

        # Simulate coherence scores from multiple chains
        # Range from poor (0.1) to excellent (0.95)
        coherence_scores = np.array(
            [
                0.1,
                0.15,
                0.2,
                0.25,  # Poor coordination (4 chains)
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,  # Fair coordination (5 chains)
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,  # Good coordination (5 chains)
            ]
        )

        thresholds = compute_adaptive_coherence_thresholds(coherence_scores)

        # Should return two thresholds
        assert "coherence_good_threshold" in thresholds
        assert "coherence_fair_threshold" in thresholds

        # Should be percentile-based
        expected_fair = np.percentile(coherence_scores, 33)
        expected_good = np.percentile(coherence_scores, 67)

        assert abs(thresholds["coherence_fair_threshold"] - expected_fair) < 0.1
        assert abs(thresholds["coherence_good_threshold"] - expected_good) < 0.1

        # Good threshold must be greater than fair
        assert thresholds["coherence_good_threshold"] > thresholds["coherence_fair_threshold"]

        # Both must be in valid correlation range [-1, 1]
        assert -1 <= thresholds["coherence_fair_threshold"] <= 1
        assert -1 <= thresholds["coherence_good_threshold"] <= 1

    def test_compute_adaptive_coherence_thresholds_all_good(self):
        """Test adaptive coherence thresholds when all chains are well-coordinated.

        Should handle uniform high coherence gracefully.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_coherence_thresholds

        # All chains highly coherent
        coherence_scores = np.array([0.85, 0.88, 0.90, 0.92, 0.95, 0.87])

        thresholds = compute_adaptive_coherence_thresholds(coherence_scores)

        # Should still return thresholds
        assert thresholds["coherence_good_threshold"] > thresholds["coherence_fair_threshold"]
        assert -1 <= thresholds["coherence_fair_threshold"] <= 1
        assert -1 <= thresholds["coherence_good_threshold"] <= 1

    def test_compute_adaptive_coherence_thresholds_negative_correlations(self):
        """Test adaptive coherence thresholds with negative correlations (anti-coordinated motion).

        Should handle negative coherence scores correctly.
        """
        from fbx_tool.analysis.velocity_analysis import compute_adaptive_coherence_thresholds

        # Mix of positive, zero, and negative coherence
        coherence_scores = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.5, 0.7])

        thresholds = compute_adaptive_coherence_thresholds(coherence_scores)

        # Should handle negative values correctly
        assert "coherence_good_threshold" in thresholds
        assert "coherence_fair_threshold" in thresholds
        assert thresholds["coherence_good_threshold"] > thresholds["coherence_fair_threshold"]


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_derivatives_single_frame(self, sample_frame_rate):
        """Test derivatives with only one frame (should raise error)."""
        positions = np.array([[1.0, 2.0, 3.0]])

        # np.gradient requires at least 2 elements
        with pytest.raises(ValueError, match="Shape of array too small"):
            compute_derivatives(positions, sample_frame_rate)

    def test_derivatives_two_frames(self, sample_frame_rate):
        """Test derivatives with two frames."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        velocity, acceleration, jerk = compute_derivatives(positions, sample_frame_rate)

        assert velocity.shape == (2, 3)
        assert acceleration.shape == (2, 3)
        assert jerk.shape == (2, 3)

    def test_jitter_empty_array(self):
        """Test jitter score with empty array."""
        values = np.array([])
        jitter = compute_jitter_score(values, window_size=5)

        assert jitter == 0.0
