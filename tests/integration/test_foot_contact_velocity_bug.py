"""Integration coverage for velocity-aware foot-contact detection helpers."""

import numpy as np


class TestFootContactVelocity:
    """Tests for contact velocity calculation and threshold separation."""

    def test_adaptive_velocity_threshold_separates_contact_and_swing(self):
        """Adaptive threshold should keep low contact velocities below high swing velocities."""
        from fbx_tool.analysis.foot_contact_analysis import calculate_adaptive_velocity_threshold

        # Test with sample velocity data
        velocities = np.array([5.0, 10.0, 8.0, 150.0, 200.0, 7.0, 9.0, 6.0])

        threshold = calculate_adaptive_velocity_threshold(velocities)

        # Should separate low velocities (contact) from high velocities (aerial)
        assert threshold > 10.0, "Threshold should be above typical contact velocities"
        assert threshold < 150.0, "Threshold should be below aerial phase velocities"
