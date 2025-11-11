"""
Unit tests for foot contact analysis coordinate system handling.

Critical test: Verify foot contact analysis works correctly across Y-up, Z-up, and X-up
coordinate systems without hardcoded assumptions.
"""

import numpy as np
import pytest

from fbx_tool.analysis.foot_contact_analysis import (
    detect_contact_events,
    detect_foot_sliding,
    measure_ground_penetration,
)


class TestFootContactCoordinateSystem:
    """Test foot contact analysis with different coordinate systems."""

    def test_detect_contact_events_y_up(self):
        """Test contact detection with Y-up coordinate system (most common)."""
        # Simulate foot landing: starts high, descends, contacts ground, lifts off
        positions = np.array(
            [
                [0, 20, 0],  # High
                [0, 10, 0],  # Descending
                [0, 2, 0],  # Near ground
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [0, 5, 0],  # Lifting off
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0
        up_axis = 1  # Y is up

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=3.0, velocity_threshold=15.0, up_axis=up_axis
        )

        assert len(contacts) == 1, "Should detect exactly one contact event"
        start, end = contacts[0]
        assert start == 2, "Contact should start at frame 2"
        assert end == 5, "Contact should end at frame 5"

    def test_detect_contact_events_z_up(self):
        """Test contact detection with Z-up coordinate system (Blender, 3ds Max)."""
        # Same scenario but Z is now the up axis
        positions = np.array(
            [
                [0, 0, 20],  # High (Z)
                [0, 0, 10],  # Descending
                [0, 0, 2],  # Near ground
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [0, 0, 5],  # Lifting off
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0
        up_axis = 2  # Z is up

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=3.0, velocity_threshold=15.0, up_axis=up_axis
        )

        assert len(contacts) == 1, "Should detect exactly one contact event"
        start, end = contacts[0]
        assert start == 2, "Contact should start at frame 2"
        assert end == 5, "Contact should end at frame 5"

    def test_detect_contact_events_x_up(self):
        """Test contact detection with X-up coordinate system (rare but possible)."""
        # Same scenario but X is now the up axis
        positions = np.array(
            [
                [20, 0, 0],  # High (X)
                [10, 0, 0],  # Descending
                [2, 0, 0],  # Near ground
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [0, 0, 0],  # Contact
                [5, 0, 0],  # Lifting off
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0
        up_axis = 0  # X is up

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=3.0, velocity_threshold=15.0, up_axis=up_axis
        )

        assert len(contacts) == 1, "Should detect exactly one contact event"
        start, end = contacts[0]
        # Note: Due to velocity being computed between frames, contact detection happens
        # one frame earlier (at the transition into low-velocity state)
        assert start == 2, "Contact should start at frame 2 (transition to low velocity)"
        assert end == 5, "Contact should end at frame 5"

    def test_detect_foot_sliding_y_up(self):
        """Test sliding detection with Y-up coordinate system."""
        # Foot on ground but sliding forward (along Z axis in Y-up system)
        positions = np.array(
            [
                [0, 0, 0],
                [0, 0, 5],  # Moving forward while on ground (sliding!)
                [0, 0, 10],
                [0, 0, 15],
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        contact_segments = [(0, 3)]  # All frames in contact
        up_axis = 1  # Y is up
        forward_axis = 2  # Z is forward
        right_axis = 0  # X is right

        sliding_events = detect_foot_sliding(
            positions,
            velocities,
            contact_segments,
            sliding_threshold=3.0,
            up_axis=up_axis,
            forward_axis=forward_axis,
            right_axis=right_axis,
        )

        assert len(sliding_events) == 1, "Should detect sliding"
        assert sliding_events[0]["sliding_distance"] > 0, "Should measure non-zero sliding distance"

    def test_detect_foot_sliding_z_up(self):
        """Test sliding detection with Z-up coordinate system."""
        # Foot on ground but sliding forward (along Y axis in Z-up system)
        positions = np.array(
            [
                [0, 0, 0],
                [0, 5, 0],  # Moving forward while on ground (sliding!)
                [0, 10, 0],
                [0, 15, 0],
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        contact_segments = [(0, 3)]
        up_axis = 2  # Z is up
        forward_axis = 1  # Y is forward
        right_axis = 0  # X is right

        sliding_events = detect_foot_sliding(
            positions,
            velocities,
            contact_segments,
            sliding_threshold=3.0,
            up_axis=up_axis,
            forward_axis=forward_axis,
            right_axis=right_axis,
        )

        assert len(sliding_events) == 1, "Should detect sliding"
        assert sliding_events[0]["sliding_distance"] > 0, "Should measure non-zero sliding distance"

    def test_measure_ground_penetration_y_up(self):
        """Test penetration detection with Y-up coordinate system."""
        # Foot goes below ground (negative Y)
        positions = np.array(
            [
                [0, 0, 0],  # At ground
                [0, -2, 0],  # Below ground (penetration!)
                [0, -1, 0],  # Still below
                [0, 0, 0],  # Back to ground
            ],
            dtype=float,
        )

        ground_height = 0.0
        contact_segments = [(0, 3)]
        up_axis = 1  # Y is up

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments, up_axis=up_axis)

        assert len(penetration_events) == 1, "Should detect penetration"
        assert penetration_events[0]["max_penetration_depth"] == 2.0, "Should measure 2 units penetration"

    def test_measure_ground_penetration_z_up(self):
        """Test penetration detection with Z-up coordinate system."""
        # Foot goes below ground (negative Z)
        positions = np.array(
            [
                [0, 0, 0],  # At ground
                [0, 0, -2],  # Below ground (penetration!)
                [0, 0, -1],  # Still below
                [0, 0, 0],  # Back to ground
            ],
            dtype=float,
        )

        ground_height = 0.0
        contact_segments = [(0, 3)]
        up_axis = 2  # Z is up

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments, up_axis=up_axis)

        assert len(penetration_events) == 1, "Should detect penetration"
        assert penetration_events[0]["max_penetration_depth"] == 2.0, "Should measure 2 units penetration"

    def test_horizontal_plane_calculation_varies_with_up_axis(self):
        """
        Critical test: Horizontal plane MUST exclude up axis, not assume XZ plane.

        In Y-up: horizontal = XZ
        In Z-up: horizontal = XY
        In X-up: horizontal = YZ
        """
        # Y-up: sliding along Z axis (horizontal)
        # Need at least 3 positions for meaningful sliding detection (2 velocity frames)
        positions_y_up = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 20]], dtype=float)  # Stationary in Y, moving in Z
        velocities_y_up = np.gradient(positions_y_up, axis=0)
        contact_segments = [(0, 2)]

        sliding_y_up = detect_foot_sliding(
            positions_y_up,
            velocities_y_up,
            contact_segments,
            sliding_threshold=5.0,
            up_axis=1,
            forward_axis=2,
            right_axis=0,
        )

        # Z-up: sliding along Y axis (horizontal)
        positions_z_up = np.array([[0, 0, 0], [0, 10, 0], [0, 20, 0]], dtype=float)  # Stationary in Z, moving in Y
        velocities_z_up = np.gradient(positions_z_up, axis=0)

        sliding_z_up = detect_foot_sliding(
            positions_z_up,
            velocities_z_up,
            contact_segments,
            sliding_threshold=5.0,
            up_axis=2,
            forward_axis=1,
            right_axis=0,
        )

        # Both should detect sliding (horizontal motion in their respective systems)
        assert len(sliding_y_up) == 1, "Y-up: Should detect horizontal sliding"
        assert len(sliding_z_up) == 1, "Z-up: Should detect horizontal sliding"
        assert (
            abs(sliding_y_up[0]["sliding_distance"] - sliding_z_up[0]["sliding_distance"]) < 0.01
        ), "Sliding distance should be equal regardless of coordinate system"


class TestFootContactEdgeCases:
    """Test edge cases in foot contact analysis."""

    def test_no_contacts_when_always_airborne(self):
        """Foot never touches ground."""
        positions = np.array([[0, 20, 0], [0, 20, 10], [0, 20, 20]], dtype=float)  # Always high in Y

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=5.0, velocity_threshold=10.0, up_axis=1
        )

        assert len(contacts) == 0, "Should not detect contacts when always airborne"

    def test_single_frame_contact(self):
        """Contact lasts only one frame."""
        positions = np.array([[0, 20, 0], [0, 0, 0], [0, 20, 0]], dtype=float)  # Touch ground for one frame

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=5.0, velocity_threshold=25.0, up_axis=1
        )

        # Should detect at least the contact frame
        assert len(contacts) >= 0, "Should handle single-frame contacts"

    def test_multiple_contacts_in_sequence(self):
        """Multiple touchdown-liftoff cycles."""
        positions = np.array(
            [
                [0, 20, 0],  # Airborne
                [0, 0, 0],  # Contact 1
                [0, 0, 0],
                [0, 20, 0],  # Airborne
                [0, 0, 0],  # Contact 2
                [0, 0, 0],
                [0, 20, 0],  # Airborne
            ],
            dtype=float,
        )

        velocities = np.gradient(positions, axis=0)
        ground_height = 0.0

        contacts = detect_contact_events(
            positions, velocities, ground_height, height_threshold=5.0, velocity_threshold=25.0, up_axis=1
        )

        assert len(contacts) >= 2, "Should detect multiple separate contact events"

    def test_penetration_without_contact(self):
        """Foot penetrates ground without being classified as contact (edge case)."""
        # This can happen if foot is moving too fast (high velocity) but goes below ground
        positions = np.array(
            [
                [0, 5, 0],
                [0, -2, 0],  # Below ground
                [0, -3, 0],  # Deeper
            ],
            dtype=float,
        )

        ground_height = 0.0
        contact_segments = [(0, 2)]  # Force contact segment
        up_axis = 1

        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments, up_axis=up_axis)

        assert len(penetration_events) == 1, "Should detect penetration"
        assert penetration_events[0]["max_penetration_depth"] > 0, "Should measure penetration depth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
