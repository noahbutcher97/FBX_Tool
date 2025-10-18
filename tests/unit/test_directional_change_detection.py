"""
Unit tests for directional_change_detection module (refactored version)

Tests the refactored analyze_directional_changes function that accepts
a scene parameter and uses cached trajectory extraction.

Tests also cover helper functions:
- Direction transition detection with noise filtering
- Direction change classification (reversal, lateral, stop/start)
- Turning event detection from angular velocity
- Turn severity classification
- Movement segmentation by direction
- Edge cases (empty data, single frame, constant direction)
"""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from fbx_tool.analysis.directional_change_detection import (
    analyze_directional_changes,
    analyze_directional_changes_from_trajectory,
    classify_direction_change,
    classify_turn_severity,
    detect_direction_transitions,
    detect_turning_events,
    segment_by_direction,
)


@pytest.mark.unit
class TestAnalyzeDirectionalChanges:
    """Test the main analyze_directional_changes function with scene parameter."""

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_uses_cached_trajectory(self, mock_open, mock_ensure_dir, mock_extract):
        """Should use extract_root_trajectory to get cached trajectory data."""
        scene = Mock()

        # Setup trajectory extraction result
        trajectory_data = [
            {"direction": "forward", "angular_velocity_y": 0.0, "rotation_y": 0.0, "frame": 0},
            {"direction": "forward", "angular_velocity_y": 0.0, "rotation_y": 0.0, "frame": 1},
            {"direction": "backward", "angular_velocity_y": 50.0, "rotation_y": 5.0, "frame": 2},
        ]

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        # Execute
        result = analyze_directional_changes(scene, output_dir="test_output")

        # Verify extract_root_trajectory was called
        mock_extract.assert_called_once_with(scene)

        # Verify result structure
        assert "direction_transitions_count" in result
        assert "turning_events_count" in result
        assert "movement_segments_count" in result

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_processes_transitions(self, mock_open, mock_ensure_dir, mock_extract):
        """Should detect direction transitions from trajectory."""
        scene = Mock()

        # Create trajectory with clear direction change
        trajectory_data = []
        for i in range(30):
            direction = "forward" if i < 15 else "backward"
            trajectory_data.append({"direction": direction, "angular_velocity_y": 0.0, "rotation_y": 0.0, "frame": i})

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_directional_changes(scene)

        # Should detect the forward→backward transition
        assert result["direction_transitions_count"] >= 1
        assert len(result["direction_transitions"]) >= 1

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_detects_turning_events(self, mock_open, mock_ensure_dir, mock_extract):
        """Should detect turning events from angular velocity."""
        scene = Mock()

        # Create trajectory with turning
        trajectory_data = []
        for i in range(30):
            # Simulate a turn from frame 10-20
            angular_vel = 60.0 if 10 <= i < 20 else 0.0
            rotation = (i - 10) * 2.0 if i >= 10 else 0.0

            trajectory_data.append(
                {"direction": "forward", "angular_velocity_y": angular_vel, "rotation_y": rotation, "frame": i}
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_directional_changes(scene)

        # Should detect turning event
        assert result["turning_events_count"] >= 1
        assert len(result["turning_events"]) >= 1

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_creates_movement_segments(self, mock_open, mock_ensure_dir, mock_extract):
        """Should create movement segments grouped by direction."""
        scene = Mock()

        # Create trajectory with multiple directions
        trajectory_data = []
        for i in range(50):
            if i < 15:
                direction = "forward"
            elif i < 30:
                direction = "backward"
            else:
                direction = "strafe_left"

            trajectory_data.append({"direction": direction, "angular_velocity_y": 0.0, "rotation_y": 0.0, "frame": i})

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_directional_changes(scene)

        # Should create segments
        assert result["movement_segments_count"] >= 1
        assert len(result["movement_segments"]) >= 1

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_writes_csv_files(self, mock_open, mock_ensure_dir, mock_extract):
        """Should write CSV files for directional changes, turning events, and segments."""
        scene = Mock()

        # Create trajectory with transitions, turns, and segments
        trajectory_data = []
        for i in range(30):
            direction = "forward" if i < 15 else "backward"
            angular_vel = 60.0 if 10 <= i < 20 else 0.0
            rotation = (i - 10) * 2.0 if i >= 10 else 0.0

            trajectory_data.append(
                {"direction": direction, "angular_velocity_y": angular_vel, "rotation_y": rotation, "frame": i}
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_directional_changes(scene, output_dir="output")

        # Should write CSV files (potentially 3: directional_changes, turning_events, movement_segments)
        # At least one file should be written
        assert mock_open.call_count >= 1


@pytest.mark.unit
class TestDirectionTransitionDetection:
    """Test direction transition detection with noise filtering."""

    def test_detect_single_direction_change(self):
        """Should detect a single clear direction change."""
        direction_sequence = ["forward"] * 20 + ["backward"] * 20
        frame_rate = 30.0

        transitions = detect_direction_transitions(direction_sequence, frame_rate)

        assert len(transitions) == 1
        assert transitions[0]["from_direction"] == "forward"
        assert transitions[0]["to_direction"] == "backward"
        assert transitions[0]["change_type"] == "reversal"

    def test_detect_multiple_direction_changes(self):
        """Should detect multiple direction changes."""
        direction_sequence = ["forward"] * 15 + ["strafe_left"] * 15 + ["backward"] * 15 + ["forward"] * 15
        frame_rate = 30.0

        transitions = detect_direction_transitions(direction_sequence, frame_rate)

        assert len(transitions) >= 3

    def test_noise_filtering_requires_stable_frames(self):
        """Should filter out brief noise by requiring DIRECTION_STABLE_FRAMES."""
        direction_sequence = ["forward"] * 10 + ["backward"] + ["forward"] * 10 + ["backward"] * 2 + ["forward"] * 10
        frame_rate = 30.0

        transitions = detect_direction_transitions(direction_sequence, frame_rate)

        assert len(transitions) == 0

    def test_empty_direction_sequence(self):
        """Should handle empty sequence gracefully."""
        direction_sequence = []
        frame_rate = 30.0

        transitions = detect_direction_transitions(direction_sequence, frame_rate)

        assert transitions == []


@pytest.mark.unit
class TestDirectionChangeClassification:
    """Test classification of different direction change types."""

    def test_classify_start_from_stationary(self):
        """Should classify movement from stationary as 'start'."""
        change_type = classify_direction_change("stationary", "forward")
        assert change_type == "start"

    def test_classify_stop_to_stationary(self):
        """Should classify movement to stationary as 'stop'."""
        change_type = classify_direction_change("forward", "stationary")
        assert change_type == "stop"

    def test_classify_forward_backward_reversal(self):
        """Should classify forward→backward as 'reversal'."""
        change_type = classify_direction_change("forward", "backward")
        assert change_type == "reversal"

    def test_classify_lateral_switch(self):
        """Should classify strafe-to-strafe as 'lateral_switch'."""
        change_type = classify_direction_change("strafe_left", "strafe_right")
        assert change_type == "lateral_switch"

    def test_classify_lateral_shift_forward_to_strafe(self):
        """Should classify forward→strafe as 'lateral_shift'."""
        change_type = classify_direction_change("forward", "strafe_left")
        assert change_type == "lateral_shift"


@pytest.mark.unit
class TestTurningEventDetection:
    """Test detection of turning events from angular velocity."""

    def test_detect_single_turn_event(self):
        """Should detect a single turning event."""
        n_frames = 100
        angular_velocity_y = np.zeros(n_frames)
        angular_velocity_y[30:50] = 45.0

        rotations_y = np.zeros(n_frames)
        for i in range(30, 50):
            rotations_y[i] = rotations_y[i - 1] + (45.0 / 30.0)

        frame_rate = 30.0

        turning_events = detect_turning_events(angular_velocity_y, rotations_y, frame_rate)

        assert len(turning_events) >= 1
        turn = turning_events[0]
        assert turn["direction"] in ["left", "right"]
        assert turn["total_angle"] > 0

    def test_no_turns_detected_when_stationary(self):
        """Should detect no turns when angular velocity is zero."""
        n_frames = 100
        angular_velocity_y = np.zeros(n_frames)
        rotations_y = np.zeros(n_frames)
        frame_rate = 30.0

        turning_events = detect_turning_events(angular_velocity_y, rotations_y, frame_rate)

        assert len(turning_events) == 0


@pytest.mark.unit
class TestTurnSeverityClassification:
    """Test classification of turn severity based on angle."""

    def test_classify_slight_turn(self):
        """Should classify turns < 45° as 'slight'."""
        severity = classify_turn_severity(30.0)
        assert severity == "slight"

    def test_classify_moderate_turn(self):
        """Should classify turns 45° - 90° as 'moderate'."""
        severity = classify_turn_severity(70.0)
        assert severity == "moderate"

    def test_classify_sharp_turn(self):
        """Should classify turns 90° - 180° as 'sharp'."""
        severity = classify_turn_severity(120.0)
        assert severity == "sharp"

    def test_classify_very_sharp_turn(self):
        """Should classify turns >= 180° as 'very_sharp'."""
        severity = classify_turn_severity(200.0)
        assert severity == "very_sharp"


@pytest.mark.unit
class TestMovementSegmentation:
    """Test segmentation of movement by direction."""

    def test_segment_single_direction(self):
        """Should create one segment for constant direction."""
        direction_sequence = ["forward"] * 50
        frame_rate = 30.0

        segments = segment_by_direction(direction_sequence, frame_rate)

        assert len(segments) == 1
        assert segments[0]["direction"] == "forward"
        assert segments[0]["duration_frames"] == 50

    def test_segment_multiple_directions(self):
        """Should create multiple segments for direction changes."""
        direction_sequence = ["forward"] * 20 + ["backward"] * 20 + ["strafe_left"] * 20
        frame_rate = 30.0

        segments = segment_by_direction(direction_sequence, frame_rate)

        assert len(segments) == 3
        assert segments[0]["direction"] == "forward"
        assert segments[1]["direction"] == "backward"
        assert segments[2]["direction"] == "strafe_left"

    def test_empty_direction_sequence(self):
        """Should handle empty sequence gracefully."""
        direction_sequence = []
        frame_rate = 30.0

        segments = segment_by_direction(direction_sequence, frame_rate)

        assert segments == []


@pytest.mark.unit
class TestIntegratedAnalysis:
    """Test end-to-end analysis from trajectory data."""

    def test_analyze_from_trajectory_basic(self):
        """Should analyze trajectory data and return summary."""
        n_frames = 60
        trajectory_data = []

        for i in range(n_frames):
            if i < 30:
                direction = "forward"
                angular_vel = 0.0
                rotation = 0.0
            else:
                direction = "backward"
                angular_vel = 60.0
                rotation = (i - 30) * 2.0

            trajectory_data.append({"direction": direction, "angular_velocity_y": angular_vel, "rotation_y": rotation})

        frame_rate = 30.0

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_directional_changes_from_trajectory(trajectory_data, frame_rate, output_dir=tmpdir)

        assert "direction_transitions_count" in result
        assert "turning_events_count" in result
        assert "movement_segments_count" in result
        assert isinstance(result["direction_transitions"], list)
        assert isinstance(result["turning_events"], list)
        assert isinstance(result["movement_segments"], list)

    def test_analyze_empty_trajectory_raises_error(self):
        """Should raise error for empty trajectory data."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No trajectory data"):
                analyze_directional_changes_from_trajectory([], 30.0, output_dir=tmpdir)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_direction_transitions_with_very_short_sequence(self):
        """Should handle sequence shorter than stability requirement."""
        direction_sequence = ["forward", "backward"]
        frame_rate = 30.0

        transitions = detect_direction_transitions(direction_sequence, frame_rate)

        assert transitions == []

    def test_segment_by_direction_single_frame(self):
        """Should handle single-frame sequence."""
        direction_sequence = ["forward"]
        frame_rate = 30.0

        segments = segment_by_direction(direction_sequence, frame_rate)

        # Single frame is too short for SEGMENT_MIN_FRAMES
        assert len(segments) == 0

    @patch("fbx_tool.analysis.directional_change_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.directional_change_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_directional_changes_with_minimal_trajectory(self, mock_open, mock_ensure_dir, mock_extract):
        """Should handle minimal trajectory gracefully."""
        scene = Mock()

        # Minimal trajectory - just one frame
        trajectory_data = [{"direction": "stationary", "angular_velocity_y": 0.0, "rotation_y": 0.0, "frame": 0}]

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_directional_changes(scene)

        # Should complete without error
        assert result is not None
        assert result["direction_transitions_count"] == 0
        assert result["turning_events_count"] == 0
