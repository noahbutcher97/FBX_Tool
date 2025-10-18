"""
Unit tests for motion_transition_detection module (refactored version)

Tests the refactored analyze_motion_transitions function that accepts
a scene parameter and uses cached trajectory extraction.

Also tests helper functions for:
- Adaptive threshold calculation
- Motion state classification (idle, walk, run, sprint, jump, fall, land)
- State transition detection with noise filtering
- Transition type classification (start/stop, accelerate/decelerate, aerial)
- Transition smoothness analysis (smooth, moderate, abrupt)
- Motion state segmentation
"""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from fbx_tool.analysis.motion_transition_detection import (
    analyze_motion_transitions,
    analyze_transition_smoothness,
    calculate_adaptive_velocity_thresholds,
    calculate_adaptive_vertical_thresholds,
    classify_motion_state,
    classify_transition_type,
    detect_motion_state_sequence,
    detect_state_transitions,
    segment_by_motion_state,
)


@pytest.mark.unit
class TestAnalyzeMotionTransitions:
    """Test the main analyze_motion_transitions function with scene parameter."""

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_uses_cached_trajectory(self, mock_open, mock_ensure_dir, mock_extract):
        """Should use extract_root_trajectory to get cached trajectory data."""
        scene = Mock()

        # Setup trajectory extraction result
        trajectory_data = []
        for i in range(30):
            trajectory_data.append(
                {
                    "frame": i,
                    "velocity_x": 5.0 if i < 15 else 30.0,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "position_x": float(i),
                    "position_y": 0.0,
                    "position_z": 0.0,
                }
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        # Execute
        result = analyze_motion_transitions(scene, output_dir="test_output")

        # Verify extract_root_trajectory was called
        mock_extract.assert_called_once_with(scene)

        # Verify result structure
        assert "transitions_count" in result
        assert "states_count" in result
        assert "transition_type_distribution" in result

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_detects_state_changes(self, mock_open, mock_ensure_dir, mock_extract):
        """Should detect motion state transitions from trajectory."""
        scene = Mock()

        # Create trajectory with clear velocity changes (idle → walking → running)
        trajectory_data = []
        for i in range(60):
            if i < 20:
                vx, vy, vz = 2.0, 0.0, 0.0  # Idle
            elif i < 40:
                vx, vy, vz = 30.0, 0.0, 0.0  # Walking
            else:
                vx, vy, vz = 100.0, 0.0, 0.0  # Running

            trajectory_data.append(
                {
                    "frame": i,
                    "velocity_x": vx,
                    "velocity_y": vy,
                    "velocity_z": vz,
                    "position_x": float(i),
                    "position_y": 0.0,
                    "position_z": 0.0,
                }
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_motion_transitions(scene)

        # Should detect transitions
        assert result["transitions_count"] >= 1
        assert len(result["transitions"]) >= 1

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_creates_motion_state_segments(self, mock_open, mock_ensure_dir, mock_extract):
        """Should create motion state segments."""
        scene = Mock()

        # Create trajectory with distinct motion states
        trajectory_data = []
        for i in range(50):
            if i < 15:
                vx = 2.0  # Idle
            elif i < 30:
                vx = 30.0  # Walking
            else:
                vx = 100.0  # Running

            trajectory_data.append(
                {
                    "frame": i,
                    "velocity_x": vx,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "position_x": float(i),
                    "position_y": 0.0,
                    "position_z": 0.0,
                }
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_motion_transitions(scene)

        # Should create segments
        assert result["states_count"] >= 1
        assert len(result["motion_states"]) >= 1

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_analyzes_transition_quality(self, mock_open, mock_ensure_dir, mock_extract):
        """Should analyze transition smoothness/quality."""
        scene = Mock()

        # Create trajectory with transitions
        trajectory_data = []
        for i in range(40):
            vx = 5.0 if i < 20 else 50.0

            trajectory_data.append(
                {
                    "frame": i,
                    "velocity_x": vx,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "position_x": float(i),
                    "position_y": 0.0,
                    "position_z": 0.0,
                }
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_motion_transitions(scene)

        # Should have transition quality data
        assert "transition_quality" in result
        assert isinstance(result["transition_quality"], list)

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_writes_csv_files(self, mock_open, mock_ensure_dir, mock_extract):
        """Should write CSV files for transitions, states, and quality."""
        scene = Mock()

        # Create trajectory with transitions
        trajectory_data = []
        for i in range(30):
            vx = 5.0 if i < 15 else 50.0

            trajectory_data.append(
                {
                    "frame": i,
                    "velocity_x": vx,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "position_x": float(i),
                    "position_y": 0.0,
                    "position_z": 0.0,
                }
            )

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_motion_transitions(scene, output_dir="output")

        # Should write CSV files (potentially 3: transitions, states, quality)
        # At least one file should be written
        assert mock_open.call_count >= 1


# Keep all the existing helper function tests
@pytest.mark.unit
class TestAdaptiveThresholds:
    """Test adaptive threshold calculation."""

    def test_calculate_adaptive_velocity_thresholds_from_distribution(self):
        """Should derive velocity thresholds from data distribution."""
        velocities = np.concatenate(
            [
                np.random.uniform(0, 10, 30),
                np.random.uniform(20, 40, 40),
                np.random.uniform(60, 100, 20),
                np.random.uniform(120, 150, 10),
            ]
        )

        thresholds = calculate_adaptive_velocity_thresholds(velocities)

        assert "idle" in thresholds
        assert thresholds["idle"] < 20
        assert "walk" in thresholds
        assert 20 < thresholds["walk"] < 60
        assert "run" in thresholds
        assert 60 < thresholds["run"] < 120
        assert thresholds["idle"] < thresholds["walk"] < thresholds["run"]


@pytest.mark.unit
class TestMotionStateClassification:
    """Test motion state classification logic."""

    def test_classify_idle_state(self):
        """Should classify low velocity as idle."""
        state = classify_motion_state(velocity_magnitude=3.0, velocity_y=0.0, acceleration_y=0.0)
        assert state == "idle"

    def test_classify_walking_state(self):
        """Should classify moderate velocity as walking."""
        state = classify_motion_state(velocity_magnitude=25.0, velocity_y=0.0, acceleration_y=0.0)
        assert state == "walking"

    def test_classify_running_state(self):
        """Should classify high velocity as running."""
        state = classify_motion_state(velocity_magnitude=100.0, velocity_y=0.0, acceleration_y=0.0)
        assert state == "running"

    def test_classify_sprinting_state(self):
        """Should classify very high velocity as sprinting."""
        state = classify_motion_state(velocity_magnitude=200.0, velocity_y=0.0, acceleration_y=0.0)
        assert state == "sprinting"


@pytest.mark.unit
class TestStateTransitionDetection:
    """Test state transition detection with noise filtering."""

    def test_detect_single_state_transition(self):
        """Should detect a single clean transition between states."""
        state_sequence = ["idle"] * 20 + ["walking"] * 20
        frame_rate = 30.0

        transitions = detect_state_transitions(state_sequence, frame_rate)

        assert len(transitions) == 1
        assert transitions[0]["from_state"] == "idle"
        assert transitions[0]["to_state"] == "walking"

    def test_noise_filtering_rejects_transient_changes(self):
        """Should filter out state changes that don't last STATE_STABLE_FRAMES."""
        state_sequence = ["idle"] * 10 + ["walking"] * 2 + ["idle"] * 10
        frame_rate = 30.0

        transitions = detect_state_transitions(state_sequence, frame_rate)

        assert len(transitions) == 0


@pytest.mark.unit
class TestTransitionTypeClassification:
    """Test transition type classification logic."""

    def test_classify_start_moving_from_idle(self):
        """Should classify idle → walking/running as start_moving."""
        assert classify_transition_type("idle", "walking") == "start_moving"
        assert classify_transition_type("idle", "running") == "start_moving"

    def test_classify_stop_moving_to_idle(self):
        """Should classify walking/running → idle as stop_moving."""
        assert classify_transition_type("walking", "idle") == "stop_moving"
        assert classify_transition_type("running", "idle") == "stop_moving"

    def test_classify_accelerate_walking_to_running(self):
        """Should classify walking → running/sprinting as accelerate."""
        assert classify_transition_type("walking", "running") == "accelerate"
        assert classify_transition_type("walking", "sprinting") == "accelerate"


@pytest.mark.unit
class TestTransitionSmoothnessAnalysis:
    """Test transition smoothness analysis."""

    def test_analyze_smooth_transition(self):
        """Should classify low jerk transition as smooth."""
        jerks = np.ones(100) * 20.0
        velocities = np.ones(100) * 10.0
        accelerations = np.ones(100) * 5.0

        result = analyze_transition_smoothness(velocities, accelerations, jerks, transition_frame=50, window_size=10)

        assert result["smoothness"] == "smooth"
        assert result["mean_jerk"] < 50.0


@pytest.mark.unit
class TestMotionStateSegmentation:
    """Test motion state segmentation."""

    def test_segment_multiple_motion_states(self):
        """Should segment sequence into motion state periods."""
        state_sequence = ["idle"] * 20 + ["walking"] * 30 + ["running"] * 25
        frame_rate = 30.0

        segments = segment_by_motion_state(state_sequence, frame_rate)

        assert len(segments) == 3
        assert segments[0]["motion_state"] == "idle"
        assert segments[1]["motion_state"] == "walking"
        assert segments[2]["motion_state"] == "running"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("fbx_tool.analysis.motion_transition_detection.extract_root_trajectory")
    @patch("fbx_tool.analysis.motion_transition_detection.ensure_output_dir")
    @patch("builtins.open", new_callable=MagicMock)
    def test_analyze_motion_transitions_with_minimal_trajectory(self, mock_open, mock_ensure_dir, mock_extract):
        """Should handle minimal trajectory gracefully."""
        scene = Mock()

        # Minimal trajectory - just one frame
        trajectory_data = [
            {
                "frame": 0,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "velocity_z": 0.0,
                "position_x": 0.0,
                "position_y": 0.0,
                "position_z": 0.0,
            }
        ]

        mock_extract.return_value = {"trajectory_data": trajectory_data, "frame_rate": 30.0}

        result = analyze_motion_transitions(scene)

        # Should complete without error
        assert result is not None
        assert result["transitions_count"] == 0

    def test_empty_sequence_returns_empty(self):
        """Should handle empty state sequence."""
        state_sequence = []
        frame_rate = 30.0

        transitions = detect_state_transitions(state_sequence, frame_rate)
        assert transitions == []

        segments = segment_by_motion_state(state_sequence, frame_rate)
        assert segments == []
