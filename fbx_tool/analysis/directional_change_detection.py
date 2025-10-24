"""
Directional Change Detection Module

Detects and analyzes changes in movement direction and turning behavior.

This module processes root motion trajectory data to identify:
- Direction transitions (forward → backward, forward → strafe, etc.)
- Turning events (left/right turns with varying intensity)
- Sudden vs gradual directional changes
- Movement pattern segmentation by direction

This enables natural language descriptions like:
"character walks forward, then turns sharply left, moves backward briefly,
before turning around and walking forward again"

Outputs:
- directional_changes.csv: Detected direction change events with timing and severity
- turning_events.csv: Detected turning events with angle and speed classification
- movement_segments.csv: Continuous movement segments grouped by direction
"""

import csv
import os

import numpy as np

from fbx_tool.analysis.utils import ensure_output_dir, extract_root_trajectory

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Directional change detection
DIRECTION_STABLE_FRAMES = 5  # Minimum frames to confirm direction change (avoids noise)
DIRECTION_CHANGE_MIN_ANGLE = 30.0  # Minimum angle change to register as directional change (degrees)

# Turning event detection
TURNING_EVENT_MIN_ANGLE = 15.0  # Minimum total rotation to register as a turn (degrees)
TURNING_EVENT_MIN_DURATION = 3  # Minimum frames for a turn event

# Turning severity classification (total angle rotated)
TURNING_ANGLE_SLIGHT = 45.0  # < 45° = slight turn
TURNING_ANGLE_MODERATE = 90.0  # < 90° = moderate turn
TURNING_ANGLE_SHARP = 180.0  # < 180° = sharp turn
# >= 180° = very sharp turn (U-turn or spin)

# Movement segment minimum duration
SEGMENT_MIN_FRAMES = 10  # Minimum frames for a valid movement segment


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def detect_direction_transitions(direction_sequence, frame_rate):
    """
    Detect transitions between movement directions.

    Filters out transient noise by requiring direction to be stable
    for DIRECTION_STABLE_FRAMES before confirming a change.

    Args:
        direction_sequence: List of direction classifications per frame
        frame_rate: Animation frame rate

    Returns:
        list: Direction transition events with timing and metadata
    """
    if len(direction_sequence) < DIRECTION_STABLE_FRAMES:
        return []

    transitions = []
    current_direction = direction_sequence[0]
    current_start_frame = 0
    confirmed_direction = current_direction
    stable_count = 0

    for frame, direction in enumerate(direction_sequence):
        if direction == confirmed_direction:
            # Continue current direction
            stable_count = 0
        elif direction == current_direction:
            # Same as candidate direction
            stable_count += 1
            if stable_count >= DIRECTION_STABLE_FRAMES:
                # Confirm direction change
                if confirmed_direction != current_direction:
                    duration = (frame - current_start_frame) / frame_rate
                    transitions.append(
                        {
                            "from_direction": confirmed_direction,
                            "to_direction": current_direction,
                            "start_frame": current_start_frame,
                            "end_frame": frame - DIRECTION_STABLE_FRAMES,
                            "transition_frame": frame - DIRECTION_STABLE_FRAMES,
                            "duration_seconds": duration,
                            "change_type": classify_direction_change(confirmed_direction, current_direction),
                        }
                    )
                    confirmed_direction = current_direction
                    current_start_frame = frame - DIRECTION_STABLE_FRAMES
        else:
            # New candidate direction
            current_direction = direction
            stable_count = 1

    return transitions


def classify_direction_change(from_direction, to_direction):
    """
    Classify the type of directional change.

    Args:
        from_direction: Starting direction
        to_direction: Ending direction

    Returns:
        str: Classification ("reversal", "lateral_shift", "stop", "start", "other")
    """
    if from_direction == "stationary":
        return "start"
    elif to_direction == "stationary":
        return "stop"
    elif from_direction == "forward" and to_direction == "backward":
        return "reversal"
    elif from_direction == "backward" and to_direction == "forward":
        return "reversal"
    elif "strafe" in from_direction and "strafe" in to_direction:
        return "lateral_switch"
    elif (from_direction in ["forward", "backward"]) and ("strafe" in to_direction):
        return "lateral_shift"
    elif ("strafe" in from_direction) and (to_direction in ["forward", "backward"]):
        return "lateral_shift"
    else:
        return "other"


def detect_turning_events(angular_velocity_y, rotations_y, frame_rate):
    """
    Detect discrete turning events from angular velocity data.

    Identifies periods of sustained rotation as turn events,
    measuring total angle rotated and classifying severity.

    Args:
        angular_velocity_y: Angular velocity (degrees/second) per frame
        rotations_y: Y-axis rotation (degrees) per frame
        frame_rate: Animation frame rate

    Returns:
        list: Turning events with timing, angle, and classification
    """
    # Import threshold from utils where it's now defined
    from fbx_tool.analysis.utils import _TURNING_THRESHOLD_SLOW

    turning_events = []
    in_turn = False
    turn_start_frame = 0
    turn_start_rotation = 0.0

    for frame, angular_vel in enumerate(angular_velocity_y):
        is_turning = abs(angular_vel) >= _TURNING_THRESHOLD_SLOW

        if is_turning and not in_turn:
            # Start of turn
            turn_start_frame = frame
            turn_start_rotation = rotations_y[frame]
            in_turn = True
        elif not is_turning and in_turn:
            # End of turn
            turn_end_frame = frame - 1
            turn_end_rotation = rotations_y[turn_end_frame]
            turn_duration_frames = turn_end_frame - turn_start_frame + 1

            if turn_duration_frames >= TURNING_EVENT_MIN_DURATION:
                # Calculate total angle rotated
                total_angle = abs(turn_end_rotation - turn_start_rotation)

                if total_angle >= TURNING_EVENT_MIN_ANGLE:
                    # Determine turn direction
                    turn_direction = "left" if (turn_end_rotation - turn_start_rotation) > 0 else "right"

                    # Classify severity
                    severity = classify_turn_severity(total_angle)

                    # Average angular velocity during turn
                    avg_angular_velocity = np.mean(np.abs(angular_velocity_y[turn_start_frame : turn_end_frame + 1]))

                    turning_events.append(
                        {
                            "start_frame": turn_start_frame,
                            "end_frame": turn_end_frame,
                            "duration_frames": turn_duration_frames,
                            "duration_seconds": turn_duration_frames / frame_rate,
                            "total_angle": total_angle,
                            "direction": turn_direction,
                            "severity": severity,
                            "avg_angular_velocity": avg_angular_velocity,
                        }
                    )

            in_turn = False

    # Handle case where animation ends mid-turn
    if in_turn:
        turn_end_frame = len(rotations_y) - 1
        turn_end_rotation = rotations_y[turn_end_frame]
        turn_duration_frames = turn_end_frame - turn_start_frame + 1

        if turn_duration_frames >= TURNING_EVENT_MIN_DURATION:
            total_angle = abs(turn_end_rotation - turn_start_rotation)
            if total_angle >= TURNING_EVENT_MIN_ANGLE:
                turn_direction = "left" if (turn_end_rotation - turn_start_rotation) > 0 else "right"
                severity = classify_turn_severity(total_angle)
                avg_angular_velocity = np.mean(np.abs(angular_velocity_y[turn_start_frame : turn_end_frame + 1]))

                turning_events.append(
                    {
                        "start_frame": turn_start_frame,
                        "end_frame": turn_end_frame,
                        "duration_frames": turn_duration_frames,
                        "duration_seconds": turn_duration_frames / frame_rate,
                        "total_angle": total_angle,
                        "direction": turn_direction,
                        "severity": severity,
                        "avg_angular_velocity": avg_angular_velocity,
                    }
                )

    return turning_events


def classify_turn_severity(total_angle):
    """
    Classify turning severity based on total angle rotated.

    Args:
        total_angle: Total rotation angle in degrees

    Returns:
        str: Severity classification
    """
    if total_angle < TURNING_ANGLE_SLIGHT:
        return "slight"
    elif total_angle < TURNING_ANGLE_MODERATE:
        return "moderate"
    elif total_angle < TURNING_ANGLE_SHARP:
        return "sharp"
    else:
        return "very_sharp"


def segment_by_direction(direction_sequence, frame_rate):
    """
    Segment the animation into continuous movement periods grouped by direction.

    Filters out very short segments to focus on sustained movement patterns.

    Args:
        direction_sequence: List of direction classifications per frame
        frame_rate: Animation frame rate

    Returns:
        list: Movement segments with direction and timing
    """
    if not direction_sequence:
        return []

    segments = []
    current_direction = direction_sequence[0]
    segment_start = 0

    for frame, direction in enumerate(direction_sequence):
        if direction != current_direction:
            # End current segment
            duration_frames = frame - segment_start
            if duration_frames >= SEGMENT_MIN_FRAMES:
                segments.append(
                    {
                        "direction": current_direction,
                        "start_frame": segment_start,
                        "end_frame": frame - 1,
                        "duration_frames": duration_frames,
                        "duration_seconds": duration_frames / frame_rate,
                    }
                )

            # Start new segment
            current_direction = direction
            segment_start = frame

    # Handle final segment
    duration_frames = len(direction_sequence) - segment_start
    if duration_frames >= SEGMENT_MIN_FRAMES:
        segments.append(
            {
                "direction": current_direction,
                "start_frame": segment_start,
                "end_frame": len(direction_sequence) - 1,
                "duration_frames": duration_frames,
                "duration_seconds": duration_frames / frame_rate,
            }
        )

    return segments


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================


def analyze_directional_changes(scene, output_dir="output/"):
    """
    Detect and analyze directional changes and turning behavior.

    Processes root motion trajectory data to identify:
    - Direction transition events
    - Turning events with severity classification
    - Movement segments grouped by direction

    Args:
        scene: FBX scene object
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary of directional changes and turning events
    """
    ensure_output_dir(output_dir)

    # Extract trajectory data using cached utility
    trajectory = extract_root_trajectory(scene)

    # Unpack trajectory data
    trajectory_data = trajectory["trajectory_data"]
    frame_rate = trajectory["frame_rate"]

    # Delegate to the analysis function
    return analyze_directional_changes_from_trajectory(trajectory_data, frame_rate, output_dir)


def analyze_directional_changes_from_trajectory(trajectory_data, frame_rate, output_dir="output/"):
    """
    Detect and analyze directional changes from trajectory data.

    Args:
        trajectory_data: List of trajectory dictionaries (from root_motion_analysis)
        frame_rate: Animation frame rate
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary of directional changes and turning events
    """
    ensure_output_dir(output_dir)

    if not trajectory_data:
        raise ValueError("No trajectory data provided")

    # Extract direction sequence and angular velocity
    direction_sequence = [frame_data["direction"] for frame_data in trajectory_data]
    angular_velocity_y = np.array(
        [frame_data["angular_velocity_yaw"] for frame_data in trajectory_data]
    )  # FIXED: Use procedural yaw axis
    rotations_y = np.array([frame_data["rotation_y"] for frame_data in trajectory_data])

    # Unwrap rotations to handle 360° wrapping for cumulative angle calculation
    rotations_y_unwrapped = np.unwrap(np.radians(rotations_y))
    rotations_y_unwrapped = np.degrees(rotations_y_unwrapped)

    # Detect direction transitions
    direction_transitions = detect_direction_transitions(direction_sequence, frame_rate)

    # Detect turning events
    turning_events = detect_turning_events(angular_velocity_y, rotations_y_unwrapped, frame_rate)

    # Segment by direction
    movement_segments = segment_by_direction(direction_sequence, frame_rate)

    # Write directional changes CSV
    if direction_transitions:
        transitions_csv_path = os.path.join(output_dir, "directional_changes.csv")
        with open(transitions_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=direction_transitions[0].keys())
            writer.writeheader()
            writer.writerows(direction_transitions)

    # Write turning events CSV
    if turning_events:
        turning_csv_path = os.path.join(output_dir, "turning_events.csv")
        with open(turning_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=turning_events[0].keys())
            writer.writeheader()
            writer.writerows(turning_events)

    # Write movement segments CSV
    if movement_segments:
        segments_csv_path = os.path.join(output_dir, "movement_segments.csv")
        with open(segments_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=movement_segments[0].keys())
            writer.writeheader()
            writer.writerows(movement_segments)

    print(f"✓ Directional change analysis complete:")
    print(f"  - {len(direction_transitions)} direction transitions detected")
    print(f"  - {len(turning_events)} turning events detected")
    print(f"  - {len(movement_segments)} movement segments identified")

    # Compute summary statistics
    turn_severity_counts = {}
    for turn in turning_events:
        severity = turn["severity"]
        turn_severity_counts[severity] = turn_severity_counts.get(severity, 0) + 1

    change_type_counts = {}
    for transition in direction_transitions:
        change_type = transition["change_type"]
        change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1

    return {
        "direction_transitions_count": len(direction_transitions),
        "turning_events_count": len(turning_events),
        "movement_segments_count": len(movement_segments),
        "turn_severity_distribution": turn_severity_counts,
        "change_type_distribution": change_type_counts,
        "direction_transitions": direction_transitions,
        "turning_events": turning_events,
        "movement_segments": movement_segments,
    }
