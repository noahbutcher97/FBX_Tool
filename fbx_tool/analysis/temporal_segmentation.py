"""
Temporal Segmentation Module

Segments animation into coherent temporal blocks based on multiple signals.

This module combines insights from:
- Root motion (direction of travel, turning)
- Motion transitions (walk/run/idle changes)
- Directional changes (forward/backward/strafe shifts)
- Gait analysis (stride patterns)

Creates unified timeline segments that represent distinct "movement phrases"
where the character is performing a consistent action (e.g., "walking forward",
"turning left while running", "idle").

These segments form the foundation for natural language generation.

Outputs:
- temporal_segments.csv: Unified movement segments with timing and classification
- segment_hierarchy.csv: Nested segment structure (coarse to fine)
- segment_transitions.csv: How segments connect and flow
"""

import csv
import os

import numpy as np

from fbx_tool.analysis.utils import ensure_output_dir

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Segment merging thresholds
MIN_SEGMENT_DURATION_SECONDS = 0.5  # Minimum segment length to keep
MAX_MERGE_GAP_SECONDS = 0.2  # Maximum gap to bridge when merging similar segments

# Segment similarity thresholds for merging
DIRECTION_MATCH_THRESHOLD = 0.8  # Fraction of frames with same direction
STATE_MATCH_THRESHOLD = 0.8  # Fraction of frames with same motion state


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def segment_by_continuity(frame_classifications, frame_rate, label_key="state"):
    """
    Segment timeline by continuous classification labels.

    Args:
        frame_classifications: List of dicts with frame-level classifications
        frame_rate: Animation frame rate
        label_key: Key to use for classification (e.g., 'state', 'direction')

    Returns:
        list: Segments with start/end frames and dominant label
    """
    if not frame_classifications:
        return []

    segments = []
    current_label = frame_classifications[0].get(label_key)
    segment_start = 0

    for i, frame_data in enumerate(frame_classifications):
        label = frame_data.get(label_key)

        if label != current_label:
            # End current segment
            duration = (i - segment_start) / frame_rate
            if duration >= MIN_SEGMENT_DURATION_SECONDS:
                segments.append(
                    {
                        "start_frame": segment_start,
                        "end_frame": i - 1,
                        "duration_seconds": duration,
                        "label": current_label,
                    }
                )

            # Start new segment
            current_label = label
            segment_start = i

    # Handle final segment
    duration = (len(frame_classifications) - segment_start) / frame_rate
    if duration >= MIN_SEGMENT_DURATION_SECONDS:
        segments.append(
            {
                "start_frame": segment_start,
                "end_frame": len(frame_classifications) - 1,
                "duration_seconds": duration,
                "label": current_label,
            }
        )

    return segments


def merge_similar_segments(segments, similarity_threshold=0.8):
    """
    Merge adjacent segments with similar characteristics.

    Bridges small gaps between similar segments to reduce fragmentation.

    Args:
        segments: List of segment dicts
        similarity_threshold: Minimum similarity to merge (0-1)

    Returns:
        list: Merged segments
    """
    if len(segments) < 2:
        return segments

    merged = []
    current = segments[0].copy()

    # Determine which label key to use (support both 'label' and 'composite_label')
    label_key = "composite_label" if "composite_label" in current else "label"

    for next_seg in segments[1:]:
        # Check if segments are similar enough to merge
        labels_match = current.get(label_key) == next_seg.get(label_key)
        gap_duration = (next_seg["start_frame"] - current["end_frame"] - 1) / 30.0  # Assume 30fps for gap calc

        if labels_match and gap_duration <= MAX_MERGE_GAP_SECONDS:
            # Merge segments
            current["end_frame"] = next_seg["end_frame"]
            current["duration_seconds"] = (current["end_frame"] - current["start_frame"] + 1) / 30.0
        else:
            # Save current and start new
            merged.append(current)
            current = next_seg.copy()

    # Add final segment
    merged.append(current)

    return merged


def create_composite_segments(motion_states, directions, turning_events, frame_rate):
    """
    Create unified segments combining motion state, direction, and turning.

    Args:
        motion_states: List of motion state segments (idle/walk/run)
        directions: List of direction segments (forward/backward/strafe)
        turning_events: List of turning event dicts
        frame_rate: Animation frame rate

    Returns:
        list: Composite segments with combined classification
    """
    # For now, use motion states as primary segmentation
    # In future iterations, can combine multiple signals
    composite_segments = []

    for motion_seg in motion_states:
        # Find overlapping direction segment
        mid_frame = (motion_seg["start_frame"] + motion_seg["end_frame"]) // 2

        # Find which direction segment contains this frame
        direction_label = "unknown"
        for dir_seg in directions:
            if dir_seg["start_frame"] <= mid_frame <= dir_seg["end_frame"]:
                direction_label = dir_seg.get("label", dir_seg.get("direction", "unknown"))
                break

        # Check if there's a turning event during this segment
        is_turning = False
        for turn_event in turning_events:
            if not (
                turn_event["end_frame"] < motion_seg["start_frame"]
                or turn_event["start_frame"] > motion_seg["end_frame"]
            ):
                is_turning = True
                break

        # Combine labels into composite description
        # Support both 'label' and 'motion_state' keys for backwards compatibility
        motion_type = motion_seg.get("label", motion_seg.get("motion_state", "unknown"))
        composite_label = f"{motion_type}_{direction_label}"
        if is_turning:
            composite_label += "_turning"

        composite_segments.append(
            {
                "start_frame": motion_seg["start_frame"],
                "end_frame": motion_seg["end_frame"],
                "duration_seconds": motion_seg["duration_seconds"],
                "motion_state": motion_type,
                "direction": direction_label,
                "is_turning": is_turning,
                "composite_label": composite_label,
            }
        )

    return composite_segments


def build_segment_hierarchy(segments):
    """
    Build hierarchical segment structure (coarse to fine granularity).

    Groups fine-grained segments into coarser movement "phrases".

    Args:
        segments: List of fine-grained segments

    Returns:
        list: Hierarchical segment structure
    """
    # Simple implementation: group consecutive segments with similar motion state
    hierarchy = []
    current_group = None

    for seg in segments:
        motion_state = seg["motion_state"]

        if current_group is None:
            current_group = {
                "motion_state": motion_state,
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "sub_segments": [seg],
            }
        elif current_group["motion_state"] == motion_state:
            # Extend current group
            current_group["end_frame"] = seg["end_frame"]
            current_group["sub_segments"].append(seg)
        else:
            # Save current group and start new one
            hierarchy.append(current_group)
            current_group = {
                "motion_state": motion_state,
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "sub_segments": [seg],
            }

    # Add final group
    if current_group:
        hierarchy.append(current_group)

    return hierarchy


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================


def analyze_temporal_segmentation(motion_states, movement_segments, turning_events, frame_rate, output_dir="output/"):
    """
    Create unified temporal segmentation from multiple motion analysis outputs.

    Combines:
    - Motion states (from motion_transition_detection)
    - Movement directions (from directional_change_detection)
    - Turning events (from directional_change_detection)

    Args:
        motion_states: List of motion state segments (idle/walk/run/etc.)
        movement_segments: List of directional movement segments
        turning_events: List of turning event dicts
        frame_rate: Animation frame rate
        output_dir: Output directory for CSV files

    Returns:
        dict: Segmentation summary with segments and hierarchy
    """
    ensure_output_dir(output_dir)

    print("Creating unified temporal segmentation...")

    # Create composite segments combining all signals
    composite_segments = create_composite_segments(motion_states, movement_segments, turning_events, frame_rate)

    # Merge similar adjacent segments
    merged_segments = merge_similar_segments(composite_segments)

    # Build hierarchical structure
    segment_hierarchy = build_segment_hierarchy(merged_segments)

    # Analyze segment transitions
    segment_transitions = []
    for i in range(len(merged_segments) - 1):
        current_seg = merged_segments[i]
        next_seg = merged_segments[i + 1]

        transition_gap = (next_seg["start_frame"] - current_seg["end_frame"] - 1) / frame_rate

        segment_transitions.append(
            {
                "from_frame": current_seg["end_frame"],
                "to_frame": next_seg["start_frame"],
                "gap_seconds": transition_gap,
                "from_state": current_seg["composite_label"],
                "to_state": next_seg["composite_label"],
                "transition_type": classify_segment_transition(current_seg, next_seg),
            }
        )

    # Write temporal segments CSV
    if merged_segments:
        segments_csv_path = os.path.join(output_dir, "temporal_segments.csv")
        with open(segments_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_segments[0].keys())
            writer.writeheader()
            writer.writerows(merged_segments)

    # Write segment hierarchy CSV
    if segment_hierarchy:
        hierarchy_data = []
        for i, group in enumerate(segment_hierarchy):
            hierarchy_data.append(
                {
                    "group_id": i,
                    "motion_state": group["motion_state"],
                    "start_frame": group["start_frame"],
                    "end_frame": group["end_frame"],
                    "sub_segment_count": len(group["sub_segments"]),
                }
            )

        hierarchy_csv_path = os.path.join(output_dir, "segment_hierarchy.csv")
        with open(hierarchy_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hierarchy_data[0].keys())
            writer.writeheader()
            writer.writerows(hierarchy_data)

    # Write segment transitions CSV
    if segment_transitions:
        transitions_csv_path = os.path.join(output_dir, "segment_transitions.csv")
        with open(transitions_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=segment_transitions[0].keys())
            writer.writeheader()
            writer.writerows(segment_transitions)

    print(f"✓ Temporal segmentation complete:")
    print(f"  - {len(merged_segments)} unified segments created")
    print(f"  - {len(segment_hierarchy)} high-level movement groups")
    print(f"  - {len(segment_transitions)} segment transitions")

    return {
        "segments_count": len(merged_segments),
        "hierarchy_groups": len(segment_hierarchy),
        "transitions_count": len(segment_transitions),
        "segments": merged_segments,
        "hierarchy": segment_hierarchy,
        "transitions": segment_transitions,
    }


def classify_segment_transition(from_segment, to_segment):
    """
    Classify the type of transition between segments.

    Args:
        from_segment: Starting segment dict
        to_segment: Ending segment dict

    Returns:
        str: Transition classification
    """
    from_state = from_segment["motion_state"]
    to_state = to_segment["motion_state"]

    # State changes
    if from_state == "idle" and to_state in ["walking", "running"]:
        return "start_moving"
    elif from_state in ["walking", "running"] and to_state == "idle":
        return "stop_moving"
    elif from_state == "walking" and to_state == "running":
        return "accelerate"
    elif from_state == "running" and to_state == "walking":
        return "decelerate"

    # Direction changes (same state, different direction)
    elif from_state == to_state:
        from_dir = from_segment.get("direction", "")
        to_dir = to_segment.get("direction", "")

        if from_dir != to_dir:
            return "direction_change"
        else:
            return "continue"

    return "other"
