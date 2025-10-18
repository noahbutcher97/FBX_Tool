"""
Motion Classification & Summary Module

Generates natural language descriptions of animation content.

This is the culmination of all motion analysis modules, combining:
- Root motion (spatial movement)
- Directional changes (turning behavior)
- Motion transitions (locomotion changes)
- Gait analysis (stride patterns)
- Temporal segmentation (movement phrases)

Produces AI-readable summaries like:
"Character begins idle, starts walking forward, accelerates into a run,
executes a sharp left turn, decelerates to a walk, and comes to a stop."

Outputs:
- motion_summary.txt: Human-readable narrative description
- motion_classification.json: Structured classification data for AI
- segment_descriptions.csv: Per-segment natural language descriptions
- animation_metadata.json: Complete metadata package for LLM consumption
"""

import json
import csv
import os
from fbx_tool.analysis.utils import ensure_output_dir


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Natural language templates for segment descriptions
SEGMENT_TEMPLATES = {
    'idle_stationary': "standing still",
    'walking_forward': "walking forward",
    'walking_backward': "walking backward",
    'walking_strafe_left': "walking while strafing left",
    'walking_strafe_right': "walking while strafing right",
    'running_forward': "running forward",
    'running_backward': "running backward",
    'sprinting_forward': "sprinting forward",
    'jumping': "jumping",
    'landing': "landing from a jump",
}

TRANSITION_TEMPLATES = {
    'start_moving': "begins moving",
    'stop_moving': "comes to a stop",
    'accelerate': "accelerates",
    'decelerate': "decelerates",
    'direction_change': "changes direction",
    'takeoff': "jumps",
    'landing': "lands",
}

TURNING_TEMPLATES = {
    'slight': "makes a slight turn",
    'moderate': "turns",
    'sharp': "makes a sharp turn",
    'very_sharp': "spins around",
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def describe_segment(segment):
    """
    Generate natural language description for a segment.

    Args:
        segment: Segment dict with motion_state, direction, is_turning

    Returns:
        str: Natural language description
    """
    composite_label = segment.get('composite_label', '')
    motion_state = segment.get('motion_state', 'unknown')
    direction = segment.get('direction', 'unknown')
    is_turning = segment.get('is_turning', False)

    # Try to find exact match in templates
    if composite_label in SEGMENT_TEMPLATES:
        description = SEGMENT_TEMPLATES[composite_label]
    else:
        # Fallback: construct description from components
        state_desc = motion_state.replace('_', ' ')
        direction_desc = direction.replace('_', ' ')

        if direction == "stationary":
            description = f"{state_desc} in place"
        elif direction == "forward":
            description = f"{state_desc} forward"
        elif direction == "backward":
            description = f"{state_desc} backward"
        else:
            description = f"{state_desc} while moving {direction_desc}"

    # Add turning qualifier
    if is_turning:
        description += " while turning"

    return description


def describe_transition(transition):
    """
    Generate natural language description for a transition.

    Args:
        transition: Transition dict with transition_type

    Returns:
        str: Natural language description
    """
    transition_type = transition.get('transition_type', 'other')

    if transition_type in TRANSITION_TEMPLATES:
        return TRANSITION_TEMPLATES[transition_type]
    else:
        return "transitions"


def generate_narrative_summary(segments, transitions, turning_events, gait_summary=None):
    """
    Generate a flowing narrative description of the animation.

    Args:
        segments: List of temporal segments
        transitions: List of segment transitions
        turning_events: List of turning events
        gait_summary: Optional gait analysis summary

    Returns:
        str: Natural language narrative
    """
    narrative_parts = []

    # Opening
    if segments:
        first_segment = segments[0]
        narrative_parts.append(f"The character {describe_segment(first_segment)}")

    # Middle: describe transitions and key segments
    for i, transition in enumerate(transitions):
        if i + 1 < len(segments):
            next_segment = segments[i + 1]

            # Add transition
            narrative_parts.append(describe_transition(transition))

            # Add next segment description
            segment_desc = describe_segment(next_segment)

            # Check if there's a turning event during this segment
            turning_desc = None
            for turn in turning_events:
                if (turn['start_frame'] >= next_segment['start_frame'] and
                    turn['end_frame'] <= next_segment['end_frame']):
                    severity = turn.get('severity', 'moderate')
                    direction = turn.get('direction', 'left')
                    turning_desc = f"{TURNING_TEMPLATES.get(severity, 'turns')} {direction}"
                    break

            if turning_desc:
                narrative_parts.append(f"while {segment_desc}, {turning_desc}")
            else:
                narrative_parts.append(segment_desc)

    # Ending
    if segments:
        last_segment = segments[-1]
        if last_segment['motion_state'] == 'idle':
            narrative_parts.append("and comes to rest")

    # Join parts into flowing narrative
    narrative = ""
    for i, part in enumerate(narrative_parts):
        if i == 0:
            narrative = part.capitalize()
        elif i == len(narrative_parts) - 1:
            narrative += f", {part}"
        else:
            narrative += f", then {part}"

    narrative += "."

    return narrative


def classify_animation_type(segments, gait_summary=None):
    """
    Classify the overall animation type.

    Args:
        segments: List of temporal segments
        gait_summary: Optional gait analysis summary

    Returns:
        dict: Classification with type and confidence
    """
    # Count segment types
    motion_state_counts = {}
    for seg in segments:
        state = seg.get('motion_state', 'unknown')
        motion_state_counts[state] = motion_state_counts.get(state, 0) + 1

    total_segments = len(segments)
    if total_segments == 0:
        return {'type': 'unknown', 'confidence': 0.0}

    # Determine dominant motion type
    dominant_state = max(motion_state_counts, key=motion_state_counts.get)
    dominance_ratio = motion_state_counts[dominant_state] / total_segments

    # Classify
    if dominance_ratio > 0.8:
        # Very consistent motion
        if dominant_state == 'idle':
            animation_type = 'static_pose'
        elif dominant_state == 'walking':
            animation_type = 'walk_cycle'
        elif dominant_state == 'running':
            animation_type = 'run_cycle'
        else:
            animation_type = f'{dominant_state}_cycle'
        confidence = dominance_ratio
    else:
        # Mixed motion
        if 'running' in motion_state_counts and 'walking' in motion_state_counts:
            animation_type = 'mixed_locomotion'
        elif 'jumping' in motion_state_counts:
            animation_type = 'acrobatic'
        else:
            animation_type = 'varied_movement'
        confidence = 0.7  # Lower confidence for mixed types

    return {
        'type': animation_type,
        'confidence': confidence,
        'dominant_state': dominant_state,
        'state_distribution': motion_state_counts
    }


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def generate_motion_summary(
    segments,
    transitions,
    turning_events,
    root_motion_summary,
    gait_summary=None,
    output_dir="output/"
):
    """
    Generate comprehensive motion classification and natural language summary.

    Args:
        segments: List of temporal segments (from temporal_segmentation)
        transitions: List of segment transitions
        turning_events: List of turning events (from directional_change_detection)
        root_motion_summary: Root motion summary dict
        gait_summary: Optional gait analysis summary dict
        output_dir: Output directory for output files

    Returns:
        dict: Complete motion summary with classification and description
    """
    ensure_output_dir(output_dir)

    print("Generating motion classification and summary...")

    # Generate narrative description
    narrative = generate_narrative_summary(segments, transitions, turning_events, gait_summary)

    # Classify animation type
    classification = classify_animation_type(segments, gait_summary)

    # Generate per-segment descriptions
    segment_descriptions = []
    for i, seg in enumerate(segments):
        segment_descriptions.append({
            'segment_id': i,
            'start_frame': seg['start_frame'],
            'end_frame': seg['end_frame'],
            'duration_seconds': seg['duration_seconds'],
            'motion_state': seg.get('motion_state', ''),
            'direction': seg.get('direction', ''),
            'description': describe_segment(seg)
        })

    # Compile complete metadata for AI consumption
    animation_metadata = {
        'narrative_description': narrative,
        'classification': classification,
        'statistics': {
            'total_duration': sum(seg['duration_seconds'] for seg in segments),
            'segment_count': len(segments),
            'transition_count': len(transitions),
            'turning_event_count': len(turning_events),
        },
        'root_motion': {
            'total_distance': root_motion_summary.get('total_distance', 0),
            'displacement': root_motion_summary.get('displacement', 0),
            'dominant_direction': root_motion_summary.get('dominant_direction', 'unknown'),
        },
        'segments': segment_descriptions,
        'gait_metrics': gait_summary if gait_summary else {}
    }

    # Write natural language summary
    summary_txt_path = os.path.join(output_dir, "motion_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("ANIMATION MOTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Classification: {classification['type']} (confidence: {classification['confidence']:.1%})\n\n")
        f.write(f"Description:\n{narrative}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  - Duration: {animation_metadata['statistics']['total_duration']:.2f} seconds\n")
        f.write(f"  - Segments: {animation_metadata['statistics']['segment_count']}\n")
        f.write(f"  - Distance traveled: {animation_metadata['root_motion']['total_distance']:.2f} units\n")

    # Write structured JSON for AI
    metadata_json_path = os.path.join(output_dir, "animation_metadata.json")
    with open(metadata_json_path, 'w') as f:
        json.dump(animation_metadata, f, indent=2)

    # Write segment descriptions CSV
    if segment_descriptions:
        descriptions_csv_path = os.path.join(output_dir, "segment_descriptions.csv")
        with open(descriptions_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=segment_descriptions[0].keys())
            writer.writeheader()
            writer.writerows(segment_descriptions)

    # Write classification JSON
    classification_json_path = os.path.join(output_dir, "motion_classification.json")
    with open(classification_json_path, 'w') as f:
        json.dump(classification, f, indent=2)

    print(f"✓ Motion summary generated:")
    print(f"  - Animation type: {classification['type']}")
    print(f"  - {len(segment_descriptions)} segments described")
    print(f"\nNarrative: {narrative}")

    return {
        'narrative': narrative,
        'classification': classification,
        'metadata': animation_metadata,
        'segment_descriptions': segment_descriptions
    }
