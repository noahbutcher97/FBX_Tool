"""
Gait Analysis Module

Performs stride segmentation and gait phase analysis from foot motion data.

Key metrics:
- Stride segmentation: Detects individual stride cycles from foot contacts
- Cycle rate: Number of complete gait cycles per second (NOT contact rate)
- Stride length: Horizontal distance traveled per stride
- Phase shift: Temporal offset between left and right feet
- Asymmetry: Difference in stride characteristics between legs
"""

import csv

import fbx
import numpy as np

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import (
    build_bone_hierarchy,
    detect_chains_from_hierarchy,
    fbx_vector_to_array,
    prepare_output_file,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Gait Confidence Calculation
# Formula: mean(π / (1 + |velocity|))
# This normalizes velocity-based confidence to [0, π] range
# Higher values indicate smoother, more consistent foot motion during stance
GAIT_CONFIDENCE_NORMALIZER = np.pi

# Gait Type Classification
# Phase shift threshold (in seconds) between left and right foot contacts
# < 0.1s = Running (feet contact ground nearly simultaneously)
# >= 0.1s = Walking (distinct alternating foot pattern)
GAIT_RUN_PHASE_THRESHOLD_SECONDS = 0.1

# Foot Bone Detection
# Keywords used to identify foot bones when using name-based detection
FOOT_BONE_KEYWORDS = ["foot", "ankle", "tarsal"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def detect_foot_bone(chain):
    """
    Detect the foot bone in a leg chain using name-based heuristics.

    Searches chain in reverse order for bones containing common foot keywords.
    Falls back to second-to-last bone if no matches found.

    Args:
        chain (list): List of bone names in the leg chain

    Returns:
        str: Name of the foot bone
    """
    # Search from end of chain (feet are typically at the end)
    for bone in reversed(chain):
        bone_lower = bone.lower()
        if any(keyword in bone_lower for keyword in FOOT_BONE_KEYWORDS):
            return bone

    # Fallback: assume second-to-last bone is foot (before toes)
    return chain[-2] if len(chain) >= 2 else chain[-1]


def compute_gait_confidence(segment_velocity):
    """
    Compute confidence score for a stride segment based on velocity smoothness.

    Formula: mean(π / (1 + |velocity|))

    Rationale:
    - Smooth velocity changes (low |v|) → high confidence (approaches π)
    - Erratic velocity changes (high |v|) → low confidence (approaches 0)
    - π normalizes output to a consistent range

    Args:
        segment_velocity: Array of velocity values during stride

    Returns:
        float: Confidence score in range [0, π]
    """
    return np.mean(GAIT_CONFIDENCE_NORMALIZER / (1 + np.abs(segment_velocity)))


def calculate_asymmetry(left_strides, right_strides):
    """
    Calculate left-right gait asymmetry based on stride timing.

    Asymmetry measures the relative difference in average stride times
    between left and right legs. Perfect symmetry = 0.0.

    Args:
        left_strides: List of stride data for left leg
        right_strides: List of stride data for right leg

    Returns:
        float: Asymmetry ratio [0, 1] where 0=perfect symmetry, 1=complete asymmetry
    """
    if not left_strides or not right_strides:
        return 0.0

    # Extract cycle times (index 4 in stride data)
    left_times = [s[4] for s in left_strides]
    right_times = [s[4] for s in right_strides]

    avg_left = np.mean(left_times)
    avg_right = np.mean(right_times)

    # Asymmetry as relative difference
    max_time = max(avg_left, avg_right)
    if max_time == 0:
        return 0.0

    asymmetry = abs(avg_left - avg_right) / max_time
    return float(asymmetry)


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================


def analyze_gait(scene, output_dir="output/"):
    """
    Performs stride segmentation and gait phase analysis.

    Analyzes foot motion to extract gait parameters:
    1. Detects foot contact events from vertical velocity zero-crossings
    2. Segments strides between consecutive contacts
    3. Computes stride metrics (length, duration, confidence)
    4. Analyzes left-right coordination and asymmetry
    5. Classifies gait type (walk vs. run) based on phase shift

    Args:
        scene: FBX scene object with animated skeleton
        output_dir (str): Output directory path (default: "output/")

    Returns:
        tuple: (stride_segments, gait_summary)
            stride_segments: List of [chain, start, end, phase, time, confidence, length, asymmetry]
            gait_summary: Dictionary with aggregated metrics
                - cycle_rate: Complete gait cycles per second (left+right)
                - confidence: Average stride confidence across all strides
                - gait_type: "GaitRun" or "GaitWalk" based on phase shift
    """
    # Get animation timing information
    anim_info = get_scene_metadata(scene)
    start = anim_info["start_time"]
    stop = anim_info["stop_time"]
    rate = anim_info["frame_rate"]
    frame_time = 1.0 / rate  # Compute frame time from frame rate
    duration = stop - start

    # Build skeleton hierarchy and detect leg chains
    hierarchy = build_bone_hierarchy(scene)
    chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

    # Extract joint transform data across all frames
    joint_data = {}
    current = start
    while current <= stop:
        t = fbx.FbxTime()
        t.SetSecondDouble(current)
        for child, parent in hierarchy.items():
            node = scene.FindNodeByName(child)
            if not node:
                continue
            child_g = node.EvaluateGlobalTransform(t)
            if parent:
                pnode = scene.FindNodeByName(parent)
                rel = pnode.EvaluateGlobalTransform(t).Inverse() * child_g
            else:
                rel = child_g
            rT = rel.GetT()
            rT_arr = fbx_vector_to_array(rT)

            key = (parent if parent else "Root", child)
            joint_data.setdefault(key, []).append([rT_arr[0], rT_arr[1], rT_arr[2]])
        current += frame_time

    # Analyze each leg chain
    legs = ["LeftLeg", "RightLeg"]
    stride_segments = []
    summary_info = []

    for cname in legs:
        if cname not in chains:
            print(f"⚠ Warning: Chain '{cname}' not found in skeleton. Available chains: {list(chains.keys())}")
            continue

        bones = chains[cname]
        foot = detect_foot_bone(bones)  # ✅ FIXED: Robust foot detection
        parent = hierarchy.get(foot)
        key = (parent, foot)

        if key not in joint_data:
            print(f"⚠ Warning: No joint data found for {cname} foot bone '{foot}'. Skipping gait analysis.")
            continue

        pos_data = np.array(joint_data[key])

        # Extract vertical (Y) and horizontal (Z) positions
        pos_vertical = pos_data[:, 1]  # Y component - for contact detection
        pos_horizontal = pos_data[:, 2]  # Z component - for stride length ✅ FIXED
        frames = np.arange(len(pos_vertical))

        # Calculate vertical velocity for contact detection
        vel = np.diff(pos_vertical, prepend=pos_vertical[0])

        # Detect foot contacts (velocity changes from negative to positive)
        # This indicates foot transitioning from descending to ascending
        contacts_idx = []
        for i in range(1, len(vel) - 1):
            if vel[i - 1] <= 0 and vel[i] >= 0:
                contacts_idx.append(i)

        if len(contacts_idx) < 2:
            print(
                f"⚠ Warning: {cname} has insufficient foot contacts ({len(contacts_idx)} found, need 2+). Skipping gait analysis."
            )
            continue

        # Segment strides between consecutive contacts
        last = contacts_idx[0]
        for c in contacts_idx[1:]:
            # Validate array bounds
            if c <= last:
                print(f"⚠ Warning: Invalid stride segment {cname} [{last}:{c}] - skipping")
                continue

            segment_vel = vel[last:c]
            if len(segment_vel) == 0:
                print(f"⚠ Warning: Empty velocity segment for {cname} - skipping")
                continue

            # Compute stride metrics
            stride_time = (c - last) / rate  # ✅ OPTIMIZED: Direct calculation
            stride_length = float(pos_horizontal[c] - pos_horizontal[last])  # ✅ FIXED: Horizontal distance
            confidence = compute_gait_confidence(segment_vel)

            # Asymmetry will be calculated after both legs processed
            stride_segments.append(
                [
                    cname,
                    int(last),
                    int(c),
                    "Stride",
                    float(stride_time),
                    round(float(confidence), 4),
                    stride_length,
                    0.0,  # Placeholder for asymmetry
                ]
            )
            last = c

        # Cross-correlation between left and right foot for phase analysis
        if cname == "LeftLeg" and "RightLeg" in chains:
            right_bones = chains["RightLeg"]
            right_foot = detect_foot_bone(right_bones)
            p2 = hierarchy.get(right_foot)
            key_r = (p2, right_foot)

            if key_r in joint_data:
                pL_data = np.array(joint_data[(parent, foot)])
                pR_data = np.array(joint_data[key_r])

                # Use Z position (forward/back) for phase analysis
                pL = pL_data[:, 2]
                pR = pR_data[:, 2]

                # Normalize and correlate to find phase shift
                pL_norm = pL - np.mean(pL)
                pR_norm = pR - np.mean(pR)
                corr = np.correlate(pL_norm, pR_norm, "full")
                shift = (np.argmax(corr) - len(pL) + 1) / rate

                # Calculate stride statistics for this leg
                leg_strides = [s for s in stride_segments if s[0] == cname]
                cycle_time = np.mean([s[4] for s in leg_strides]) if leg_strides else 0

                summary_info.append(
                    [
                        cname,
                        len(contacts_idx),
                        duration,
                        float(cycle_time),
                        float(np.mean(pos_vertical)),  # Mean foot height
                        round(float(abs(shift)), 4),
                        "GaitRun" if abs(shift) < GAIT_RUN_PHASE_THRESHOLD_SECONDS else "GaitWalk",
                    ]
                )

    # ✅ FIXED: Calculate actual asymmetry between legs
    left_strides = [s for s in stride_segments if s[0] == "LeftLeg"]
    right_strides = [s for s in stride_segments if s[0] == "RightLeg"]
    asymmetry = calculate_asymmetry(left_strides, right_strides)

    # Update all stride segments with computed asymmetry
    for stride in stride_segments:
        stride[7] = round(asymmetry, 4)

    # Write stride segments CSV
    output_path = output_dir + "chain_gait_segments.csv"
    prepare_output_file(output_path)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Chain", "FrameStart", "FrameEnd", "Phase", "CycleTime", "Confidence", "StrideLength", "Asymmetry"])
        w.writerows(stride_segments)

    # Write gait summary CSV
    output_path = output_dir + "gait_summary.csv"
    prepare_output_file(output_path)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Chain",
                "ContactCount",
                "Duration",
                "MeanCycleTime",
                "MeanStrideHeight",
                "LeftRightPhaseShift",
                "GaitType",
            ]
        )
        w.writerows(summary_info)

    # ✅ FIXED: Correct cycle rate calculation
    # A full gait cycle involves BOTH feet (left contact + right contact)
    # So cycle_rate = contacts/duration/2
    if summary_info and summary_info[0][2] > 0:
        total_contacts = summary_info[0][1]
        cycle_rate = float(total_contacts / summary_info[0][2] / 2.0)  # Divide by 2 for full cycles
    else:
        cycle_rate = 0.0

    gait_summary = {
        "cycle_rate": cycle_rate,
        "confidence": float(np.mean([s[5] for s in stride_segments])) if stride_segments else 0.0,
        "gait_type": summary_info[0][6] if summary_info else "Unknown",
        "asymmetry": asymmetry,
    }

    return stride_segments, gait_summary
