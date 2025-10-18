"""
Chain Analysis Module

Performs chain-level IK suitability and cross-temporal coherence analysis.

Key metrics:
- IK Suitability: How well a joint behaves for Inverse Kinematics solving
  - Stability: Low rotation variance = predictable = good for IK
  - Range: Sufficient but not excessive rotation range
- Temporal Coherence: How consistent motion is across time windows
- Chain IK Confidence: Combined metric of IK suitability and temporal smoothness

This helps identify which skeletal chains are suitable for IK retargeting and which
may have motion artifacts or inconsistencies.
"""

import csv

import fbx
import numpy as np

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import (
    build_bone_hierarchy,
    compute_ik_suitability,
    detect_chains_from_hierarchy,
    fbx_vector_to_array,
    prepare_output_file,
)

# ==============================================================================
# CONSTANTS - Temporal Coherence
# ==============================================================================

# Coherence window size (in seconds)
# Shorter windows = detect high-frequency noise
# Longer windows = detect sustained motion patterns
COHERENCE_WINDOW_SECONDS = 0.25  # 250ms windows for correlation analysis

# Final Confidence Weights
# Mechanical suitability (IK score) weighted higher than temporal smoothness
CONFIDENCE_IK_WEIGHT = 0.7  # Primary: can joint solve IK reliably?
CONFIDENCE_TEMPORAL_WEIGHT = 0.3  # Secondary: is motion temporally smooth?

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def compute_temporal_coherence(position_data, frame_rate):
    """
    Compute temporal coherence using non-overlapping window correlations.

    Measures how predictably motion evolves over time by correlating
    non-overlapping windows. This avoids artificially high correlations
    from overlapping data.

    Args:
        position_data: Array of shape (n_frames, 3) - XYZ positions over time
        frame_rate: Animation frame rate (fps)

    Returns:
        float: Mean correlation across non-overlapping windows [0, 1]
               Returns 0.0 if insufficient data or no valid correlations
    """
    frames = len(position_data)
    window_size = int(frame_rate * COHERENCE_WINDOW_SECONDS)

    if frames < 2 * window_size:
        # Need at least 2 windows for comparison
        return 0.0

    correlations = []

    # Use NON-overlapping windows to avoid artificial correlation
    # Compare window N with window N+1
    for i in range(0, frames - 2 * window_size, window_size):
        w1 = position_data[i : i + window_size].flatten()
        w2 = position_data[i + window_size : i + 2 * window_size].flatten()

        if len(w1) == len(w2) and len(w1) > 0:
            corr_matrix = np.corrcoef(w1, w2)
            if corr_matrix.shape == (2, 2):  # Valid correlation matrix
                correlation = corr_matrix[0, 1]
                if np.isfinite(correlation):
                    correlations.append(correlation)

    if not correlations:
        return 0.0

    # Convert correlations from [-1, 1] to coherence [0, 1]
    # Higher absolute correlation = higher coherence
    # Normalize: coherence = (correlation + 1) / 2
    mean_correlation = np.mean(correlations)
    coherence = (mean_correlation + 1.0) / 2.0

    return float(np.clip(coherence, 0.0, 1.0))


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================


def analyze_chains(scene, output_dir="output/"):
    """
    Performs chain-level IK suitability and cross-temporal coherence analysis.

    For each kinematic chain in the skeleton:
    1. Computes per-joint IK suitability (stability + rotation range)
    2. Aggregates joint scores to chain-level IK score
    3. Analyzes temporal coherence (motion predictability)
    4. Combines into final chain IK confidence score

    This analysis helps identify:
    - Which chains are suitable for IK retargeting
    - Which joints have motion artifacts
    - Which chains have temporally inconsistent motion

    Args:
        scene: FBX scene object with animated skeleton
        output_dir (str): Output directory path (default: "output/")

    Returns:
        dict: Chain confidence data
            Keys: chain names (e.g., "LeftArm", "RightLeg")
            Values: {
                "mean_ik": Average IK suitability score [0, 1]
                "cross_temp": Temporal coherence score [0, 1]
                "confidence": Final chain confidence [0, 1]
            }
    """
    # Get animation timing information
    anim_info = get_scene_metadata(scene)
    start = anim_info["start_time"]
    stop = anim_info["stop_time"]
    rate = anim_info["frame_rate"]
    frame_time = 1.0 / rate  # Compute frame time from frame rate

    # Build skeleton hierarchy and detect chains
    hierarchy = build_bone_hierarchy(scene)
    chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

    # Extract joint transform data (translation + rotation) across all frames
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
            rT, rR = rel.GetT(), rel.GetR()
            rT_arr = fbx_vector_to_array(rT)
            rR_arr = fbx_vector_to_array(rR)

            key = (parent if parent else "Root", child)
            joint_data.setdefault(key, []).append(
                [
                    rT_arr[0],
                    rT_arr[1],
                    rT_arr[2],  # Translation XYZ
                    rR_arr[0],
                    rR_arr[1],
                    rR_arr[2],  # Rotation XYZ (Euler angles)
                ]
            )
        current += frame_time

    # Compute per-joint IK suitability scores
    joint_summary = {}
    for joint, vals in joint_data.items():
        arr = np.array(vals)
        rotation_data = arr[:, 3:6]  # Extract rotation columns

        stability, range_score, ik_score = compute_ik_suitability(rotation_data)
        joint_summary[joint] = (stability, range_score, ik_score)

    # Analyze each chain
    chain_results = []
    for cname, chain in chains.items():
        # Build segments with detailed logging
        segs = []
        missing_bones = []
        for b in chain:
            parent = hierarchy.get(b)
            if not parent:
                missing_bones.append(f"{b} (no parent)")
                continue
            if (parent, b) not in joint_data:
                missing_bones.append(f"{b} (no joint data)")
                continue
            segs.append((b, parent))

        if missing_bones:
            print(f"ℹ Chain {cname}: Excluded bones: {', '.join(missing_bones)}")

        if len(segs) < 2:
            print(f"⚠ Chain {cname}: Insufficient valid segments ({len(segs)}/min 2) - skipping")
            continue

        # Aggregate IK scores for all joints in chain
        chain_iks = []
        for bone, parent in segs:  # ✅ FIXED: Use segs directly, no redundant lookup
            joint_key = (parent, bone)
            if joint_key in joint_summary:
                chain_iks.append(joint_summary[joint_key][2])  # Get ik_score

        mean_ik = np.mean(chain_iks) if chain_iks else 0.0

        # Analyze chain tip for temporal coherence
        tip = chain[-1]
        if tip not in hierarchy:
            print(f"⚠ Chain {cname}: Tip bone '{tip}' not in hierarchy - skipping")
            continue

        tip_key = (hierarchy.get(tip), tip)
        if tip_key not in joint_data:
            print(f"⚠ Chain {cname}: No joint data for tip bone '{tip}' - skipping")
            continue

        frames = len(joint_data[tip_key])
        if frames < 3:
            print(f"⚠ Chain {cname}: Insufficient frames ({frames} < 3) - skipping")
            continue

        # Extract position data (first 3 columns) and compute coherence
        position_data = np.array(joint_data[tip_key])[:, :3]
        cross_temporal = compute_temporal_coherence(position_data, rate)  # ✅ FIXED: Non-overlapping windows

        # Combine IK suitability and temporal coherence into final confidence
        final_conf = np.clip(CONFIDENCE_IK_WEIGHT * mean_ik + CONFIDENCE_TEMPORAL_WEIGHT * cross_temporal, 0, 1)

        chain_results.append(
            [cname, round(float(mean_ik), 4), round(float(cross_temporal), 4), round(float(final_conf), 4)]
        )

    # Write results CSV
    output_path = output_dir + "chain_confidence.csv"
    prepare_output_file(output_path)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Chain", "MeanIKSuitability", "CrossTemporalCoherence", "ChainIKConfidence"])
        w.writerows(chain_results)

    # Convert to dictionary for return value
    chain_conf = {row[0]: {"mean_ik": row[1], "cross_temp": row[2], "confidence": row[3]} for row in chain_results}

    return chain_conf
