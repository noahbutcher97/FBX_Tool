"""
Constraint Violation Detection Module

Detects violations of animation constraints including:
- IK chain integrity
- Hierarchy consistency
- Animation curve discontinuities
- Keyframe timing irregularities
- Parent-child relationship violations

Author: FBX Tool
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import diffs
import fbx
import numpy as np

from fbx_tool.analysis.pose_validity_analysis import _extract_bone_animation_data
from fbx_tool.analysis.utils import build_bone_hierarchy, detect_chains_from_hierarchy


def validate_ik_chain_length(
    bone_positions: Dict[str, np.ndarray], chain: List[str], tolerance: float = 0.05, check_orientation: bool = False
) -> List[Dict[str, Any]]:
    """
    Validate IK chain length consistency across frames.

    Args:
        bone_positions: Dictionary mapping bone names to position arrays (frames, 3)
        chain: List of bone names in the chain
        tolerance: Acceptable length variation as fraction (default 5%)
        check_orientation: Whether to check orientation consistency (optional)

    Returns:
        List of violation events
    """
    violations = []

    if len(chain) < 2:
        return violations

    # Compute total chain length for each frame
    num_frames = len(bone_positions[chain[0]])
    chain_lengths = np.zeros(num_frames)

    for i in range(len(chain) - 1):
        bone1 = chain[i]
        bone2 = chain[i + 1]

        if bone1 not in bone_positions or bone2 not in bone_positions:
            continue

        # Compute segment lengths
        vectors = bone_positions[bone2] - bone_positions[bone1]
        segment_lengths = np.linalg.norm(vectors, axis=1)
        chain_lengths += segment_lengths

    # Find reference length (median)
    reference_length = np.median(chain_lengths)

    if reference_length == 0:
        return violations

    # Compute deviations
    deviations = (chain_lengths - reference_length) / reference_length
    deviation_percent = deviations * 100

    # Detect stretching
    stretch_mask = deviations > tolerance
    # Detect compression
    compress_mask = deviations < -tolerance

    def find_segments(mask, violation_type):
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_violation in enumerate(mask):
            if is_violation and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_violation and in_segment:
                segments.append((start_frame, i - 1, violation_type))
                in_segment = False

        if in_segment:
            segments.append((start_frame, len(mask) - 1, violation_type))

        return segments

    # Process stretching violations
    for start, end, vtype in find_segments(stretch_mask, "stretch"):
        segment_deviations = deviation_percent[start : end + 1]
        max_dev = np.max(segment_deviations)
        mean_dev = np.mean(segment_deviations)

        # Classify severity
        if max_dev > 30:
            severity = "high"
        elif max_dev > 15:
            severity = "medium"
        else:
            severity = "low"

        violations.append(
            {
                "frame_start": start,
                "frame_end": end,
                "type": "stretch",
                "max_deviation_percent": max_dev,
                "mean_deviation_percent": mean_dev,
                "severity": severity,
                "chain": " -> ".join(chain),
            }
        )

    # Process compression violations
    for start, end, vtype in find_segments(compress_mask, "compression"):
        segment_deviations = np.abs(deviation_percent[start : end + 1])
        max_dev = np.max(segment_deviations)
        mean_dev = np.mean(segment_deviations)

        # Classify severity
        if max_dev > 30:
            severity = "high"
        elif max_dev > 15:
            severity = "medium"
        else:
            severity = "low"

        violations.append(
            {
                "frame_start": start,
                "frame_end": end,
                "type": "compression",
                "max_deviation_percent": max_dev,
                "mean_deviation_percent": mean_dev,
                "severity": severity,
                "chain": " -> ".join(chain),
            }
        )

    return violations


def detect_chain_breaks(
    bone_positions: Dict[str, np.ndarray], chain: List[str], max_distance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Detect breaks/discontinuities in bone chains.

    A break is detected when there's a sudden CHANGE in velocity (acceleration),
    not just fast movement. This distinguishes discontinuities from smooth motion.

    Args:
        bone_positions: Dictionary mapping bone names to position arrays
        chain: List of bone names in the chain
        max_distance: Maximum allowed sudden change in displacement (acceleration threshold)

    Returns:
        List of break events
    """
    breaks = []

    if len(chain) < 2:
        return breaks

    # Check each bone in chain for discontinuities
    for bone_name in chain:
        if bone_name not in bone_positions:
            continue

        positions = bone_positions[bone_name]
        num_frames = len(positions)

        if num_frames < 3:
            continue

        # Compute frame-to-frame displacement (velocity)
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Compute changes in displacement (acceleration/jerk)
        # This detects DISCONTINUITIES rather than just fast movement
        displacement_changes = np.abs(np.diff(displacements))

        # Find breaks where displacement suddenly changes
        # Group consecutive high-change frames together as one break
        in_break = False

        for i, change in enumerate(displacement_changes):
            if change > max_distance and not in_break:
                # Start of a new break (sudden change in velocity)
                # The actual break occurs at frame i+1 (between frames i and i+1)
                breaks.append(
                    {
                        "frame": i + 1,
                        "bone": bone_name,
                        "displacement": displacements[i + 1],  # Displacement after the break
                        "severity": "high" if change > max_distance * 5 else "medium",
                    }
                )
                in_break = True
            elif change <= max_distance:
                # Back to smooth movement
                in_break = False

    return breaks


def validate_parent_child_consistency(
    parent_pos: np.ndarray, child_pos: np.ndarray, expected_distance: float, tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Validate parent-child bone relationship consistency.

    Args:
        parent_pos: Parent bone positions (frames, 3)
        child_pos: Child bone positions (frames, 3)
        expected_distance: Expected distance between parent and child
        tolerance: Acceptable distance variation as fraction

    Returns:
        List of consistency violations
    """
    violations = []

    # Compute distances
    vectors = child_pos - parent_pos
    distances = np.linalg.norm(vectors, axis=1)

    # Compute deviations
    deviations = np.abs(distances - expected_distance) / expected_distance

    # Find violations
    violation_mask = deviations > tolerance

    # Find continuous violation segments
    in_segment = False
    start_frame = 0

    for i, is_violation in enumerate(violation_mask):
        if is_violation and not in_segment:
            start_frame = i
            in_segment = True
        elif not is_violation and in_segment:
            # End of violation segment
            segment_deviations = deviations[start_frame:i]
            max_dev = np.max(segment_deviations) * 100

            # Classify severity
            if max_dev > 50:
                severity = "high"
            elif max_dev > 25:
                severity = "medium"
            else:
                severity = "low"

            violations.append(
                {
                    "frame_start": start_frame,
                    "frame_end": i - 1,
                    "type": "distance_violation",
                    "max_deviation_percent": max_dev,
                    "severity": severity,
                }
            )
            in_segment = False

    if in_segment:
        # Segment extends to end
        segment_deviations = deviations[start_frame:]
        max_dev = np.max(segment_deviations) * 100

        if max_dev > 50:
            severity = "high"
        elif max_dev > 25:
            severity = "medium"
        else:
            severity = "low"

        violations.append(
            {
                "frame_start": start_frame,
                "frame_end": len(violation_mask) - 1,
                "type": "distance_violation",
                "max_deviation_percent": max_dev,
                "severity": severity,
            }
        )

    return violations


def detect_curve_discontinuities(
    curve_data: np.ndarray, threshold: float = 5.0, use_derivative: bool = False
) -> List[Dict[str, Any]]:
    """
    Detect discontinuities in animation curves.

    Args:
        curve_data: Animation curve values (frames,)
        threshold: Threshold for detecting discontinuities
        use_derivative: Whether to use derivative analysis

    Returns:
        List of discontinuity events
    """
    discontinuities = []

    if len(curve_data) < 2:
        return discontinuities

    # Filter NaN values
    valid_mask = ~np.isnan(curve_data)
    if not np.any(valid_mask):
        return discontinuities

    if use_derivative:
        # Use first derivative (velocity)
        diffs = np.diff(curve_data)

        # Find large changes in derivative
        for i, diff in enumerate(diffs):
            if not np.isnan(diff) and abs(diff) > threshold:
                discontinuities.append({"frame": i + 1, "magnitude": abs(diff), "type": "derivative_spike"})
    else:
        # Direct value comparison
        diffs = np.abs(np.diff(curve_data))

        # Find discontinuities
        for i, diff in enumerate(diffs):
            if not np.isnan(diff) and diff > threshold:
                discontinuities.append({"frame": i + 1, "magnitude": diff, "type": "value_jump"})

    return discontinuities


def validate_keyframe_timing(
    keyframes: np.ndarray, expected_interval: int = 1, tolerance: int = 0
) -> List[Dict[str, Any]]:
    """
    Validate keyframe timing and spacing.

    Args:
        keyframes: Array of keyframe indices
        expected_interval: Expected spacing between keyframes
        tolerance: Acceptable deviation from expected interval

    Returns:
        List of timing violations
    """
    violations = []

    if len(keyframes) < 2:
        return violations

    # Check for duplicates
    unique_keyframes, counts = np.unique(keyframes, return_counts=True)
    duplicates = unique_keyframes[counts > 1]

    for frame in duplicates:
        violations.append({"frame": int(frame), "type": "duplicate_keyframe", "severity": "medium"})

    # Check spacing
    intervals = np.diff(keyframes)

    for i, interval in enumerate(intervals):
        deviation = abs(interval - expected_interval)

        if deviation > tolerance:
            # Check if it's a missing keyframe (double interval)
            if abs(interval - expected_interval * 2) <= tolerance:
                violations.append(
                    {"frame": int(keyframes[i] + expected_interval), "type": "missing_keyframe", "severity": "low"}
                )
            else:
                violations.append(
                    {
                        "frame": int(keyframes[i]),
                        "type": "irregular_spacing",
                        "expected_interval": expected_interval,
                        "actual_interval": int(interval),
                        "severity": "low" if deviation <= expected_interval else "medium",
                    }
                )

    return violations


def check_end_effector_reachability(
    chain_lengths: List[float], target_distance: float, check_min: bool = False
) -> bool:
    """
    Check if end effector can reach target distance.

    Args:
        chain_lengths: List of bone lengths in the chain
        target_distance: Distance to target
        check_min: Whether to check minimum reach as well

    Returns:
        True if reachable, False otherwise
    """
    if not chain_lengths:
        return False

    # Maximum reach is sum of all lengths
    max_reach = sum(chain_lengths)

    if target_distance > max_reach:
        return False

    if check_min:
        # Minimum reach is difference of longest and sum of others
        sorted_lengths = sorted(chain_lengths, reverse=True)
        if len(sorted_lengths) > 1:
            min_reach = abs(sorted_lengths[0] - sum(sorted_lengths[1:]))
        else:
            min_reach = sorted_lengths[0]

        if target_distance < min_reach:
            return False

    return True


def detect_hierarchy_violations(hierarchy: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    """
    Detect hierarchy integrity violations.

    Args:
        hierarchy: Dictionary mapping bone names to parent names (None for root)

    Returns:
        List of hierarchy violations
    """
    violations = []

    if not hierarchy:
        return violations

    # Check for circular dependencies
    def has_circular_dependency(bone, visited):
        if bone in visited:
            return True
        if bone not in hierarchy:
            return False
        if hierarchy[bone] is None:
            return False

        visited.add(bone)
        return has_circular_dependency(hierarchy[bone], visited)

    for bone in hierarchy:
        if has_circular_dependency(bone, set()):
            violations.append({"bone": bone, "type": "circular_dependency", "severity": "high"})

    # Check for orphaned bones
    all_bones = set(hierarchy.keys())
    for bone, parent in hierarchy.items():
        if parent is not None and parent not in all_bones:
            violations.append({"bone": bone, "parent": parent, "type": "orphaned_bone", "severity": "high"})

    # Check for multiple roots
    roots = [bone for bone, parent in hierarchy.items() if parent is None]
    if len(roots) > 1:
        violations.append({"roots": roots, "type": "multiple_roots", "severity": "medium"})

    return violations


def analyze_constraint_violations(scene, output_dir: str = ".") -> Dict[str, Any]:
    """
    Analyze constraint violations in FBX animation.

    Performs comprehensive constraint validation including:
    - IK chain integrity
    - Hierarchy consistency
    - Animation curve continuity
    - Keyframe timing

    Args:
        scene: FBX scene object
        output_dir: Directory for output CSV files

    Returns:
        Dictionary with analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = {
        "total_chains": 0,
        "ik_violations": 0,
        "hierarchy_violations": 0,
        "curve_discontinuities": 0,
        "overall_constraint_score": 1.0,
    }

    # For mock scene (testing), return default results
    if hasattr(scene, "GetRootNode"):
        root = scene.GetRootNode()
        if root.GetChildCount() == 0:
            # Mock scene with no bones
            _write_empty_csv_files(output_path)
            return results

    # Extract bone data
    bones = _get_all_bones(scene)

    if len(bones) == 0:
        _write_empty_csv_files(output_path)
        return results

    # Analyze IK chains
    ik_violations_list = []

    # Step 1: Extract bone animation data (positions and rotations over time)
    # This returns: {bone_name: {"positions": np.array, "rotations": np.array}}
    bone_data = _extract_bone_animation_data(scene, bones)

    # Step 2: Get hierarchy (parent-child relationships) using shared utility
    # Uses build_bone_hierarchy() from utils.py for consistency across modules
    hierarchy = build_bone_hierarchy(scene)
    hierarchy_violations_list = detect_hierarchy_violations(hierarchy)
    results["hierarchy_violations"] = len(hierarchy_violations_list)

    # Step 3: Detect IK chains from the hierarchy
    # min_chain_length=3 means we only find chains with 3+ bones (e.g., hip->thigh->calf->foot)
    chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)
    results["total_chains"] = len(chains)

    # Step 4: Validate each IK chain for integrity violations with ADAPTIVE thresholds
    for chain in chains:
        # Build a dictionary of just the bones in THIS chain
        bone_positions = {}
        for bone_name in chain:
            if bone_name in bone_data:
                bone_positions[bone_name] = bone_data[bone_name]["positions"]

        # Skip chains where we don't have enough data
        if len(bone_positions) < 2:
            continue

        # ADAPTIVE TOLERANCE: Compute chain length variance to determine natural flexibility
        # Higher variance = more flexible chain (e.g., tail, cloth) → higher tolerance
        # Lower variance = rigid chain (e.g., arm, leg) → stricter tolerance
        num_frames = len(bone_positions[chain[0]])
        chain_lengths = np.zeros(num_frames)

        # Compute total chain length per frame
        for i in range(len(chain) - 1):
            bone1, bone2 = chain[i], chain[i + 1]
            if bone1 in bone_positions and bone2 in bone_positions:
                vectors = bone_positions[bone2] - bone_positions[bone1]
                segment_lengths = np.linalg.norm(vectors, axis=1)
                chain_lengths += segment_lengths

        # Calculate coefficient of variation (CV = std / mean)
        if np.mean(chain_lengths) > 0:
            coeff_variation = np.std(chain_lengths) / np.mean(
                chain_lengths
            )  # TODO Address this warning: "Unexpected Types"

            # Adaptive tolerance based on CV:
            # Low CV (0.0-0.02) = very rigid → strict 3% tolerance
            # Medium CV (0.02-0.05) = normal IK → standard 5% tolerance
            # High CV (0.05+) = flexible → lenient 10% tolerance
            if coeff_variation < 0.02:
                adaptive_tolerance = 0.03  # Strict for rigid chains
            elif coeff_variation < 0.05:
                adaptive_tolerance = 0.05  # Standard for normal IK
            else:
                adaptive_tolerance = min(0.10, coeff_variation * 2.0)  # Lenient for flexible, cap at 10%
        else:
            adaptive_tolerance = 0.05  # Fallback to standard

        # Validate chain length with adaptive tolerance
        length_violations = validate_ik_chain_length(bone_positions, chain, tolerance=adaptive_tolerance)
        ik_violations_list.extend(length_violations)

        # ADAPTIVE MAX_DISTANCE: Compute acceleration-based threshold for chain breaks
        # Use the same MAD-based approach as curve discontinuity detection
        # Aggregate accelerations across all bones in the chain
        all_accelerations = []
        for bone_name in chain:
            if bone_name in bone_positions:
                positions = bone_positions[bone_name]
                if len(positions) >= 3:
                    # Compute velocity (displacement per frame)
                    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    # Compute acceleration (change in velocity)
                    accelerations = np.abs(np.diff(velocities))
                    all_accelerations.extend(accelerations)

        if len(all_accelerations) > 2:
            # Use MAD for robust outlier detection
            all_accelerations = np.array(all_accelerations)
            median_accel = np.median(all_accelerations)
            mad = np.median(np.abs(all_accelerations - median_accel))

            if mad > 0:
                # 3.5 modified Z-scores (robust outlier threshold)
                adaptive_max_distance = median_accel + (3.5 * 1.4826 * mad)
            else:
                # Fallback: use 3x median if MAD is 0 (constant acceleration)
                adaptive_max_distance = median_accel * 3.0
        else:
            adaptive_max_distance = 0.1  # Fallback to default

        # Detect chain breaks with adaptive threshold
        break_violations = detect_chain_breaks(bone_positions, chain, max_distance=adaptive_max_distance)
        ik_violations_list.extend(break_violations)

    # Step 5: Detect curve discontinuities using ACCELERATION analysis
    # A true discontinuity shows as a SPIKE in acceleration (sudden jerk), not just fast velocity
    # This properly distinguishes smooth rapid motion from actual discontinuities
    curve_discontinuities_list = []
    for bone_name, data in bone_data.items():
        # data is a dict: {"positions": array, "rotations": array}
        positions = data["positions"]

        # Need at least 4 frames for acceleration analysis (position -> velocity -> acceleration)
        if len(positions) < 4:
            continue

        # Check each axis (X, Y, Z) independently
        for axis_idx in range(3):
            axis_name = ["X", "Y", "Z"][axis_idx]

            # Extract single axis curve
            curve = positions[:, axis_idx]

            # STEP 1: Compute velocities (first derivative)
            velocities = np.diff(curve)

            # STEP 2: Compute accelerations (second derivative - rate of velocity change)
            # TRUE discontinuities show as acceleration spikes, not just velocity spikes
            accelerations = np.diff(velocities)

            if len(accelerations) < 2:
                continue

            # STEP 3: Compute adaptive threshold for acceleration outliers
            abs_accelerations = np.abs(accelerations)

            # Skip nearly-static bones (mean acceleration < 0.01)
            mean_accel = np.mean(abs_accelerations)
            if mean_accel < 0.01:
                continue

            std_accel = np.std(abs_accelerations)

            # Use modified Z-score (median absolute deviation) for robustness to outliers
            # This is more reliable than mean/std for detecting true anomalies
            median_accel = np.median(abs_accelerations)
            mad = np.median(np.abs(abs_accelerations - median_accel))  # Median Absolute Deviation

            if mad > 0:
                # Modified Z-score threshold: 3.5 is standard for outlier detection
                # Values beyond 3.5 modified Z-scores are considered anomalies
                threshold_mad = 3.5 * 1.4826 * mad  # 1.4826 is constant to make MAD consistent with std

                # Detect frames where acceleration exceeds threshold
                acceleration_outliers = abs_accelerations > (median_accel + threshold_mad)

                # Only flag if the outlier is ALSO significantly larger than typical accelerations
                # Must be at least 5x the median to be a true discontinuity
                for i, is_outlier in enumerate(acceleration_outliers):
                    if is_outlier and abs_accelerations[i] > (median_accel * 5.0):
                        # Found a discontinuity at frame i+2 (accounting for two derivatives)
                        # Calculate the corresponding velocity jump
                        velocity_jump = abs(velocities[i + 1])

                        curve_discontinuities_list.append(
                            {
                                "frame": i + 2,
                                "bone": bone_name,
                                "axis": axis_name,
                                "magnitude": velocity_jump,
                                "type": "acceleration_spike",
                                "adaptive_threshold": median_accel + threshold_mad,
                                "acceleration_magnitude": abs_accelerations[i],
                            }
                        )

    results["ik_violations"] = len(ik_violations_list)
    results["curve_discontinuities"] = len(curve_discontinuities_list)

    # Compute overall score
    total_violations = len(ik_violations_list) + len(hierarchy_violations_list) + len(curve_discontinuities_list)

    # Fix confidence score bug: distinguish between "no violations" and "no data"
    if results["total_chains"] == 0:
        # No IK chains detected = no data to analyze
        results["overall_constraint_score"] = 0.0
    elif total_violations == 0:
        # All checks passed = perfect score
        results["overall_constraint_score"] = 1.0
    else:
        # Penalize based on number of violations
        # Each violation reduces score by 10%, capped at 80% reduction (min score 0.2)
        # NOTE: These are heuristic thresholds (reasonable defaults, not data-driven)
        penalty = min(total_violations * 0.1, 0.8)
        results["overall_constraint_score"] = max(0.2, 1.0 - penalty)

    # Write output files
    _write_ik_violations_csv(output_path / "ik_chain_violations.csv", ik_violations_list)
    _write_hierarchy_violations_csv(output_path / "hierarchy_violations.csv", hierarchy_violations_list)
    _write_curve_discontinuities_csv(output_path / "curve_discontinuities.csv", curve_discontinuities_list)
    _write_constraint_summary_csv(output_path / "constraint_summary.csv", results)

    return results


def _get_all_bones(scene) -> List:
    """
    Extract all skeleton bones from FBX scene.

    Uses the same FBX API pattern as build_bone_hierarchy() in utils.py
    for consistency across modules.

    Args:
        scene: FBX scene object

    Returns:
        List of FBX skeleton nodes
    """
    bones = []
    root = scene.GetRootNode()

    def traverse(node):
        # Check if this node is a skeleton bone
        attr = node.GetNodeAttribute()
        if attr:
            attr_type = attr.GetAttributeType()
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                bones.append(node)

        # Recursively traverse children
        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i))

    traverse(root)
    return bones


def _extract_hierarchy(bones) -> Dict[str, Optional[str]]:
    """
    Extract parent-child hierarchy from bones.

    Uses the same pattern as build_bone_hierarchy() in utils.py
    to ensure compatibility with detect_chains_from_hierarchy().

    Args:
        bones: List of FBX skeleton nodes

    Returns:
        dict: {bone_name: parent_bone_name or None}
    """
    hierarchy = {}
    bone_names = {bone.GetName() for bone in bones}  # Set of all skeleton bones

    for bone in bones:
        bone_name = bone.GetName()
        parent = bone.GetParent()

        # Walk up parent chain until we find another skeleton bone
        parent_bone_name = None
        while parent:
            if parent.GetName() in bone_names:
                parent_bone_name = parent.GetName()
                break
            parent = parent.GetParent()

        hierarchy[bone_name] = parent_bone_name

    return hierarchy


def _write_empty_csv_files(output_path: Path):
    """Write empty CSV files."""
    # IK violations
    with open(output_path / "ik_chain_violations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain", "frame_start", "frame_end", "type", "severity"])

    # Hierarchy violations
    with open(output_path / "hierarchy_violations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bone", "type", "severity", "details"])

    # Curve discontinuities
    with open(output_path / "curve_discontinuities.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "bone", "type", "magnitude"])

    # Summary
    with open(output_path / "constraint_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])


def _write_ik_violations_csv(filepath: Path, violations: List[Dict]):
    """Write IK violations to CSV."""
    with open(filepath, "w", newline="") as f:
        if violations:
            fieldnames = ["chain", "frame_start", "frame_end", "type", "severity"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(violations)
        else:
            writer = csv.writer(f)
            writer.writerow(["chain", "frame_start", "frame_end", "type", "severity"])


def _write_hierarchy_violations_csv(filepath: Path, violations: List[Dict]):
    """Write hierarchy violations to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bone", "type", "severity", "details"])

        for v in violations:
            bone = v.get("bone", v.get("roots", ""))
            details = v.get("parent", "")
            writer.writerow([bone, v["type"], v["severity"], details])


def _write_curve_discontinuities_csv(filepath: Path, discontinuities: List[Dict]):
    """Write curve discontinuities to CSV."""
    with open(filepath, "w", newline="") as f:
        if discontinuities:
            fieldnames = ["frame", "bone", "axis", "type", "magnitude", "adaptive_threshold", "acceleration_magnitude"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(discontinuities)
        else:
            writer = csv.writer(f)
            writer.writerow(
                ["frame", "bone", "axis", "type", "magnitude", "adaptive_threshold", "acceleration_magnitude"]
            )


def _write_constraint_summary_csv(filepath: Path, results: Dict):
    """Write constraint summary to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in results.items():
            writer.writerow([key, value])
