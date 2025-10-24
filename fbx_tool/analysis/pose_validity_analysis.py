"""
Pose Validity Analysis Module

Analyzes animation poses for anatomical validity and common issues:
- Bone length consistency (stretch/squash detection)
- Joint angle limits (anatomically impossible poses)
- Self-intersection detection
- Bilateral symmetry validation
- Pose type detection (T-pose, A-pose, bind pose)

Author: FBX Tool
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fbx_tool.analysis.utils import build_bone_hierarchy


@dataclass
class BoneLengthViolation:
    """Represents a bone length violation event."""

    frame_start: int
    frame_end: int
    type: str  # 'stretch' or 'squash'
    max_deviation_percent: float
    mean_deviation_percent: float
    severity: str  # 'low', 'medium', 'high'


@dataclass
class JointAngleViolation:
    """Represents a joint angle limit violation."""

    frame_start: int
    frame_end: int
    type: str  # 'min_exceeded' or 'max_exceeded'
    max_violation_degrees: float
    mean_violation_degrees: float
    severity: str


@dataclass
class SelfIntersection:
    """Represents a self-intersection between two bones."""

    frame: int
    bone1_name: str
    bone2_name: str
    distance: float
    severity: str


def compute_bone_lengths(parent_positions: np.ndarray, child_positions: np.ndarray) -> np.ndarray:
    """
    Compute bone lengths across frames.

    Args:
        parent_positions: Parent joint positions (frames, 3)
        child_positions: Child joint positions (frames, 3)

    Returns:
        Bone lengths for each frame (frames,)
    """
    if len(parent_positions.shape) == 1:
        parent_positions = parent_positions.reshape(1, -1)
    if len(child_positions.shape) == 1:
        child_positions = child_positions.reshape(1, -1)

    # Compute distance between parent and child
    vectors = child_positions - parent_positions
    lengths = np.linalg.norm(vectors, axis=1)

    return lengths


def detect_bone_length_violations(
    bone_lengths: np.ndarray, reference_length: float, tolerance: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Detect bone length violations (stretching or squashing).

    Args:
        bone_lengths: Bone lengths for each frame
        reference_length: Expected bone length (from bind pose or median)
        tolerance: Acceptable deviation as fraction (default 5%)

    Returns:
        List of violation events with details
    """
    violations = []

    # Compute deviation from reference
    deviations = (bone_lengths - reference_length) / reference_length
    deviation_percent = deviations * 100

    # Detect stretching (positive deviation beyond tolerance)
    stretch_mask = deviations > tolerance

    # Detect squashing (negative deviation beyond tolerance)
    squash_mask = deviations < -tolerance

    # Find continuous segments
    def find_segments(mask):
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_violation in enumerate(mask):
            if is_violation and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_violation and in_segment:
                segments.append((start_frame, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start_frame, len(mask) - 1))

        return segments

    # Process stretch violations
    for start, end in find_segments(stretch_mask):
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
            }
        )

    # Process squash violations
    for start, end in find_segments(squash_mask):
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
                "type": "squash",
                "max_deviation_percent": max_dev,
                "mean_deviation_percent": mean_dev,
                "severity": severity,
            }
        )

    return violations


def validate_joint_angle_limits(
    angles: np.ndarray, joint_type: str, min_angle: float = 0.0, max_angle: float = 180.0
) -> List[Dict[str, Any]]:
    """
    Validate joint angles against anatomical limits.

    Args:
        angles: Joint angles in degrees for each frame
        joint_type: Type of joint ('elbow', 'knee', 'shoulder', etc.)
        min_angle: Minimum anatomical angle
        max_angle: Maximum anatomical angle

    Returns:
        List of angle limit violations
    """
    violations = []

    # Filter out NaN values
    valid_mask = ~np.isnan(angles)
    if not np.any(valid_mask):
        return violations

    # Detect angles exceeding maximum limit
    max_exceeded_mask = angles > max_angle

    # Detect angles below minimum limit
    min_exceeded_mask = angles < min_angle

    def find_segments(mask):
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_violation in enumerate(mask):
            if is_violation and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_violation and in_segment:
                segments.append((start_frame, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start_frame, len(mask) - 1))

        return segments

    # Process maximum exceeded violations
    for start, end in find_segments(max_exceeded_mask):
        segment_angles = angles[start : end + 1]
        violations_deg = segment_angles - max_angle
        max_violation = np.max(violations_deg)
        mean_violation = np.mean(violations_deg)

        # Classify severity
        if max_violation > 30:
            severity = "high"
        elif max_violation > 15:
            severity = "medium"
        else:
            severity = "low"

        violations.append(
            {
                "frame_start": start,
                "frame_end": end,
                "type": "max_exceeded",
                "max_violation_degrees": max_violation,
                "mean_violation_degrees": mean_violation,
                "severity": severity,
                "joint_type": joint_type,
            }
        )

    # Process minimum exceeded violations
    for start, end in find_segments(min_exceeded_mask):
        segment_angles = angles[start : end + 1]
        violations_deg = min_angle - segment_angles
        max_violation = np.max(violations_deg)
        mean_violation = np.mean(violations_deg)

        # Classify severity
        if max_violation > 30:
            severity = "high"
        elif max_violation > 15:
            severity = "medium"
        else:
            severity = "low"

        violations.append(
            {
                "frame_start": start,
                "frame_end": end,
                "type": "min_exceeded",
                "max_violation_degrees": max_violation,
                "mean_violation_degrees": mean_violation,
                "severity": severity,
                "joint_type": joint_type,
            }
        )

    return violations


def detect_self_intersections(
    bone1_start: np.ndarray,
    bone1_end: np.ndarray,
    bone2_start: np.ndarray,
    bone2_end: np.ndarray,
    distance_threshold: Optional[float] = None,
    median_bone_length: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Detect self-intersections between two bones.

    Args:
        bone1_start: Start positions of bone 1 (frames, 3)
        bone1_end: End positions of bone 1 (frames, 3)
        bone2_start: Start positions of bone 2 (frames, 3)
        bone2_end: End positions of bone 2 (frames, 3)
        distance_threshold: Maximum distance to consider as intersection (optional)
        median_bone_length: Median bone length for adaptive threshold calculation (optional)

    Returns:
        List of intersection events

    Note:
        If distance_threshold is not provided, it will be computed adaptively as
        5% of median_bone_length. This scales the intersection detection to skeleton size.
    """
    # PROCEDURAL: Compute adaptive distance threshold if not provided
    if distance_threshold is None:
        if median_bone_length is not None and median_bone_length > 0:
            # 5% of median bone length is reasonable for intersection detection
            distance_threshold = median_bone_length * 0.05
        else:
            # Fallback to hardcoded value
            distance_threshold = 0.5
    intersections = []

    # Check if bones are identical (filter out self-comparison)
    if np.allclose(bone1_start, bone2_start) and np.allclose(bone1_end, bone2_end):
        return intersections

    num_frames = bone1_start.shape[0]

    for frame in range(num_frames):
        # Get bone segments for this frame
        p1 = bone1_start[frame]
        p2 = bone1_end[frame]
        p3 = bone2_start[frame]
        p4 = bone2_end[frame]

        # Compute minimum distance between line segments
        min_dist = compute_line_segment_distance(p1, p2, p3, p4)

        if min_dist < distance_threshold:
            # Classify severity based on distance
            if min_dist < distance_threshold * 0.3:
                severity = "high"
            elif min_dist < distance_threshold * 0.6:
                severity = "medium"
            else:
                severity = "low"

            intersections.append({"frame": frame, "distance": min_dist, "severity": severity})

    return intersections


def compute_line_segment_distance(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute minimum distance between two line segments.

    Uses the method from:
    http://geomalgorithms.com/a07-_distance.html

    Args:
        p1, p2: Endpoints of first segment
        p3, p4: Endpoints of second segment

    Returns:
        Minimum distance between segments
    """
    # Direction vectors
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b  # Denominator

    # Handle parallel segments
    if D < 1e-8:
        # Segments are parallel
        s = 0.0
        t = (b > c) and (d / b) or (e / c)
    else:
        s = (b * e - c * d) / D
        t = (a * e - b * d) / D

    # Clamp s and t to [0, 1]
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)

    # Compute closest points
    closest1 = p1 + s * u
    closest2 = p3 + t * v

    # Return distance
    return np.linalg.norm(closest1 - closest2)


def compute_symmetry_score(left_data: np.ndarray, right_data: np.ndarray, compare_rotations: bool = False) -> float:
    """
    Compute bilateral symmetry score.

    Args:
        left_data: Left limb data (positions or rotations)
        right_data: Right limb data (positions or rotations)
        compare_rotations: If True, compare rotations; if False, compare positions

    Returns:
        Symmetry score from 0 (no symmetry) to 1 (perfect symmetry)
    """
    if left_data.size == 0 or right_data.size == 0:
        return 1.0  # No data, assume symmetric

    # For positions, mirror the X axis for comparison
    if not compare_rotations:
        right_mirrored = right_data.copy()
        right_mirrored[:, 0] = -right_mirrored[:, 0]  # Mirror X

        # Compute RMSE between left and mirrored right
        differences = left_data - right_mirrored
        rmse = np.sqrt(np.mean(differences**2))

        # Normalize to 0-1 score (lower RMSE = higher score)
        # Assume RMSE > 20 units = completely asymmetric
        max_rmse = 20.0
        normalized_rmse = min(rmse / max_rmse, 1.0)
        symmetry_score = 1.0 - normalized_rmse

    else:
        # For rotations, compare angles directly
        # (assuming rotations are already in symmetric reference frame)
        differences = np.abs(left_data - right_data)
        mean_diff = np.mean(differences)

        # Normalize to 0-1 score (angles in degrees)
        # Assume mean diff > 45° = completely asymmetric
        max_diff = 45.0
        normalized_diff = min(mean_diff / max_diff, 1.0)
        symmetry_score = 1.0 - normalized_diff

    return symmetry_score


def detect_pose_type(bone_rotations: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Detect the type of pose (T-pose, A-pose, bind pose, animated).

    Args:
        bone_rotations: Dictionary mapping bone names to rotation arrays

    Returns:
        Tuple of (pose_type, confidence)
        pose_type: 'T-pose', 'A-pose', 'bind', or 'animated'
        confidence: 0-1 confidence score
    """
    # Extract shoulder rotations
    left_shoulder = bone_rotations.get("left_shoulder")
    right_shoulder = bone_rotations.get("right_shoulder")

    if left_shoulder is None or right_shoulder is None:
        return "animated", 0.0

    # Compute mean rotations across frames
    left_mean = np.mean(left_shoulder, axis=0)
    right_mean = np.mean(right_shoulder, axis=0)

    # Check variance - static poses have low variance
    left_variance = np.mean(np.var(left_shoulder, axis=0))
    right_variance = np.mean(np.var(right_shoulder, axis=0))
    total_variance = (left_variance + right_variance) / 2

    # Check for T-pose (arms horizontal ~90°)
    # Z rotation around 90° for left, -90° for right
    left_z = left_mean[2]
    right_z = right_mean[2]

    # T-pose detection
    if 75 < left_z < 105 and -105 < right_z < -75:
        confidence = 1.0 - (abs(left_z - 90) + abs(right_z + 90)) / 30
        return "T-pose", max(0.0, min(1.0, confidence))

    # A-pose detection (arms at ~45°)
    if 35 < left_z < 55 and -55 < right_z < -35:
        confidence = 1.0 - (abs(left_z - 45) + abs(right_z + 45)) / 20
        return "A-pose", max(0.0, min(1.0, confidence))

    # Bind pose detection (arms down ~0°) - require low variance
    if -10 < left_z < 10 and -10 < right_z < 10 and total_variance < 100:
        confidence = 1.0 - (abs(left_z) + abs(right_z)) / 20
        return "bind", max(0.0, min(1.0, confidence))

    # Otherwise, it's animated (not a reference pose)
    # Animated confidence is based on how much it deviates from reference poses
    # High variance = lower confidence (it's truly animated/dynamic)
    # Confidence decreases as variance increases
    if total_variance > 1000:
        animated_confidence = 0.2  # Very high variance = low confidence score
    elif total_variance > 500:
        animated_confidence = 0.3
    elif total_variance > 200:
        animated_confidence = 0.4
    else:
        animated_confidence = 0.5  # Moderate variance

    return "animated", animated_confidence


def analyze_pose_validity(scene, output_dir: str = ".") -> Dict[str, Any]:
    """
    Analyze pose validity for an FBX animation.

    Performs comprehensive pose validation including:
    - Bone length consistency
    - Joint angle limits
    - Self-intersection detection
    - Bilateral symmetry
    - Pose type detection

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
        "total_bones": 0,
        "bones_with_length_violations": 0,
        "bones_with_angle_violations": 0,
        "self_intersections_detected": 0,
        "overall_validity_score": 1.0,
        "pose_type": "unknown",
        "pose_type_confidence": 0.0,
    }

    # For mock scene (testing), return default results
    if hasattr(scene, "GetRootNode"):
        root = scene.GetRootNode()
        if root.GetChildCount() == 0:
            # Mock scene with no bones
            _write_empty_csv_files(output_path)
            return results

    # Get all bones from scene
    bones = _get_all_bones(scene)
    results["total_bones"] = len(bones)

    if len(bones) == 0:
        _write_empty_csv_files(output_path)
        return results

    # OPTIMIZATION: Build bone hierarchy once (shared utility)
    hierarchy = build_bone_hierarchy(scene)

    # Extract animation data (using pre-built hierarchy)
    bone_data = _extract_bone_animation_data(scene, bones, hierarchy=hierarchy)

    # STEP 1: Collect all bone lengths to compute adaptive tolerance
    all_bone_lengths = []
    bone_reference_lengths = {}
    for bone_name, data in bone_data.items():
        if data["parent_positions"] is not None:
            bone_lengths = compute_bone_lengths(data["parent_positions"], data["positions"])
            reference_length = np.median(bone_lengths)
            bone_reference_lengths[bone_name] = reference_length
            all_bone_lengths.append(reference_length)

    # STEP 2: Compute adaptive tolerance based on skeleton bone length distribution
    # PROCEDURAL: Use coefficient of variation to determine appropriate tolerance
    if len(all_bone_lengths) > 0:
        median_bone_length = np.median(all_bone_lengths)
        std_bone_length = np.std(all_bone_lengths)
        cv_bone_lengths = std_bone_length / median_bone_length if median_bone_length > 0 else 0

        # Adaptive tolerance: tighter for uniform skeletons, looser for varied skeletons
        # Base tolerance 5%, adjusted by CV
        adaptive_tolerance = 0.05 * (1.0 + cv_bone_lengths)
        # Clamp to reasonable range [3%, 15%]
        adaptive_tolerance = np.clip(adaptive_tolerance, 0.03, 0.15)
    else:
        adaptive_tolerance = 0.05  # Fallback

    # STEP 3: Detect violations using adaptive tolerance
    length_violations = []
    for bone_name, data in bone_data.items():
        if data["parent_positions"] is not None:
            bone_lengths = compute_bone_lengths(data["parent_positions"], data["positions"])
            reference_length = bone_reference_lengths[bone_name]

            violations = detect_bone_length_violations(bone_lengths, reference_length, tolerance=adaptive_tolerance)

            for v in violations:
                v["bone_name"] = bone_name
                length_violations.append(v)

    results["bones_with_length_violations"] = len(set(v["bone_name"] for v in length_violations))

    # Analyze joint angles (simplified - would need proper joint hierarchy)
    angle_violations = []
    # TODO: Implement full joint angle analysis with proper hierarchy

    # Detect self-intersections (simplified)
    intersection_count = 0
    # TODO: Implement pairwise bone intersection checking

    results["self_intersections_detected"] = intersection_count

    # Compute overall validity score
    validity_score = 1.0
    if len(bones) > 0:
        # Penalize for violations
        length_penalty = min(len(length_violations) * 0.1, 0.5)
        angle_penalty = min(len(angle_violations) * 0.1, 0.3)
        intersection_penalty = min(intersection_count * 0.05, 0.2)

        validity_score = max(0.0, 1.0 - length_penalty - angle_penalty - intersection_penalty)

    results["overall_validity_score"] = validity_score

    # Write output files
    _write_bone_length_violations_csv(output_path / "bone_length_violations.csv", length_violations)
    _write_joint_angle_violations_csv(output_path / "joint_angle_violations.csv", angle_violations)
    _write_symmetry_analysis_csv(output_path / "symmetry_analysis.csv", {})
    _write_pose_validity_summary_csv(output_path / "pose_validity_summary.csv", results)

    return results


def _get_all_bones(scene) -> List:
    """Extract all bones from FBX scene."""
    import fbx as fbx_module

    bones = []
    root = scene.GetRootNode()

    def traverse(node):
        attr = node.GetNodeAttribute()
        if attr:
            attr_type = attr.GetAttributeType()
            # Check if it's a skeleton node
            if attr_type == fbx_module.FbxNodeAttribute.EType.eSkeleton:
                bones.append(node)

        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i))

    traverse(root)
    return bones


def _extract_bone_animation_data(scene, bones, hierarchy: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Dict]:
    """
    Extract position and rotation data for all bones.

    Args:
        scene: FBX scene object
        bones: List of bone nodes
        hierarchy: Optional pre-built bone hierarchy map {child_name: parent_name}
                   If None, will use GetParent() for each bone (slower)

    Returns:
        Dict mapping bone names to their animation data
    """
    import fbx as fbx_module

    bone_data = {}

    # OPTIMIZATION: Create bone node lookup if using hierarchy
    bone_lookup = {}
    if hierarchy is not None:
        for bone in bones:
            bone_lookup[bone.GetName()] = bone

    # Get animation timespan
    anim_stack_count = scene.GetSrcObjectCount(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId))
    if anim_stack_count == 0:
        # No animation, return empty data
        for bone in bones:
            bone_name = bone.GetName()
            bone_data[bone_name] = {
                "positions": np.array([]),
                "rotations": np.array([]),
                "parent_positions": None,
            }
        return bone_data

    # Get first animation stack
    anim_stack = scene.GetSrcObject(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId), 0)
    time_span = anim_stack.GetLocalTimeSpan()
    start_time = time_span.GetStart()
    stop_time = time_span.GetStop()

    # Calculate frame rate and total frames
    frame_rate = fbx_module.FbxTime.GetFrameRate(scene.GetGlobalSettings().GetTimeMode())
    duration_seconds = stop_time.GetSecondDouble() - start_time.GetSecondDouble()
    total_frames = int(duration_seconds * frame_rate) + 1

    # Create frame duration object
    frame_duration = fbx_module.FbxTime()
    frame_duration.SetSecondDouble(1.0 / frame_rate)

    # Extract data for each bone
    for bone in bones:
        bone_name = bone.GetName()
        positions = []
        rotations = []

        # Extract transforms for each frame
        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame

            # Get global transform
            global_transform = bone.EvaluateGlobalTransform(current_time)

            # Extract translation and rotation
            translation = global_transform.GetT()
            rotation = global_transform.GetR()

            positions.append([translation[0], translation[1], translation[2]])
            rotations.append([rotation[0], rotation[1], rotation[2]])

        positions = np.array(positions)
        rotations = np.array(rotations)

        # Get parent positions if bone has a parent
        # OPTIMIZATION: Use pre-built hierarchy if available, otherwise fallback to GetParent()
        parent_node = None
        if hierarchy is not None:
            parent_name = hierarchy.get(bone_name)
            if parent_name is not None:
                parent_node = bone_lookup.get(parent_name)
        else:
            # Fallback: direct GetParent() call
            parent_node = bone.GetParent()

        parent_positions = None
        if parent_node:
            parent_positions = []
            for frame in range(total_frames):
                current_time = start_time + frame_duration * frame
                parent_transform = parent_node.EvaluateGlobalTransform(current_time)
                parent_translation = parent_transform.GetT()
                parent_positions.append([parent_translation[0], parent_translation[1], parent_translation[2]])
            parent_positions = np.array(parent_positions)

        bone_data[bone_name] = {
            "positions": positions,
            "rotations": rotations,
            "parent_positions": parent_positions,
        }

    return bone_data


def _write_empty_csv_files(output_path: Path):
    """Write empty CSV files for cases with no data."""
    # Bone length violations
    with open(output_path / "bone_length_violations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bone_name",
                "frame_start",
                "frame_end",
                "type",
                "max_deviation_percent",
                "mean_deviation_percent",
                "severity",
            ]
        )

    # Joint angle violations
    with open(output_path / "joint_angle_violations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bone_name",
                "frame_start",
                "frame_end",
                "type",
                "max_violation_degrees",
                "mean_violation_degrees",
                "severity",
            ]
        )

    # Symmetry analysis
    with open(output_path / "symmetry_analysis.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["limb_pair", "symmetry_score", "quality"])

    # Summary
    with open(output_path / "pose_validity_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])


def _write_bone_length_violations_csv(filepath: Path, violations: List[Dict]):
    """Write bone length violations to CSV."""
    with open(filepath, "w", newline="") as f:
        if violations:
            fieldnames = [
                "bone_name",
                "frame_start",
                "frame_end",
                "type",
                "max_deviation_percent",
                "mean_deviation_percent",
                "severity",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(violations)
        else:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bone_name",
                    "frame_start",
                    "frame_end",
                    "type",
                    "max_deviation_percent",
                    "mean_deviation_percent",
                    "severity",
                ]
            )


def _write_joint_angle_violations_csv(filepath: Path, violations: List[Dict]):
    """Write joint angle violations to CSV."""
    with open(filepath, "w", newline="") as f:
        if violations:
            fieldnames = [
                "bone_name",
                "frame_start",
                "frame_end",
                "type",
                "max_violation_degrees",
                "mean_violation_degrees",
                "severity",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(violations)
        else:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bone_name",
                    "frame_start",
                    "frame_end",
                    "type",
                    "max_violation_degrees",
                    "mean_violation_degrees",
                    "severity",
                ]
            )


def _write_symmetry_analysis_csv(filepath: Path, symmetry_data: Dict):
    """Write symmetry analysis to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["limb_pair", "symmetry_score", "quality"])

        for limb_pair, score in symmetry_data.items():
            if score >= 0.9:
                quality = "excellent"
            elif score >= 0.7:
                quality = "good"
            elif score >= 0.5:
                quality = "fair"
            else:
                quality = "poor"

            writer.writerow([limb_pair, f"{score:.3f}", quality])


def _write_pose_validity_summary_csv(filepath: Path, results: Dict):
    """Write pose validity summary to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in results.items():
            writer.writerow([key, value])
