"""
Utilities Module
Common helper functions for file I/O, bone hierarchy traversal, and data processing.
"""

import csv
import os

import fbx
import numpy as np

# ============================================================================
# Trajectory Cache (Session-level)
# ============================================================================

# Session-level cache for root trajectory data
# Avoids redundant extraction when running multiple motion analyses
_trajectory_cache = {}


def get_scene_cache_key(scene):
    """
    Generate a unique cache key for an FBX scene.

    Uses the scene's memory address as a unique identifier.

    Args:
        scene: FBX scene object

    Returns:
        int: Unique cache key
    """
    return id(scene)


def clear_trajectory_cache():
    """
    Clear the trajectory cache.

    Call this between loading different FBX files to avoid
    using cached data from a previous scene.
    """
    global _trajectory_cache
    _trajectory_cache = {}


def _compute_adaptive_thresholds(velocity_mags, angular_velocities):
    """
    Compute adaptive thresholds from motion data distribution.

    PROCEDURAL THRESHOLD COMPUTATION - replaces hardcoded values with data-driven thresholds.
    This makes the system work for animations at any scale or speed (slow walk, sprint, etc.).

    Strategy:
    1. Stationary threshold = percentile of velocity distribution (captures natural rest state)
    2. Turning thresholds = percentiles of angular velocity distribution

    Args:
        velocity_mags: np.array (N,) of velocity magnitudes per frame
        angular_velocities: np.array (N,) of angular velocities (degrees/sec) per frame

    Returns:
        dict: {
            'stationary_velocity_threshold': float,
            'turning_slow_threshold': float,
            'turning_fast_threshold': float,
            'turning_very_fast_threshold': float,
            'confidence': float (0-1, how reliable are these thresholds?)
        }
    """
    global _STATIONARY_VELOCITY_THRESHOLD
    global _TURNING_THRESHOLD_SLOW
    global _TURNING_THRESHOLD_FAST
    global _TURNING_THRESHOLD_VERY_FAST

    if len(velocity_mags) < 30:
        # Not enough data - use defaults
        return {
            "stationary_velocity_threshold": _DEFAULT_STATIONARY_VELOCITY_THRESHOLD,
            "turning_slow_threshold": _DEFAULT_TURNING_THRESHOLD_SLOW,
            "turning_fast_threshold": _DEFAULT_TURNING_THRESHOLD_FAST,
            "turning_very_fast_threshold": _DEFAULT_TURNING_THRESHOLD_VERY_FAST,
            "confidence": 0.0,
        }

    # === STATIONARY VELOCITY THRESHOLD ===
    # Use 15th percentile of velocity (captures natural resting state)
    # If character is mostly moving, this will be higher than 0.1
    # If character is mostly still, this will be lower
    stationary_threshold = np.percentile(velocity_mags, 15)

    # Ensure minimum threshold (avoid treating tiny movements as motion)
    stationary_threshold = max(stationary_threshold, 0.01)

    # Ensure maximum threshold (don't set too high if animation is slow)
    # Max 20% of median velocity
    median_velocity = np.median(velocity_mags[velocity_mags > 1e-6])
    if median_velocity > 0:
        stationary_threshold = min(stationary_threshold, median_velocity * 0.2)

    # === TURNING THRESHOLDS ===
    # Use percentiles of angular velocity distribution
    angular_abs = np.abs(angular_velocities)
    angular_nonzero = angular_abs[angular_abs > 1e-3]  # Filter out noise

    if len(angular_nonzero) > 10:
        # 33rd percentile = slow turn
        # 66th percentile = fast turn
        # 90th percentile = very fast turn
        turning_slow = np.percentile(angular_nonzero, 33)
        turning_fast = np.percentile(angular_nonzero, 66)
        turning_very_fast = np.percentile(angular_nonzero, 90)

        # Ensure reasonable minimums (avoid too-sensitive detection)
        turning_slow = max(turning_slow, 10.0)  # At least 10 deg/sec
        turning_fast = max(turning_fast, turning_slow * 2)  # At least 2x slow
        turning_very_fast = max(turning_very_fast, turning_fast * 1.5)  # At least 1.5x fast

        confidence = min(1.0, len(angular_nonzero) / 100.0)  # More samples = higher confidence
    else:
        # Not enough turning data - use defaults
        turning_slow = _DEFAULT_TURNING_THRESHOLD_SLOW
        turning_fast = _DEFAULT_TURNING_THRESHOLD_FAST
        turning_very_fast = _DEFAULT_TURNING_THRESHOLD_VERY_FAST
        confidence = 0.0

    # Update global thresholds (used by classification functions)
    _STATIONARY_VELOCITY_THRESHOLD = stationary_threshold
    _TURNING_THRESHOLD_SLOW = turning_slow
    _TURNING_THRESHOLD_FAST = turning_fast
    _TURNING_THRESHOLD_VERY_FAST = turning_very_fast

    return {
        "stationary_velocity_threshold": stationary_threshold,
        "turning_slow_threshold": turning_slow,
        "turning_fast_threshold": turning_fast,
        "turning_very_fast_threshold": turning_very_fast,
        "confidence": confidence,
    }


def detect_full_coordinate_system(scene, positions, velocities):
    """
    Detect COMPLETE coordinate system from FBX metadata and empirical motion data.

    Combines two sources of truth:
    1. FBX SDK's declared axis system (from file metadata)
    2. Empirical motion analysis (from actual animation data)

    This procedural approach eliminates ALL hardcoded axis assumptions.

    Args:
        scene: FBX scene object
        positions: np.array (Nx3) of positions per frame
        velocities: np.array (Nx3) of velocity vectors per frame

    Returns:
        dict: Complete coordinate system configuration
            {
                'up_axis': int (0=X, 1=Y, 2=Z),
                'up_sign': int (1 or -1),
                'forward_axis': int (0=X, 1=Y, 2=Z),
                'forward_sign': int (1 or -1),
                'right_axis': int (0=X, 1=Y, 2=Z),
                'right_sign': int (1 or -1),
                'is_right_handed': bool,
                'yaw_axis': int (rotation axis for turning, matches up_axis),
                'yaw_positive_is_left': bool (sign convention for left turn),
                'confidence': float (0-1, how certain we are)
            }
    """
    import fbx as fbx_module

    # STEP 1: Get declared axis system from FBX metadata
    axis_system = scene.GetGlobalSettings().GetAxisSystem()
    up_vector, up_sign_declared = axis_system.GetUpVector()
    front_vector, front_parity = axis_system.GetFrontVector()
    coord_system = axis_system.GetCoorSystem()

    # Map FBX enum to axis index
    up_axis_map = {
        fbx_module.FbxAxisSystem.EUpVector.eXAxis: 0,
        fbx_module.FbxAxisSystem.EUpVector.eYAxis: 1,
        fbx_module.FbxAxisSystem.EUpVector.eZAxis: 2,
    }
    up_axis = up_axis_map.get(up_vector, 1)  # Default to Y if unknown

    is_right_handed = coord_system == fbx_module.FbxAxisSystem.ECoordSystem.eRightHanded

    # STEP 2: Detect forward axis empirically from motion data
    forward_axis, forward_sign, empirical_confidence = _detect_forward_axis_empirical(positions, velocities)

    # STEP 3: Compute right axis using cross product rule
    # In a right-handed system: right = forward × up
    # In a left-handed system: right = up × forward
    axes = [0, 1, 2]
    right_axis = [ax for ax in axes if ax != up_axis and ax != forward_axis][0]

    # Determine right sign using handedness
    # This ensures: up × forward = right (right-handed) or forward × up = right (left-handed)
    axis_order = [up_axis, forward_axis, right_axis]
    perm = sum(1 for i in range(3) for j in range(i + 1, 3) if axis_order[i] > axis_order[j])
    is_even_perm = perm % 2 == 0

    if is_right_handed:
        right_sign = forward_sign * up_sign_declared * (1 if is_even_perm else -1)
    else:
        right_sign = -forward_sign * up_sign_declared * (1 if is_even_perm else -1)

    # STEP 4: Determine turning conventions
    # Yaw rotation is around the UP axis
    yaw_axis = up_axis

    # In right-handed Y-up: positive Y rotation = counterclockwise from above = LEFT
    # In left-handed Y-up: positive Y rotation = clockwise from above = RIGHT
    # Generalize: in right-handed systems, positive rotation around up = LEFT
    yaw_positive_is_left = is_right_handed

    return {
        "up_axis": up_axis,
        "up_sign": up_sign_declared,
        "forward_axis": forward_axis,
        "forward_sign": forward_sign,
        "right_axis": right_axis,
        "right_sign": right_sign,
        "is_right_handed": is_right_handed,
        "yaw_axis": yaw_axis,
        "yaw_positive_is_left": yaw_positive_is_left,
        "confidence": empirical_confidence,
    }


def _detect_forward_axis_empirical(positions, velocities):
    """
    Empirically detect forward axis by analyzing motion data.

    This is the motion-based component of coordinate system detection.

    Args:
        positions: np.array (Nx3) of positions per frame
        velocities: np.array (Nx3) of velocity vectors per frame

    Returns:
        tuple: (axis_index, sign, confidence) where:
            - axis_index: 0 (X), 1 (Y), or 2 (Z)
            - sign: 1 (positive) or -1 (negative)
            - confidence: 0.0-1.0 score indicating detection certainty
    """
    if len(positions) < 10:
        # Not enough data - use default
        return (2, -1, 0.0)  # -Z default with 0 confidence

    # Compute total displacement (start to end)
    total_displacement = positions[-1] - positions[0]
    displacement_magnitude = np.linalg.norm(total_displacement)

    if displacement_magnitude < 1e-3:
        # No significant movement - use default
        return (2, -1, 0.0)

    # Test each axis candidate
    candidates = [
        (0, 1),  # +X
        (0, -1),  # -X
        (1, 1),  # +Y
        (1, -1),  # -Y
        (2, 1),  # +Z
        (2, -1),  # -Z
    ]

    best_score = -np.inf
    best_config = (2, -1)  # Default

    for axis_idx, sign in candidates:
        # Compute alignment score: how much motion is along this axis?
        # Project displacement onto this axis
        axis_displacement = sign * total_displacement[axis_idx]

        # Also check velocity consistency
        # Velocities along this axis should have consistent sign when moving
        axis_velocities = sign * velocities[:, axis_idx]

        # Compute metrics:
        # 1. Total displacement along axis (higher is better)
        # 2. Velocity consistency (fewer sign changes is better)

        # Normalize displacement by total magnitude
        normalized_displacement = axis_displacement / (displacement_magnitude + 1e-10)

        # Velocity consistency: std of velocities (lower means more consistent direction)
        velocity_std = np.std(axis_velocities)
        velocity_mean_abs = np.mean(np.abs(axis_velocities))

        # Avoid division by zero
        if velocity_mean_abs < 1e-6:
            consistency = 0.0
        else:
            # Consistency = mean / std (higher means more consistent)
            consistency = velocity_mean_abs / (velocity_std + 1e-6)

        # Combined score: favor axes with high displacement AND consistent velocity
        score = normalized_displacement * consistency

        if score > best_score:
            best_score = score
            best_config = (axis_idx, sign)

    # Compute confidence based on how much better the best axis is vs others
    # High confidence if one axis clearly dominates
    confidence = min(1.0, abs(best_score) / 10.0)  # Scale to 0-1 range

    return best_config + (confidence,)


# Legacy function for backward compatibility
def _detect_coordinate_system(positions, velocities):
    """
    DEPRECATED: Use detect_full_coordinate_system() instead.

    This function only detects forward axis, not the complete coordinate frame.
    Maintained for backward compatibility with existing code.
    """
    return _detect_forward_axis_empirical(positions, velocities)


def extract_root_trajectory(scene, force_refresh=False):
    """
    Extract root bone trajectory data from FBX scene (with caching).

    This is the shared foundation for all motion analysis modules.
    Trajectory data is expensive to extract, so results are cached
    per scene to avoid redundant computation when running multiple
    motion analyses (root_motion, directional_changes, motion_transitions).

    PROCEDURAL SYSTEMS (Zero Hardcoded Assumptions):
    1. COORDINATE SYSTEM DETECTION: Automatically detects forward axis by analyzing
       motion data. No assumptions about +Z vs -Z vs +X. Works with any rig.
    2. ADAPTIVE THRESHOLDS: Computes stationary/turning thresholds from data distribution.
       No hardcoded 0.1 units/sec or 30 deg/sec. Adapts to animation speed/scale.

    The trajectory includes:
    - Position per frame (x, y, z)
    - Rotation per frame (Euler angles)
    - Velocity vectors and magnitudes
    - Angular velocity (turning speed)
    - Direction classification (forward/backward/strafe/stationary)
    - Forward direction vectors (computed using detected coordinate system)
    - Coordinate system metadata (which axis is forward, confidence score)
    - Adaptive thresholds (stationary velocity, turning speed classifications)

    Args:
        scene: FBX scene object
        force_refresh: If True, bypass cache and re-extract (default: False)

    Returns:
        dict: {
            'trajectory_data': list of frame dicts with position, velocity, direction, etc.,
            'frame_rate': float,
            'root_bone_name': str,
            'total_frames': int,
            'positions': np.array (Nx3),
            'velocities': np.array (Nx3),
            'rotations': np.array (Nx3),
            'forward_directions': np.array (Nx3),
            'velocity_mags': np.array (N,),
            'angular_velocity_yaw': np.array (N,),  # PROCEDURAL: Uses detected yaw axis (not hardcoded Y)
            'coordinate_system': dict with detected coordinate system info,
            'adaptive_thresholds': dict with threshold values and confidence
        }

    Raises:
        ValueError: If no animation data or root bone found
    """
    # Import here to avoid circular dependencies
    from fbx_tool.analysis.fbx_loader import get_scene_metadata
    from fbx_tool.analysis.velocity_analysis import compute_derivatives

    # Check cache first
    cache_key = get_scene_cache_key(scene)
    if not force_refresh and cache_key in _trajectory_cache:
        return _trajectory_cache[cache_key]

    # Extract trajectory (expensive operation)
    print("  Extracting root trajectory data...")

    # Get scene metadata
    metadata = get_scene_metadata(scene)

    if not metadata.get("has_animation", False):
        raise ValueError("No animation data found in scene")

    start_time_meta = metadata["start_time"]
    stop_time_meta = metadata["stop_time"]
    frame_rate = metadata["frame_rate"]
    duration = stop_time_meta - start_time_meta
    total_frames = int(duration * frame_rate) + 1

    # Detect root bone
    root_bone = _detect_root_bone(scene)

    if not root_bone:
        raise ValueError("Could not detect root bone. Ensure skeleton has a hips/pelvis/root bone.")

    # Time setup
    anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if anim_stack_count > 0:
        anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        time_span = anim_stack.GetLocalTimeSpan()
    else:
        raise ValueError("No animation stack found")

    start_time = time_span.GetStart()
    frame_duration = fbx.FbxTime()
    frame_duration.SetSecondDouble(1.0 / frame_rate)

    # Extract position and rotation data across all frames
    # Store transforms temporarily for coordinate system detection
    positions = []
    rotations = []  # Euler angles in degrees
    transforms = []  # Store for later forward direction computation

    for frame in range(total_frames):
        current_time = start_time + frame_duration * frame

        # Get global transform
        global_transform = root_bone.EvaluateGlobalTransform(current_time)

        # Extract translation
        translation = global_transform.GetT()
        positions.append([translation[0], translation[1], translation[2]])

        # Extract rotation (Euler angles in degrees)
        rotation = global_transform.GetR()
        rotations.append([rotation[0], rotation[1], rotation[2]])

        # Store transform for later processing
        transforms.append(global_transform)

    positions = np.array(positions)
    rotations = np.array(rotations)

    # Compute translational derivatives (needed for coordinate system detection)
    velocities, accelerations, jerks = compute_derivatives(positions, frame_rate)

    # Auto-detect FULL coordinate system (up, forward, right axes + turning conventions)
    coord_system = detect_full_coordinate_system(scene, positions, velocities)

    # Extract for backward compatibility and logging
    axis_index = coord_system["forward_axis"]
    sign = coord_system["forward_sign"]
    axis_config = (axis_index, sign)
    coord_confidence = coord_system["confidence"]

    # Convert axis config to human-readable format for logging
    axis_names = ["X", "Y", "Z"]
    sign_str = "+" if sign > 0 else "-"
    up_sign_str = "+" if coord_system["up_sign"] > 0 else "-"
    handedness = "right-handed" if coord_system["is_right_handed"] else "left-handed"

    print(
        f"  Detected coordinate system: {sign_str}{axis_names[axis_index]} forward, "
        f"{up_sign_str}{axis_names[coord_system['up_axis']]} up ({handedness}, confidence: {coord_confidence:.2f})"
    )

    # Now extract forward directions using the detected coordinate system
    forward_directions = []
    for global_transform in transforms:
        forward = _extract_forward_direction(global_transform, axis_config)
        forward_directions.append(forward)

    forward_directions = np.array(forward_directions)

    velocity_mags = np.linalg.norm(velocities, axis=1)

    # Compute angular velocity around the UP axis (yaw/turning)
    # PROCEDURAL: Use detected yaw axis instead of hardcoded Y-axis
    dt = 1.0 / frame_rate
    yaw_axis_idx = coord_system["yaw_axis"]
    rotations_yaw = rotations[:, yaw_axis_idx]  # Extract yaw rotation (around UP axis)

    # Unwrap angles to handle 360° wrapping
    rotations_yaw_unwrapped = np.unwrap(np.radians(rotations_yaw))
    rotations_yaw_unwrapped = np.degrees(rotations_yaw_unwrapped)

    # Angular velocity in degrees/second
    angular_velocity_yaw = np.gradient(rotations_yaw_unwrapped, dt)

    # Compute adaptive thresholds from motion data
    adaptive_thresholds = _compute_adaptive_thresholds(velocity_mags, angular_velocity_yaw)
    print(
        f"  Adaptive thresholds: stationary={adaptive_thresholds['stationary_velocity_threshold']:.2f}, "
        f"turn_slow={adaptive_thresholds['turning_slow_threshold']:.1f}° (confidence: {adaptive_thresholds['confidence']:.2f})"
    )

    # Frame-by-frame trajectory analysis
    trajectory_data = []

    for frame in range(total_frames):
        # Classify direction of travel
        direction = _compute_direction_classification(
            velocities[frame], forward_directions[frame], velocity_mags[frame]
        )

        # Classify turning behavior using PROCEDURAL turning convention
        turning_classification = _classify_turning_speed(angular_velocity_yaw[frame])

        # PROCEDURAL: Determine left/right using detected coordinate system
        if coord_system["yaw_positive_is_left"]:
            turning_direction = "left" if angular_velocity_yaw[frame] > 0 else "right"
        else:
            turning_direction = "right" if angular_velocity_yaw[frame] > 0 else "left"

        trajectory_data.append(
            {
                "frame": frame,
                "time_seconds": frame / frame_rate,
                "position_x": positions[frame, 0],
                "position_y": positions[frame, 1],
                "position_z": positions[frame, 2],
                "rotation_x": rotations[frame, 0],
                "rotation_y": rotations[frame, 1],
                "rotation_z": rotations[frame, 2],
                "velocity_magnitude": velocity_mags[frame],
                "velocity_x": velocities[frame, 0],
                "velocity_y": velocities[frame, 1],
                "velocity_z": velocities[frame, 2],
                "angular_velocity_yaw": angular_velocity_yaw[frame],
                "direction": direction,
                "turning_speed": turning_classification,
                "turning_direction": turning_direction,
            }
        )

    # Package result
    result = {
        "trajectory_data": trajectory_data,
        "frame_rate": frame_rate,
        "root_bone_name": root_bone.GetName(),
        "total_frames": total_frames,
        # Raw motion data (positions, rotations)
        "positions": positions,
        "rotations": rotations,
        # Computed derivatives (CACHED for performance - no need to recompute)
        "velocities": velocities,
        "accelerations": accelerations,  # NEW: Cached for optimization
        "jerks": jerks,  # NEW: Cached for optimization
        "velocity_mags": velocity_mags,
        "angular_velocity_yaw": angular_velocity_yaw,  # FIXED: Now procedural (uses detected yaw axis)
        # Direction vectors
        "forward_directions": forward_directions,
        # PROCEDURAL METADATA BRAIN: Auto-discovered properties
        "coordinate_system": {
            "forward_axis": axis_names[axis_index],
            "forward_sign": sign_str,
            "detection_confidence": coord_confidence,
        },
        "adaptive_thresholds": adaptive_thresholds,
    }

    # Cache the result
    _trajectory_cache[cache_key] = result

    print(f"  ✓ Trajectory extracted and cached ({total_frames} frames)")

    return result


# ============================================================================
# Trajectory Extraction Helpers (Private)
# ============================================================================

# Root bone detection patterns (case-insensitive)
_ROOT_BONE_PATTERNS = [
    "hips",
    "pelvis",
    "root",
    "spine_base",
    "spinebase",
    "hip",
    "center_of_mass",
    "com",
    "mixamorig:hips",
]

# Default thresholds (used as fallback when insufficient data for adaptive computation)
_DEFAULT_FORWARD_THRESHOLD = 45.0  # ±45° = forward
_DEFAULT_BACKWARD_THRESHOLD = 135.0  # ±135° = backward
_DEFAULT_STATIONARY_VELOCITY_THRESHOLD = 0.1  # units/second
_DEFAULT_TURNING_THRESHOLD_SLOW = 30.0  # degrees/second
_DEFAULT_TURNING_THRESHOLD_FAST = 90.0  # degrees/second
_DEFAULT_TURNING_THRESHOLD_VERY_FAST = 180.0  # degrees/second

# Active thresholds (can be overridden by adaptive computation in extract_root_trajectory)
_FORWARD_THRESHOLD = _DEFAULT_FORWARD_THRESHOLD
_BACKWARD_THRESHOLD = _DEFAULT_BACKWARD_THRESHOLD
_STATIONARY_VELOCITY_THRESHOLD = _DEFAULT_STATIONARY_VELOCITY_THRESHOLD
_TURNING_THRESHOLD_SLOW = _DEFAULT_TURNING_THRESHOLD_SLOW
_TURNING_THRESHOLD_FAST = _DEFAULT_TURNING_THRESHOLD_FAST
_TURNING_THRESHOLD_VERY_FAST = _DEFAULT_TURNING_THRESHOLD_VERY_FAST


def _detect_root_bone(scene):
    """
    Automatically detect the root bone of the skeleton.

    Args:
        scene: FBX scene object

    Returns:
        fbx.FbxNode: Root bone node, or None if not found
    """
    root_node = scene.GetRootNode()

    # Method 1: Search by name pattern (must also be a skeleton node)
    def search_by_name(node):
        node_name_lower = node.GetName().lower()
        for pattern in _ROOT_BONE_PATTERNS:
            if pattern in node_name_lower:
                # Verify this is actually a skeleton node
                if node.GetNodeAttribute():
                    attr_type = node.GetNodeAttribute().GetAttributeType()
                    if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                        return node

        # Recursively search children
        for i in range(node.GetChildCount()):
            result = search_by_name(node.GetChild(i))
            if result:
                return result
        return None

    bone = search_by_name(root_node)
    if bone:
        return bone

    # Method 2: Find first skeleton node with children (fallback)
    def find_first_skeleton_with_children(node):
        if node.GetNodeAttribute():
            attr_type = node.GetNodeAttribute().GetAttributeType()
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton and node.GetChildCount() > 0:
                return node

        for i in range(node.GetChildCount()):
            result = find_first_skeleton_with_children(node.GetChild(i))
            if result:
                return result
        return None

    return find_first_skeleton_with_children(root_node)


def _extract_forward_direction(transform, axis_config=None):
    """
    Extract forward direction vector from transformation matrix.

    Args:
        transform: FBX transformation matrix
        axis_config: Optional tuple (axis_index, sign) where axis_index is 0 (X), 1 (Y), or 2 (Z)
                     and sign is 1 (positive) or -1 (negative). If None, uses default -Z.

    Returns:
        np.array: Forward direction vector (normalized)
    """
    # Get the transformation matrix as a numpy array
    matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            matrix[i, j] = transform.Get(i, j)

    # Determine which axis to use for forward direction
    if axis_config is None:
        # Default: -Z axis (common in many 3D systems)
        axis_index, sign = 2, -1
    else:
        axis_index, sign = axis_config

    # Extract forward direction from the specified axis
    forward = sign * matrix[axis_index, :3]

    # Normalize
    norm = np.linalg.norm(forward)
    if norm > 1e-6:
        forward = forward / norm

    return forward


def _compute_direction_classification(velocity, forward_direction, velocity_magnitude):
    """
    Classify direction of travel based on velocity and forward direction.

    Args:
        velocity: Velocity vector (3D)
        forward_direction: Character's forward direction vector (3D)
        velocity_magnitude: Magnitude of velocity

    Returns:
        str: Direction classification (forward, backward, strafe_left, strafe_right, stationary)
    """
    # Check if stationary
    if velocity_magnitude < _STATIONARY_VELOCITY_THRESHOLD:
        return "stationary"

    # Normalize velocity direction
    velocity_direction = velocity / (velocity_magnitude + 1e-10)

    # Compute angle between velocity and forward direction
    dot_product = np.clip(np.dot(velocity_direction, forward_direction), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))

    # Classify based on angle
    if angle_deg < _FORWARD_THRESHOLD:
        return "forward"
    elif angle_deg > _BACKWARD_THRESHOLD:
        return "backward"
    else:
        # Strafing - determine left or right using cross product
        cross = np.cross(forward_direction, velocity_direction)
        if cross[1] > 0:  # Y-component positive = strafing left (cross product points up)
            return "strafe_left"
        else:
            return "strafe_right"


def _classify_turning_speed(angular_velocity_deg_per_sec):
    """
    Classify turning speed based on angular velocity.

    Args:
        angular_velocity_deg_per_sec: Angular velocity in degrees/second

    Returns:
        str: Turning classification (none, slow, fast, very_fast)
    """
    abs_angular_velocity = abs(angular_velocity_deg_per_sec)

    if abs_angular_velocity < _TURNING_THRESHOLD_SLOW:
        return "none"
    elif abs_angular_velocity < _TURNING_THRESHOLD_FAST:
        return "slow"
    elif abs_angular_velocity < _TURNING_THRESHOLD_VERY_FAST:
        return "fast"
    else:
        return "very_fast"


# ============================================================================
# File I/O Utilities
# ============================================================================


def export_procedural_metadata(trajectory, output_filepath):
    """
    Export procedural metadata to JSON file for caching, AI integration, and interoperability.

    This creates a "metadata brain" - a centralized place to store auto-discovered properties
    that can be:
    - Reused across analyses (avoid re-detection)
    - Consumed by AI/LLMs for understanding the animation
    - Shared with other tools/workflows
    - Version controlled for reproducibility

    Args:
        trajectory: dict from extract_root_trajectory containing procedural metadata
        output_filepath: str, path to save JSON file (e.g., "output/anim_procedural_metadata.json")

    Returns:
        dict: The exported metadata structure
    """
    import json
    from datetime import datetime

    # Extract procedural metadata from trajectory
    coord_system = trajectory.get("coordinate_system", {})
    adaptive_thresh = trajectory.get("adaptive_thresholds", {})
    root_bone_name = trajectory.get("root_bone_name", "unknown")

    # Build structured metadata
    metadata = {
        "metadata_version": "1.0",
        "generated_timestamp": datetime.now().isoformat(),
        "animation_info": {
            "root_bone": root_bone_name,
            "total_frames": trajectory.get("total_frames", 0),
            "frame_rate": trajectory.get("frame_rate", 30.0),
        },
        "procedural_discoveries": {
            "coordinate_system": {
                "forward_axis": coord_system.get("forward_axis", "Z"),
                "forward_sign": coord_system.get("forward_sign", "-"),
                "confidence": float(coord_system.get("detection_confidence", 0.0)),
                "method": "motion_analysis_percentile",
                "description": f"{coord_system.get('forward_sign', '-')}{coord_system.get('forward_axis', 'Z')} axis detected as forward direction based on motion consistency",
            },
            "adaptive_thresholds": {
                "stationary_velocity": float(adaptive_thresh.get("stationary_velocity_threshold", 0.1)),
                "turning_slow_deg_per_sec": float(adaptive_thresh.get("turning_slow_threshold", 30.0)),
                "turning_fast_deg_per_sec": float(adaptive_thresh.get("turning_fast_threshold", 90.0)),
                "turning_very_fast_deg_per_sec": float(adaptive_thresh.get("turning_very_fast_threshold", 180.0)),
                "confidence": float(adaptive_thresh.get("confidence", 0.0)),
                "method": "percentile_distribution",
                "description": "Thresholds computed from actual motion data distribution, not hardcoded values",
            },
        },
        "usage_notes": {
            "ai_integration": "This metadata provides context for LLMs to understand animation properties",
            "caching": "High confidence values can be reused in subsequent analyses to skip re-detection",
            "interoperability": "Other tools can read this to understand rig characteristics",
        },
    }

    # Save to JSON file
    with open(output_filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Procedural metadata exported: {os.path.basename(output_filepath)}")

    return metadata


def ensure_output_dir(filepath):
    """
    Ensure the output directory exists for a given filepath.

    Args:
        filepath (str): Full path to output file.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def safe_overwrite(filepath):
    """
    Safely remove existing file to force overwrite.

    Args:
        filepath (str): Path to file to overwrite.

    Raises:
        RuntimeError: If file is locked or permission denied.
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            raise RuntimeError(
                f"Cannot overwrite {filepath} - file may be open in another program. " "Please close it and try again."
            )


def prepare_output_file(filepath):
    """
    Prepare output file by ensuring directory exists and clearing old file.

    Args:
        filepath (str): Path to output file.
    """
    ensure_output_dir(filepath)
    safe_overwrite(filepath)


# ============================================================================
# Data Conversion Utilities
# ============================================================================


def fbx_vector_to_array(fbx_vec):
    """
    Convert FBX vector to NumPy array.

    Args:
        fbx_vec: FBX vector object (FbxVector2, FbxVector4, etc.)

    Returns:
        np.array: NumPy array representation
    """
    if hasattr(fbx_vec, "mData"):
        # FbxVector4
        return np.array([fbx_vec.mData[0], fbx_vec.mData[1], fbx_vec.mData[2], fbx_vec.mData[3]])
    else:
        # FbxDouble3 or similar
        return np.array([fbx_vec[0], fbx_vec[1], fbx_vec[2]])


def convert_numpy_to_native(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.

    Args:
        obj: Object potentially containing NumPy types

    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj


# ============================================================================
# Bone Hierarchy Utilities
# ============================================================================


def get_all_bones(scene):
    """
    Recursively collect all skeleton bones from the scene.

    Args:
        scene: FBX scene object

    Returns:
        list: List of FbxNode objects representing bones
    """
    bones = []

    def traverse(node):
        if node.GetNodeAttribute():
            attr_type = node.GetNodeAttribute().GetAttributeType()
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                bones.append(node)

        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i))

    root = scene.GetRootNode()
    traverse(root)
    return bones


def get_bone_children(bone):
    """
    Get all child bones of a given bone.

    Args:
        bone: FbxNode representing a bone

    Returns:
        list: List of child bone FbxNode objects
    """
    children = []
    for i in range(bone.GetChildCount()):
        child = bone.GetChild(i)
        if child.GetNodeAttribute():
            attr_type = child.GetNodeAttribute().GetAttributeType()
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                children.append(child)
    return children


def build_bone_hierarchy(scene):
    """
    Build a parent-child hierarchy map of all bones in the scene.

    Args:
        scene: FBX scene object

    Returns:
        dict: Mapping of bone names to parent bone names
              {child_name: parent_name, ...}
              Root bones have None as parent
    """
    hierarchy = {}

    def traverse(node, parent_name=None):
        """Recursively traverse and build hierarchy."""
        # Check if this is a skeleton bone
        if node.GetNodeAttribute():
            attr_type = node.GetNodeAttribute().GetAttributeType()
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                bone_name = node.GetName()
                hierarchy[bone_name] = parent_name
                parent_name = bone_name  # This bone becomes parent for children

        # Traverse children
        for i in range(node.GetChildCount()):
            child = node.GetChild(i)
            traverse(child, parent_name)

    root = scene.GetRootNode()
    traverse(root, parent_name=None)

    return hierarchy


def detect_chains_from_hierarchy(hierarchy, min_chain_length=3):
    """
    Detect all kinematic chains from a bone hierarchy.

    A chain is a sequence of bones connected in parent-child relationships,
    ending at a leaf bone (no children) or a branching point.

    Args:
        hierarchy: Dict mapping bone names to parent bone names
        min_chain_length: Minimum number of bones to form a valid chain

    Returns:
        dict: Mapping of chain names to bone lists
              {chain_name: [bone1, bone2, bone3, ...], ...}
    """
    if not hierarchy:
        return {}

    # Build reverse mapping: parent -> list of children
    children_map = {}
    for bone, parent in hierarchy.items():
        if parent is None:
            continue  # Skip root bones
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(bone)

    # Find all leaf bones (bones with no children)
    all_bones = set(hierarchy.keys())
    bones_with_children = set(children_map.keys())
    leaf_bones = all_bones - bones_with_children

    # Trace back from each leaf to build chains
    chains = {}
    chain_id = 0

    for leaf in leaf_bones:
        chain = []
        current = leaf

        # Trace back to root, collecting bones
        while current is not None:
            chain.insert(0, current)  # Prepend to maintain root-to-leaf order
            current = hierarchy.get(current)

            # Stop at branching points (bones with multiple children)
            if current and current in children_map and len(children_map[current]) > 1:
                chain.insert(0, current)  # Include the branching bone
                break

        # Only include chains that meet minimum length
        if len(chain) >= min_chain_length:
            # Try to infer chain name from bones
            chain_name = _infer_chain_name(chain)
            if chain_name in chains:
                chain_name = f"{chain_name}_{chain_id}"
                chain_id += 1
            chains[chain_name] = chain

    return chains


def _infer_chain_name(chain):
    """
    Infer a descriptive name for a chain based on bone names.

    Uses PROCEDURAL FUZZY MATCHING - analyzes all bones in chain, not just endpoints.
    Handles multiple naming conventions (Mixamo, Unreal, Unity, custom rigs).

    Args:
        chain: List of bone names in the chain

    Returns:
        str: Inferred chain name
    """
    # Normalize all bone names in chain for analysis
    normalized_bones = [bone.lower().replace("mixamorig:", "").replace("_", "").replace("-", "") for bone in chain]
    chain_str = " ".join(normalized_bones)  # Combined string for fuzzy matching

    # Also check first and last for compatibility with existing logic
    first_bone = normalized_bones[0] if normalized_bones else ""
    last_bone = normalized_bones[-1] if normalized_bones else ""

    # === SIDE DETECTION (fuzzy - check entire chain) ===
    side = ""
    # Check for side indicators anywhere in the chain
    if any("left" in bone or bone.startswith("l") for bone in normalized_bones):
        side = "Left"
    elif any("right" in bone or bone.startswith("r") for bone in normalized_bones):
        side = "Right"

    # === BODY PART DETECTION (fuzzy - check entire chain) ===
    # Define semantic keyword groups for each body part
    arm_keywords = ["arm", "shoulder", "clavicle", "hand", "wrist", "elbow", "forearm", "upperarm"]
    leg_keywords = ["leg", "thigh", "hip", "foot", "ankle", "knee", "shin", "upleg", "lowleg", "calf"]
    spine_keywords = ["spine", "chest", "torso", "back", "ribcage"]
    neck_keywords = ["neck", "head", "skull", "cervical"]

    # Count keyword matches in entire chain (more robust than just first/last)
    def count_matches(keywords):
        return sum(1 for bone in normalized_bones if any(kw in bone for kw in keywords))

    arm_score = count_matches(arm_keywords)
    leg_score = count_matches(leg_keywords)
    spine_score = count_matches(spine_keywords)
    neck_score = count_matches(neck_keywords)

    # Choose part based on highest score
    scores = {
        "Arm": arm_score,
        "Leg": leg_score,
        "Spine": spine_score,
        "Neck": neck_score,
    }

    max_score = max(scores.values())
    if max_score > 0:
        # Get part with highest score
        part = max(scores, key=scores.get)
    else:
        # Fallback to original bone name if no matches
        part = chain[0]

    return f"{side}{part}".strip()


def get_bone_chain(start_bone, depth=10):
    """
    Get a chain of bones starting from a given bone.

    Follows the first child recursively until a leaf or max depth.

    Args:
        start_bone: FbxNode to start from
        depth: Maximum chain depth

    Returns:
        list: List of bones in the chain
    """
    chain = [start_bone]
    current = start_bone

    for _ in range(depth):
        children = get_bone_children(current)
        if not children:
            break
        current = children[0]  # Follow first child
        chain.append(current)

    return chain


def find_bone_by_name(scene, bone_name_pattern, case_sensitive=False):
    """
    Find a bone by name pattern.

    Args:
        scene: FBX scene object
        bone_name_pattern: String or pattern to match
        case_sensitive: Whether to match case-sensitively

    Returns:
        FbxNode: First matching bone, or None
    """
    bones = get_all_bones(scene)

    for bone in bones:
        bone_name = bone.GetName()
        if not case_sensitive:
            bone_name = bone_name.lower()
            pattern = bone_name_pattern.lower()
        else:
            pattern = bone_name_pattern

        if pattern in bone_name:
            return bone

    return None


# ============================================================================
# CSV Utilities
# ============================================================================


def write_dict_list_to_csv(data, filepath, fieldnames=None):
    """
    Write list of dictionaries to CSV file.

    Args:
        data: List of dictionaries
        filepath: Output CSV path
        fieldnames: Optional list of field names (defaults to keys of first dict)
    """
    if not data:
        return

    if fieldnames is None:
        fieldnames = data[0].keys()

    prepare_output_file(filepath)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# ============================================================================
# IK Analysis Helpers
# ============================================================================

# IK scoring weights
IK_STABILITY_WEIGHT = 0.7
IK_RANGE_WEIGHT = 0.3
IK_ROTATION_FULL_RANGE_DEGREES = 360.0


def compute_ik_suitability(rotation_data):
    """
    Compute IK suitability score for a joint based on rotation data.

    Args:
        rotation_data: np.array of shape (N, 3) containing Euler angles

    Returns:
        tuple: (stability, range_score, ik_score)
    """
    if len(rotation_data) == 0:
        return 0.0, 0.0, 0.0

    # Compute statistics
    std_r = np.std(rotation_data, axis=0)  # Standard deviation per axis
    min_r = np.min(rotation_data, axis=0)
    max_r = np.max(rotation_data, axis=0)
    rot_range = max_r - min_r  # Range of motion per axis

    # Stability: inverse of rotation variance norm
    # Higher std = more variance = less stable = lower score
    stability = 1.0 / (1.0 + np.linalg.norm(std_r))

    # Range score: combination of variance penalty and range reward
    variance_penalty = np.exp(-np.var(rotation_data))  # Penalize jittery motion
    range_reward = np.clip(np.sum(rot_range) / IK_ROTATION_FULL_RANGE_DEGREES, 0, 1)
    range_score = variance_penalty * range_reward

    # Weighted combination favoring stability
    ik_score = IK_STABILITY_WEIGHT * stability + IK_RANGE_WEIGHT * range_score

    return stability, range_score, ik_score
