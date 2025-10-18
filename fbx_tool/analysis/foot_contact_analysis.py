"""
Foot Contact Analysis Module

Detects and analyzes foot-ground contact events for locomotion quality assessment.

Features:
- **Adaptive Threshold Calculation:** Automatically derives velocity and height thresholds
  from animation data distribution using gap detection and percentile methods. Works
  across different character scales and animation styles without hardcoded values.
- **Root Motion Compatible:** Percentile-based ground height estimation (5th percentile)
  ignores glitches and works with elevated ground planes.
- **Scale Invariant:** Uses data-driven thresholds that adapt to small characters
  (cm scale) and large characters (m scale) alike.
- **Outlier Robust:** Filters extreme outliers using statistical methods (MAD, IQR,
  cluster size validation) before threshold calculation.
- Automatic foot bone detection (supports Mixamo, Unity, Blender, custom rigs)
- Ground contact detection (touchdown/liftoff events)
- Foot sliding detection and quantification
- Ground penetration depth measurement
- Contact stability scoring
- Contact phase segmentation (stance vs swing)

Adaptive Algorithms:
- `calculate_adaptive_velocity_threshold()`: Gap detection with cluster validation
- `calculate_adaptive_height_threshold()`: Bimodal distribution separation
- `compute_ground_height_percentile()`: Robust ground plane estimation
- `detect_contact_events_adaptive()`: End-to-end adaptive contact detection

Outputs:
- foot_contacts.csv: Contact events with frame ranges and metrics
- foot_sliding.csv: Detected sliding with distance and severity
- foot_trajectories.csv: Frame-by-frame foot positions and states
- contact_summary.csv: Overall contact quality metrics per foot

See CLAUDE.md and docs/development/INCOMPLETE_MODULES.md for implementation details.
"""

import csv
import os

import fbx
import numpy as np

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import ensure_output_dir
from fbx_tool.analysis.velocity_analysis import compute_derivatives

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Contact Detection Thresholds (Default fallbacks for non-adaptive mode)
CONTACT_HEIGHT_THRESHOLD = 5.0  # Maximum height above ground for contact (units)
CONTACT_VELOCITY_THRESHOLD = 10.0  # Maximum velocity magnitude for contact (units/s)

# Adaptive Threshold Percentiles
VELOCITY_THRESHOLD_PERCENTILE = 30  # Use 30th percentile of velocity distribution
HEIGHT_THRESHOLD_PERCENTILE = 75  # Use 75th percentile of low-height distribution
GROUND_HEIGHT_PERCENTILE = 5  # Use 5th percentile to avoid glitches

# Minimum thresholds to avoid zero/negative values
MIN_VELOCITY_THRESHOLD = 0.1  # Minimum velocity threshold (units/s)
MIN_HEIGHT_THRESHOLD = 0.5  # Minimum height threshold (units)

# Sliding Detection
SLIDING_THRESHOLD = 5.0  # Minimum horizontal velocity for sliding (units/s)
SLIDING_DISTANCE_HIGH = 20.0  # High severity threshold (units)
SLIDING_DISTANCE_MEDIUM = 5.0  # Medium severity threshold (units)

# Penetration Severity
PENETRATION_DEPTH_HIGH = 5.0  # High severity threshold (units)
PENETRATION_DEPTH_MEDIUM = 2.0  # Medium severity threshold (units)

# Stability Scoring Weights
STABILITY_WEIGHT_POSITION = 0.5  # Weight for position variance
STABILITY_WEIGHT_VELOCITY = 0.3  # Weight for velocity magnitude
STABILITY_WEIGHT_ACCELERATION = 0.2  # Weight for acceleration magnitude

# Stability Quality Thresholds
STABILITY_EXCELLENT = 0.8
STABILITY_GOOD = 0.6
STABILITY_FAIR = 0.4

# Common foot bone naming patterns (case-insensitive)
FOOT_BONE_PATTERNS = {
    "left": [
        "left foot",
        "leftfoot",
        "l_foot",
        "lfoot",
        "foot_l",
        "footl",
        "left ankle",
        "leftankle",
        "l_ankle",
        "lankle",
        "ankle_l",
        "anklel",
        "left toe",
        "lefttoe",
        "l_toe",
        "ltoe",
        "toe_l",
        "toel",
    ],
    "right": [
        "right foot",
        "rightfoot",
        "r_foot",
        "rfoot",
        "foot_r",
        "footr",
        "right ankle",
        "rightankle",
        "r_ankle",
        "rankle",
        "ankle_r",
        "ankler",
        "right toe",
        "righttoe",
        "r_toe",
        "rtoe",
        "toe_r",
        "toer",
    ],
}


# ==============================================================================
# ADAPTIVE THRESHOLD CALCULATION
# ==============================================================================


def calculate_adaptive_velocity_threshold(velocities, percentile=None):
    """
    Calculate velocity threshold adaptively from data distribution.

    Uses gap detection or percentile-based approach to separate stance (low velocity)
    from swing (high velocity) phases. Robust to outliers and different animation scales.

    Args:
        velocities: Array of velocity magnitudes (1D or scalar velocities)
        percentile: Percentile to use (default: VELOCITY_THRESHOLD_PERCENTILE)

    Returns:
        float: Adaptive velocity threshold
    """
    if percentile is None:
        percentile = VELOCITY_THRESHOLD_PERCENTILE

    velocities = np.asarray(velocities).flatten()

    # Handle empty or invalid data
    if len(velocities) == 0:
        return MIN_VELOCITY_THRESHOLD

    # Filter out NaN/inf values
    valid_velocities = velocities[np.isfinite(velocities)]
    if len(valid_velocities) == 0:
        return MIN_VELOCITY_THRESHOLD

    # Handle zero variance (all values the same)
    if np.std(valid_velocities) < 1e-10:
        # Return the constant value (or slightly above if zero)
        constant_value = np.mean(valid_velocities)
        return max(constant_value, MIN_VELOCITY_THRESHOLD)

    # Remove extreme outliers using MAD (Median Absolute Deviation)
    # More robust than IQR when most values are identical
    median = np.median(valid_velocities)
    mad = np.median(np.abs(valid_velocities - median))

    if mad > 1e-10:  # MAD is non-zero
        # Modified Z-score: (value - median) / (1.4826 * MAD)
        # 1.4826 is the constant to make MAD comparable to std dev for normal distribution
        # Values with modified z-score > 3.5 are outliers
        modified_z_scores = 0.6745 * (valid_velocities - median) / mad
        filtered_velocities = valid_velocities[np.abs(modified_z_scores) < 3.5]

        # Use filtered data if we removed outliers
        if len(filtered_velocities) > 0 and len(filtered_velocities) < len(valid_velocities):
            valid_velocities = filtered_velocities
    else:
        # MAD is zero - try IQR as fallback
        q1 = np.percentile(valid_velocities, 25)
        q3 = np.percentile(valid_velocities, 75)
        iqr = q3 - q1

        if iqr > 0:
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            filtered_velocities = valid_velocities[
                (valid_velocities >= lower_bound) & (valid_velocities <= upper_bound)
            ]

            if len(filtered_velocities) > 0 and len(filtered_velocities) < len(valid_velocities):
                valid_velocities = filtered_velocities

    # Strategy: Look for gap in sorted velocities (bimodal distribution)
    # Stance velocities (low) vs swing velocities (high) typically have a gap
    sorted_velocities = np.sort(valid_velocities)

    # Calculate gaps between consecutive values
    if len(sorted_velocities) > 1:
        gaps = np.diff(sorted_velocities)

        # Find the largest gap (likely between stance and swing clusters)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)

            # Additional check: gap should not involve only outliers
            # At least 10% of data should be on each side of the gap
            values_below_gap = max_gap_idx + 1  # Number of values below gap
            values_above_gap = len(sorted_velocities) - values_below_gap
            min_cluster_size = max(3, int(0.1 * len(sorted_velocities)))  # At least 10% or 3 values

            # Threshold is the midpoint of the largest gap
            # Only use gap if:
            # 1. Gap is significant (> 10% of data range)
            # 2. Both sides have reasonable number of points (not just outliers)
            data_range = sorted_velocities[-1] - sorted_velocities[0]
            if (
                data_range > 0
                and gaps[max_gap_idx] > 0.1 * data_range
                and values_below_gap >= min_cluster_size
                and values_above_gap >= min_cluster_size
            ):
                threshold = (sorted_velocities[max_gap_idx] + sorted_velocities[max_gap_idx + 1]) / 2
                threshold = max(threshold, MIN_VELOCITY_THRESHOLD)
                return float(threshold)

    # Fallback: Use percentile-based approach
    # Use higher percentile (50th) to better separate clusters when no clear gap
    threshold = np.percentile(valid_velocities, min(50, percentile * 1.5))

    # Ensure minimum threshold
    threshold = max(threshold, MIN_VELOCITY_THRESHOLD)

    return float(threshold)


def calculate_adaptive_height_threshold(heights_above_ground, percentile=None):
    """
    Calculate height threshold adaptively from foot trajectory.

    Analyzes the distribution of heights to separate ground contact (low height)
    from aerial phases (high height). Uses gap detection for bimodal distributions.

    Args:
        heights_above_ground: Array of heights above ground level
        percentile: Percentile of low heights to use (default: HEIGHT_THRESHOLD_PERCENTILE)

    Returns:
        float: Adaptive height threshold
    """
    if percentile is None:
        percentile = HEIGHT_THRESHOLD_PERCENTILE

    heights = np.asarray(heights_above_ground).flatten()

    # Handle empty or invalid data
    if len(heights) == 0:
        return MIN_HEIGHT_THRESHOLD

    # Filter out NaN/inf values
    valid_heights = heights[np.isfinite(heights)]
    if len(valid_heights) == 0:
        return MIN_HEIGHT_THRESHOLD

    # Handle zero variance
    if np.std(valid_heights) < 1e-10:
        constant_value = np.mean(valid_heights)
        # If foot is always at same height, use a small threshold above it
        return max(constant_value + 1.0, MIN_HEIGHT_THRESHOLD)

    # Strategy: Look for gap in sorted heights (bimodal: ground vs aerial)
    sorted_heights = np.sort(valid_heights)

    # Calculate gaps between consecutive values
    if len(sorted_heights) > 1:
        gaps = np.diff(sorted_heights)

        # Find the largest gap
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)

            # Use gap if it's significant (> 10% of range)
            data_range = sorted_heights[-1] - sorted_heights[0]
            if data_range > 0 and gaps[max_gap_idx] > 0.1 * data_range:
                threshold = (sorted_heights[max_gap_idx] + sorted_heights[max_gap_idx + 1]) / 2
                threshold = max(threshold, MIN_HEIGHT_THRESHOLD)
                return float(threshold)

    # Fallback: Use percentile of lower half
    median_height = np.median(valid_heights)
    low_heights = valid_heights[valid_heights <= median_height]

    if len(low_heights) > 0:
        # Use high percentile of low heights
        threshold = np.percentile(low_heights, percentile)
    else:
        threshold = np.percentile(valid_heights, percentile / 2)

    # Ensure minimum threshold
    threshold = max(threshold, MIN_HEIGHT_THRESHOLD)

    return float(threshold)


def compute_ground_height_percentile(y_positions, percentile=None):
    """
    Estimate ground height using percentile instead of absolute minimum.

    This approach is robust to glitches and penetration errors in the data.

    Args:
        y_positions: Array of Y positions across all frames
        percentile: Percentile to use (default: GROUND_HEIGHT_PERCENTILE)

    Returns:
        float: Estimated ground height
    """
    if percentile is None:
        percentile = GROUND_HEIGHT_PERCENTILE

    y_positions = np.asarray(y_positions).flatten()

    # Filter out NaN/inf values
    valid_positions = y_positions[np.isfinite(y_positions)]

    if len(valid_positions) == 0:
        return 0.0

    # Use percentile instead of minimum to avoid glitches
    # 5th percentile captures the "typical lowest position" while ignoring extreme outliers
    ground_height = np.percentile(valid_positions, percentile)

    return float(ground_height)


def detect_contact_events_adaptive(positions, velocities, ground_height):
    """
    Detect contact events with adaptive thresholds calculated from data.

    Automatically determines appropriate thresholds based on the animation's
    velocity and height distributions. Works across different character scales
    and animation styles.

    Args:
        positions: Array of foot positions (n_frames, 3)
        velocities: Array of foot velocities (n_frames, 3)
        ground_height: Estimated ground level

    Returns:
        list: Contact segments as (start_frame, end_frame) tuples
    """
    if len(positions) == 0:
        return []

    # Calculate heights above ground
    heights = positions[:, 1] - ground_height

    # Calculate velocity magnitudes
    velocity_mags = np.linalg.norm(velocities, axis=1)

    # Adaptively calculate thresholds from data distribution
    velocity_threshold = calculate_adaptive_velocity_threshold(velocity_mags)
    height_threshold = calculate_adaptive_height_threshold(heights)

    # Use the adaptive thresholds for detection
    return detect_contact_events(
        positions, velocities, ground_height, height_threshold=height_threshold, velocity_threshold=velocity_threshold
    )


def detect_foot_bones(bones):
    """
    Automatically detect left and right foot bones from skeleton.

    Supports multiple naming conventions (Mixamo, Unity, Blender, custom rigs).

    Args:
        bones: List of bone nodes

    Returns:
        dict: {'left': bone_node, 'right': bone_node} or None if not found
    """
    foot_bones = {"left": None, "right": None}

    for bone in bones:
        bone_name_lower = bone.GetName().lower()

        # Check left foot patterns
        if foot_bones["left"] is None:
            for pattern in FOOT_BONE_PATTERNS["left"]:
                if pattern in bone_name_lower:
                    foot_bones["left"] = bone
                    break

        # Check right foot patterns
        if foot_bones["right"] is None:
            for pattern in FOOT_BONE_PATTERNS["right"]:
                if pattern in bone_name_lower:
                    foot_bones["right"] = bone
                    break

        # Early exit if both found
        if foot_bones["left"] is not None and foot_bones["right"] is not None:
            break

    return foot_bones


def compute_ground_height(bones, scene, start_time, frame_duration, total_frames, foot_bones):
    """
    Estimate ground height from foot trajectories.

    Uses the minimum Y position across all frames as ground level.

    Args:
        bones: All bone nodes
        scene: FBX scene
        start_time: Animation start time
        frame_duration: Duration per frame
        total_frames: Total number of frames
        foot_bones: Dict of detected foot bones

    Returns:
        float: Estimated ground height
    """
    min_y_positions = []

    # Sample foot positions
    for side, foot_bone in foot_bones.items():
        if foot_bone is None:
            continue

        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame
            translation = foot_bone.EvaluateGlobalTransform(current_time).GetT()
            min_y_positions.append(translation[1])  # Y is up

    if not min_y_positions:
        return 0.0

    # Ground is minimum Y position (with small buffer for noise)
    ground_height = np.min(min_y_positions)

    return ground_height


def detect_contact_events(positions, velocities, ground_height, height_threshold=5.0, velocity_threshold=10.0):
    """
    Detect ground contact events (touchdown and liftoff).

    Contact criteria:
    - Height: Foot Y position is close to ground (within threshold)
    - Velocity: Foot vertical velocity is low (nearly stationary)

    Args:
        positions: Array of foot positions (n_frames, 3)
        velocities: Array of foot velocities (n_frames, 3)
        ground_height: Estimated ground level
        height_threshold: Maximum height above ground for contact (units)
        velocity_threshold: Maximum velocity magnitude for contact (units/s)

    Returns:
        list: Contact segments as (start_frame, end_frame) tuples
    """
    n_frames = len(positions)

    # Compute height above ground
    heights = positions[:, 1] - ground_height  # Y is up

    # Compute velocity magnitudes
    velocity_mags = np.linalg.norm(velocities, axis=1)

    # Contact mask: both height and velocity criteria
    contact_mask = (heights < height_threshold) & (velocity_mags < velocity_threshold)

    # Find consecutive contact frames
    contact_segments = []
    in_contact = False
    start_frame = 0

    for frame in range(n_frames):
        if contact_mask[frame] and not in_contact:
            # Touchdown
            start_frame = frame
            in_contact = True
        elif not contact_mask[frame] and in_contact:
            # Liftoff
            contact_segments.append((start_frame, frame - 1))
            in_contact = False

    # Handle case where animation ends in contact
    if in_contact:
        contact_segments.append((start_frame, n_frames - 1))

    return contact_segments


def detect_foot_sliding(positions, velocities, contact_segments, sliding_threshold=5.0):
    """
    Detect foot sliding during ground contact.

    Sliding occurs when foot moves horizontally while in contact with ground.

    Args:
        positions: Array of foot positions (n_frames, 3)
        velocities: Array of foot velocities (n_frames, 3)
        contact_segments: List of (start_frame, end_frame) contact tuples
        sliding_threshold: Minimum horizontal velocity for sliding (units/s)

    Returns:
        list: Sliding events with metrics
    """
    sliding_events = []

    for start_frame, end_frame in contact_segments:
        if end_frame - start_frame < 2:
            continue  # Need at least 2 frames

        # Extract contact segment data
        segment_positions = positions[start_frame : end_frame + 1]
        segment_velocities = velocities[start_frame : end_frame + 1]

        # Horizontal velocity (XZ plane)
        horizontal_velocities = segment_velocities.copy()
        horizontal_velocities[:, 1] = 0  # Zero out Y component
        horizontal_speeds = np.linalg.norm(horizontal_velocities, axis=1)

        # Detect sliding frames
        sliding_mask = horizontal_speeds > sliding_threshold

        if np.any(sliding_mask):
            # Calculate sliding distance
            sliding_distance = 0.0
            for i in range(len(segment_positions) - 1):
                if sliding_mask[i]:
                    # XZ distance between consecutive frames
                    pos1 = segment_positions[i][[0, 2]]  # X, Z
                    pos2 = segment_positions[i + 1][[0, 2]]
                    sliding_distance += np.linalg.norm(pos2 - pos1)

            # Peak sliding speed
            peak_sliding_speed = np.max(horizontal_speeds[sliding_mask])

            # Percentage of contact frames with sliding
            sliding_percentage = (np.sum(sliding_mask) / len(sliding_mask)) * 100

            sliding_events.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "sliding_distance": sliding_distance,
                    "peak_sliding_speed": peak_sliding_speed,
                    "sliding_percentage": sliding_percentage,
                    "severity": (
                        "high"
                        if sliding_distance > SLIDING_DISTANCE_HIGH
                        else ("medium" if sliding_distance > SLIDING_DISTANCE_MEDIUM else "low")
                    ),
                }
            )

    return sliding_events


def measure_ground_penetration(positions, ground_height, contact_segments):
    """
    Measure ground penetration depth during contact.

    Penetration occurs when foot goes below ground level.

    Args:
        positions: Array of foot positions (n_frames, 3)
        ground_height: Estimated ground level
        contact_segments: List of (start_frame, end_frame) contact tuples

    Returns:
        list: Penetration events with depth metrics
    """
    penetration_events = []

    for start_frame, end_frame in contact_segments:
        segment_positions = positions[start_frame : end_frame + 1]
        segment_heights = segment_positions[:, 1] - ground_height

        # Find frames with penetration (negative height)
        penetration_mask = segment_heights < 0

        if np.any(penetration_mask):
            penetration_depths = np.abs(segment_heights[penetration_mask])
            max_penetration = np.max(penetration_depths)
            mean_penetration = np.mean(penetration_depths)
            penetration_percentage = (np.sum(penetration_mask) / len(segment_heights)) * 100

            penetration_events.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "max_penetration_depth": max_penetration,
                    "mean_penetration_depth": mean_penetration,
                    "penetration_percentage": penetration_percentage,
                    "severity": (
                        "high"
                        if max_penetration > PENETRATION_DEPTH_HIGH
                        else ("medium" if max_penetration > PENETRATION_DEPTH_MEDIUM else "low")
                    ),
                }
            )

    return penetration_events


def compute_contact_stability(positions, velocities, accelerations, contact_segments):
    """
    Compute stability score for each contact segment.

    Stable contact has:
    - Low position variance (foot stays still)
    - Low velocity (minimal movement)
    - Low acceleration (smooth transition)

    Args:
        positions: Array of foot positions (n_frames, 3)
        velocities: Array of foot velocities (n_frames, 3)
        accelerations: Array of foot accelerations (n_frames, 3)
        contact_segments: List of (start_frame, end_frame) contact tuples

    Returns:
        list: Stability metrics for each contact
    """
    stability_scores = []

    for start_frame, end_frame in contact_segments:
        segment_positions = positions[start_frame : end_frame + 1]
        segment_velocities = velocities[start_frame : end_frame + 1]
        segment_accelerations = accelerations[start_frame : end_frame + 1]

        # Position stability (low variance = stable)
        position_variance = np.var(segment_positions, axis=0)
        position_stability = 1.0 / (1.0 + np.sum(position_variance))

        # Velocity stability (low mean velocity = stable)
        velocity_mags = np.linalg.norm(segment_velocities, axis=1)
        mean_velocity = np.mean(velocity_mags)
        velocity_stability = 1.0 / (1.0 + mean_velocity * 0.1)

        # Acceleration stability (low mean acceleration = smooth)
        acceleration_mags = np.linalg.norm(segment_accelerations, axis=1)
        mean_acceleration = np.mean(acceleration_mags)
        acceleration_stability = 1.0 / (1.0 + mean_acceleration * 0.01)

        # Overall stability (weighted average)
        overall_stability = (
            STABILITY_WEIGHT_POSITION * position_stability
            + STABILITY_WEIGHT_VELOCITY * velocity_stability
            + STABILITY_WEIGHT_ACCELERATION * acceleration_stability
        )

        stability_scores.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "position_stability": position_stability,
                "velocity_stability": velocity_stability,
                "acceleration_stability": acceleration_stability,
                "overall_stability": overall_stability,
                "quality": (
                    "excellent"
                    if overall_stability > STABILITY_EXCELLENT
                    else (
                        "good"
                        if overall_stability > STABILITY_GOOD
                        else ("fair" if overall_stability > STABILITY_FAIR else "poor")
                    )
                ),
            }
        )

    return stability_scores


def analyze_foot_contacts(scene, output_dir="output/"):
    """
    Comprehensive foot contact analysis for locomotion quality.

    Args:
        scene: FBX scene object
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary statistics
    """
    ensure_output_dir(output_dir)

    # Get scene metadata
    metadata = get_scene_metadata(scene)

    if not metadata.get("has_animation", False):
        raise ValueError("No animation data found in scene")

    start_time_meta = metadata["start_time"]
    stop_time_meta = metadata["stop_time"]
    frame_rate = metadata["frame_rate"]
    duration = stop_time_meta - start_time_meta
    total_frames = int(duration * frame_rate) + 1  # +1 to include both start and end frames

    # Get root node and collect all bones
    root_node = scene.GetRootNode()
    bones = []

    def collect_bones(node):
        if node.GetNodeAttribute():
            attr_type = node.GetNodeAttribute().GetAttributeType()
            # Check if node is a skeleton bone
            if attr_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                bones.append(node)
        for i in range(node.GetChildCount()):
            collect_bones(node.GetChild(i))

    collect_bones(root_node)

    if not bones:
        raise ValueError("No bones found in scene")

    # Detect foot bones
    print("Detecting foot bones...")
    foot_bones = detect_foot_bones(bones)

    if foot_bones["left"] is None and foot_bones["right"] is None:
        raise ValueError("Could not detect foot bones. Ensure skeleton has 'foot', 'ankle', or 'toe' bones.")

    detected_feet = []
    if foot_bones["left"]:
        detected_feet.append(f"Left: {foot_bones['left'].GetName()}")
    if foot_bones["right"]:
        detected_feet.append(f"Right: {foot_bones['right'].GetName()}")

    print(f"  ✓ Detected: {', '.join(detected_feet)}")

    # Time setup
    # Get the current animation stack's time span
    anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if anim_stack_count > 0:
        anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        time_span = anim_stack.GetLocalTimeSpan()
    else:
        raise ValueError("No animation stack found")

    start_time = time_span.GetStart()
    frame_duration = fbx.FbxTime()
    frame_duration.SetSecondDouble(1.0 / frame_rate)

    # Estimate ground height using percentile method (robust to glitches)
    print("Computing ground height...")
    # Collect all foot Y positions
    all_y_positions = []
    for side, foot_bone in foot_bones.items():
        if foot_bone is None:
            continue
        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame
            translation = foot_bone.EvaluateGlobalTransform(current_time).GetT()
            all_y_positions.append(translation[1])

    ground_height = compute_ground_height_percentile(all_y_positions, percentile=GROUND_HEIGHT_PERCENTILE)
    print(f"  ✓ Ground height: {ground_height:.2f} units (using {GROUND_HEIGHT_PERCENTILE}th percentile)")

    # Results storage
    all_contacts = []
    all_sliding = []
    all_trajectories = []
    contact_summaries = []

    # Analyze each foot
    for side, foot_bone in foot_bones.items():
        if foot_bone is None:
            continue

        print(f"\nAnalyzing {side} foot ({foot_bone.GetName()})...")

        # Extract position data
        positions = []
        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame
            translation = foot_bone.EvaluateGlobalTransform(current_time).GetT()
            positions.append([translation[0], translation[1], translation[2]])

        positions = np.array(positions)

        # Compute derivatives
        velocities, accelerations, _ = compute_derivatives(positions, frame_rate)

        # Detect contact events using adaptive thresholds
        contact_segments = detect_contact_events_adaptive(positions, velocities, ground_height)

        print(f"  ✓ {len(contact_segments)} contact events detected (adaptive thresholds)")

        # Detect foot sliding
        sliding_events = detect_foot_sliding(
            positions, velocities, contact_segments, sliding_threshold=SLIDING_THRESHOLD
        )

        if sliding_events:
            print(f"  ⚠ {len(sliding_events)} sliding events detected")

        # Measure ground penetration
        penetration_events = measure_ground_penetration(positions, ground_height, contact_segments)

        if penetration_events:
            print(f"  ⚠ {len(penetration_events)} penetration events detected")

        # Compute contact stability
        stability_scores = compute_contact_stability(positions, velocities, accelerations, contact_segments)

        # Store contacts with all metrics
        for i, (start, end) in enumerate(contact_segments):
            contact_duration = (end - start + 1) / frame_rate

            # Find matching sliding/penetration/stability
            sliding_info = next((s for s in sliding_events if s["start_frame"] == start), None)
            penetration_info = next((p for p in penetration_events if p["start_frame"] == start), None)
            stability_info = stability_scores[i] if i < len(stability_scores) else None

            all_contacts.append(
                {
                    "foot": side,
                    "bone_name": foot_bone.GetName(),
                    "start_frame": start,
                    "end_frame": end,
                    "duration_frames": end - start + 1,
                    "duration_seconds": contact_duration,
                    "has_sliding": sliding_info is not None,
                    "sliding_distance": sliding_info["sliding_distance"] if sliding_info else 0.0,
                    "has_penetration": penetration_info is not None,
                    "max_penetration": penetration_info["max_penetration_depth"] if penetration_info else 0.0,
                    "stability_score": stability_info["overall_stability"] if stability_info else 0.0,
                    "quality": stability_info["quality"] if stability_info else "unknown",
                }
            )

        # Store sliding events
        for sliding in sliding_events:
            sliding["foot"] = side
            sliding["bone_name"] = foot_bone.GetName()
            all_sliding.append(sliding)

        # Store frame-by-frame trajectories
        for frame in range(total_frames):
            height = positions[frame, 1] - ground_height
            velocity_mag = np.linalg.norm(velocities[frame])

            # Determine contact state
            in_contact = any(start <= frame <= end for start, end in contact_segments)

            all_trajectories.append(
                {
                    "foot": side,
                    "bone_name": foot_bone.GetName(),
                    "frame": frame,
                    "position_x": positions[frame, 0],
                    "position_y": positions[frame, 1],
                    "position_z": positions[frame, 2],
                    "height_above_ground": height,
                    "velocity": velocity_mag,
                    "in_contact": in_contact,
                }
            )

        # Compute summary statistics
        total_sliding_distance = sum(s["sliding_distance"] for s in sliding_events)
        mean_stability = np.mean([s["overall_stability"] for s in stability_scores]) if stability_scores else 0.0

        contact_summaries.append(
            {
                "foot": side,
                "bone_name": foot_bone.GetName(),
                "total_contacts": len(contact_segments),
                "contacts_with_sliding": len(sliding_events),
                "contacts_with_penetration": len(penetration_events),
                "total_sliding_distance": total_sliding_distance,
                "mean_stability_score": mean_stability,
                "overall_quality": (
                    "excellent"
                    if mean_stability > STABILITY_EXCELLENT
                    else (
                        "good"
                        if mean_stability > STABILITY_GOOD
                        else ("fair" if mean_stability > STABILITY_FAIR else "poor")
                    )
                ),
            }
        )

    # Write foot contacts CSV
    contacts_csv_path = os.path.join(output_dir, "foot_contacts.csv")
    with open(contacts_csv_path, "w", newline="") as f:
        if all_contacts:
            writer = csv.DictWriter(f, fieldnames=all_contacts[0].keys())
            writer.writeheader()
            writer.writerows(all_contacts)

    # Write foot sliding CSV
    sliding_csv_path = os.path.join(output_dir, "foot_sliding.csv")
    with open(sliding_csv_path, "w", newline="") as f:
        if all_sliding:
            writer = csv.DictWriter(f, fieldnames=all_sliding[0].keys())
            writer.writeheader()
            writer.writerows(all_sliding)

    # Write foot trajectories CSV
    trajectories_csv_path = os.path.join(output_dir, "foot_trajectories.csv")
    with open(trajectories_csv_path, "w", newline="") as f:
        if all_trajectories:
            writer = csv.DictWriter(f, fieldnames=all_trajectories[0].keys())
            writer.writeheader()
            writer.writerows(all_trajectories)

    # Write contact summary CSV
    summary_csv_path = os.path.join(output_dir, "contact_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        if contact_summaries:
            writer = csv.DictWriter(f, fieldnames=contact_summaries[0].keys())
            writer.writeheader()
            writer.writerows(contact_summaries)

    print(f"\n✓ Foot contact analysis complete:")
    print(f"  - {len(all_contacts)} contact events analyzed")
    print(f"  - {len(all_sliding)} sliding events detected")
    print(f"  - Ground height: {ground_height:.2f} units")

    # Return summary
    return {
        "ground_height": ground_height,
        "feet_detected": len([f for f in foot_bones.values() if f is not None]),
        "total_contacts": len(all_contacts),
        "contacts_with_sliding": len(all_sliding),
        "total_sliding_distance": sum(s["sliding_distance"] for s in all_sliding),
        "contact_summaries": contact_summaries,
    }
