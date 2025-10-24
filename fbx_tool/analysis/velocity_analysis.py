"""
Velocity, Acceleration, and Jitter Analysis Module

Analyzes motion quality at three levels:

1. PER-BONE: Individual joint analysis
   - TRANSLATIONAL: Velocity, acceleration, jerk (position-based)
   - ROTATIONAL: Angular velocity, angular acceleration, angular jerk (rotation-based)
   - Spike detection (velocity, acceleration, jerk)
   - Frozen frame detection
   - Directional (per-axis) jitter analysis

2. CHAIN-LEVEL: Kinematic chain analysis
   - Chain coherence (coordinated motion)
   - Velocity propagation along chain
   - Parent-child coordination

3. HOLISTIC: Whole-body analysis
   - Center-of-mass velocity
   - Total kinetic energy
   - Global smoothness score

Outputs:
- velocity_summary.csv: Per-bone translational velocity statistics
- angular_velocity_summary.csv: Per-bone rotational velocity statistics
- acceleration_peaks.csv: Sudden acceleration events
- jerk_spikes.csv: Sudden jerk events (NEW)
- jitter_analysis.csv: Per-bone jitter scores with directional breakdown
- jitter_temporal.csv: Frame-by-frame jitter values (NEW)
- smoothness_temporal.csv: Frame-by-frame smoothness values (NEW)
- velocity_spikes.csv: Flagged problematic frames
- chain_velocity.csv: Chain-level coordination metrics
- holistic_motion.csv: Whole-body motion quality
"""

import csv
import os

import numpy as np

from fbx_tool.analysis.utils import ensure_output_dir

# ==============================================================================
# CONSTANTS - Spike Detection
# ==============================================================================

# Statistical outlier detection thresholds (number of standard deviations)
VELOCITY_SPIKE_THRESHOLD_SIGMA = 3.0  # Velocity spikes: mean + 3σ
ACCELERATION_SPIKE_THRESHOLD_SIGMA = 2.5  # Acceleration spikes: mean + 2.5σ
JERK_SPIKE_THRESHOLD_SIGMA = 2.5  # Jerk spikes: mean + 2.5σ

# Minimum standard deviation to avoid false positives on constant data
SPIKE_DETECTION_MIN_STD = 1e-6

# ==============================================================================
# CONSTANTS - Jitter and Smoothness
# ==============================================================================

# Sliding window size for jitter computation (frames)
# Smaller window = detects high-frequency noise
# Larger window = measures sustained jitter
JITTER_WINDOW_SIZE = 5

# Smoothness score scaling factors
# Formula: 1 / (1 + jerk_magnitude * SCALE)
# Higher scale = more sensitive to jerk (lower scores for same jerk)
# These values empirically balance scores to 0-1 range for typical animation data
SMOOTHNESS_SCALE_TRANSLATIONAL = 0.1  # For position-based jerk (units/s³)
SMOOTHNESS_SCALE_ROTATIONAL = 0.01  # For rotation-based jerk (degrees/s³) - more sensitive due to larger typical values

# Frozen frame detection threshold (velocity magnitude)
# Joints moving slower than this are considered stationary
FROZEN_FRAME_VELOCITY_THRESHOLD = 0.001

# ==============================================================================
# CONSTANTS - Smoothing Recommendations
# ==============================================================================

# Jitter thresholds for smoothing intensity classification
JITTER_HIGH_THRESHOLD = 1.0  # Above this = severe jitter, needs aggressive smoothing
JITTER_MEDIUM_THRESHOLD = 0.1  # Above this = moderate jitter, needs medium smoothing

# Smoothing kernel sizes (must be odd for Savitzky-Golay filter)
SMOOTHING_KERNEL_HIGH = 7  # Aggressive smoothing for high jitter
SMOOTHING_KERNEL_MEDIUM = 5  # Moderate smoothing
SMOOTHING_KERNEL_LOW = 3  # Minimal smoothing

# Gaussian filter sigma values
GAUSSIAN_SIGMA_HIGH = 2.0  # Wide gaussian for aggressive smoothing
GAUSSIAN_SIGMA_MEDIUM = 1.0  # Moderate gaussian
GAUSSIAN_SIGMA_LOW = 0.5  # Narrow gaussian

# Low-pass filter cutoff frequencies (as fraction of frame rate)
CUTOFF_FRACTION_HIGH_JITTER = 0.1  # Filter out 90% of frequencies for high jitter
CUTOFF_FRACTION_MEDIUM_JITTER = 0.25  # Filter out 75% of frequencies
CUTOFF_FRACTION_LOW_JITTER = 0.4  # Filter out 60% of frequencies

# Butterworth filter orders
BUTTERWORTH_ORDER_HIGH = 4  # Higher order = sharper cutoff
BUTTERWORTH_ORDER_LOW = 2  # Lower order = gentler cutoff

# Savitzky-Golay polynomial orders
SAVGOL_POLYORDER_HIGH = 3  # Use cubic polynomial for large windows
SAVGOL_POLYORDER_LOW = 2  # Use quadratic polynomial for small windows

# ==============================================================================
# CONSTANTS - Quality Thresholds
# ==============================================================================

# ⚠️ DEPRECATED: Use compute_adaptive_coherence_thresholds() instead
# Chain coherence thresholds (correlation coefficient)
COHERENCE_GOOD_THRESHOLD = 0.7  # High correlation = coordinated motion
COHERENCE_FAIR_THRESHOLD = 0.4  # Medium correlation = acceptable motion

# Global smoothness quality thresholds
SMOOTHNESS_EXCELLENT_THRESHOLD = 0.8  # Very smooth motion
SMOOTHNESS_GOOD_THRESHOLD = 0.6  # Acceptable smoothness
SMOOTHNESS_FAIR_THRESHOLD = 0.4  # Marginal smoothness


# ==============================================================================
# ADAPTIVE THRESHOLD COMPUTATION (Proceduralization)
# ==============================================================================


def compute_adaptive_jitter_thresholds(jitter_scores):
    """
    Compute adaptive jitter classification thresholds from data distribution.

    PROCEDURAL DESIGN: Uses percentile-based classification instead of hardcoded
    thresholds (JITTER_HIGH_THRESHOLD = 1.0, JITTER_MEDIUM_THRESHOLD = 0.1).

    Strategy:
    - Low jitter: < 33rd percentile
    - Medium jitter: 33rd to 67th percentile
    - High jitter: > 67th percentile

    This adapts to the animation's overall noise level, making classification
    relative to the data rather than absolute.

    Args:
        jitter_scores: np.array of jitter scores from all bones

    Returns:
        dict: {
            'jitter_medium_threshold': float (33rd percentile),
            'jitter_high_threshold': float (67th percentile)
        }
    """
    if len(jitter_scores) == 0:
        # Fallback for empty data
        return {"jitter_medium_threshold": 0.1, "jitter_high_threshold": 1.0}  # Fallback to old constant

    if len(jitter_scores) == 1:
        # Single bone: use value itself as medium, scale up for high
        single_value = jitter_scores[0]
        return {
            "jitter_medium_threshold": single_value,
            "jitter_high_threshold": single_value * 2.0,  # 2x for high threshold
        }

    # Use percentiles for classification
    jitter_medium_threshold = np.percentile(jitter_scores, 33)
    jitter_high_threshold = np.percentile(jitter_scores, 67)

    # Ensure high > medium (can fail if data is constant)
    if jitter_high_threshold <= jitter_medium_threshold:
        # Add small offset to maintain ordering
        jitter_high_threshold = jitter_medium_threshold + 0.01

    return {"jitter_medium_threshold": jitter_medium_threshold, "jitter_high_threshold": jitter_high_threshold}


def compute_adaptive_coherence_thresholds(coherence_scores):
    """
    Compute adaptive coherence classification thresholds from data distribution.

    PROCEDURAL DESIGN: Uses percentile-based classification instead of hardcoded
    thresholds (COHERENCE_GOOD_THRESHOLD = 0.7, COHERENCE_FAIR_THRESHOLD = 0.4).

    Strategy:
    - Poor coherence: < 33rd percentile
    - Fair coherence: 33rd to 67th percentile
    - Good coherence: > 67th percentile

    This adapts to the animation's overall coordination level, making classification
    relative to the data rather than absolute.

    Args:
        coherence_scores: np.array of coherence scores (correlation coefficients) from all chains

    Returns:
        dict: {
            'coherence_fair_threshold': float (33rd percentile),
            'coherence_good_threshold': float (67th percentile)
        }
    """
    if len(coherence_scores) == 0:
        # Fallback for empty data
        return {"coherence_fair_threshold": 0.4, "coherence_good_threshold": 0.7}  # Fallback to old constant

    if len(coherence_scores) == 1:
        # Single chain: use value itself as fair, scale up for good
        single_value = coherence_scores[0]
        # Ensure we stay within valid correlation range [-1, 1]
        good_threshold = min(single_value + 0.2, 1.0)
        return {"coherence_fair_threshold": single_value, "coherence_good_threshold": good_threshold}

    # Use percentiles for classification
    coherence_fair_threshold = np.percentile(coherence_scores, 33)
    coherence_good_threshold = np.percentile(coherence_scores, 67)

    # Ensure good > fair (can fail if data is constant)
    if coherence_good_threshold <= coherence_fair_threshold:
        # Add small offset to maintain ordering
        coherence_good_threshold = min(coherence_fair_threshold + 0.1, 1.0)

    # Clamp to valid correlation range [-1, 1]
    coherence_fair_threshold = max(-1.0, min(1.0, coherence_fair_threshold))
    coherence_good_threshold = max(-1.0, min(1.0, coherence_good_threshold))

    return {"coherence_fair_threshold": coherence_fair_threshold, "coherence_good_threshold": coherence_good_threshold}


def compute_derivatives(positions, frame_rate):
    """
    Compute velocity, acceleration, and jerk from position data.

    Args:
        positions: np.array of shape (n_frames, 3) - position over time
        frame_rate: float - frames per second

    Returns:
        tuple: (velocity, acceleration, jerk) arrays
    """
    dt = 1.0 / frame_rate  # Time step

    # Velocity (1st derivative)
    velocity = np.gradient(positions, dt, axis=0)

    # Acceleration (2nd derivative)
    acceleration = np.gradient(velocity, dt, axis=0)

    # Jerk (3rd derivative) - smoothness indicator
    jerk = np.gradient(acceleration, dt, axis=0)

    return velocity, acceleration, jerk


def compute_magnitudes(vectors):
    """Compute magnitude for each vector in array."""
    return np.linalg.norm(vectors, axis=1)


def compute_angular_derivatives(rotations, frame_rate):
    """
    Compute angular velocity, angular acceleration, and angular jerk from rotation data.

    Args:
        rotations: np.array of shape (n_frames, 3) - Euler angles (X, Y, Z) in degrees over time
        frame_rate: float - frames per second

    Returns:
        tuple: (angular_velocity, angular_acceleration, angular_jerk) arrays in degrees/second
    """
    dt = 1.0 / frame_rate  # Time step

    # Handle angle wrapping (e.g., 359° -> 1° should be +2°, not -358°)
    unwrapped_rotations = np.copy(rotations)
    for axis in range(3):
        unwrapped_rotations[:, axis] = np.unwrap(np.radians(rotations[:, axis]))
        unwrapped_rotations[:, axis] = np.degrees(unwrapped_rotations[:, axis])

    # Angular velocity (1st derivative) - degrees/second
    angular_velocity = np.gradient(unwrapped_rotations, dt, axis=0)

    # Angular acceleration (2nd derivative) - degrees/second²
    angular_acceleration = np.gradient(angular_velocity, dt, axis=0)

    # Angular jerk (3rd derivative) - degrees/second³
    angular_jerk = np.gradient(angular_acceleration, dt, axis=0)

    return angular_velocity, angular_acceleration, angular_jerk


def compute_directional_jitter(values_xyz, window_size=5):
    """
    Compute per-axis jitter scores from 3D vector data.

    Args:
        values_xyz: np.array of shape (n_frames, 3) - 3D vectors over time
        window_size: Size of sliding window

    Returns:
        dict: {'x': jitter_x, 'y': jitter_y, 'z': jitter_z, 'magnitude': jitter_mag}
    """
    jitter_scores = {}

    # Compute jitter for each axis
    for axis, axis_name in enumerate(["x", "y", "z"]):
        jitter_scores[axis_name] = compute_jitter_score(values_xyz[:, axis], window_size)

    # Also compute jitter on magnitude
    magnitudes = compute_magnitudes(values_xyz)
    jitter_scores["magnitude"] = compute_jitter_score(magnitudes, window_size)

    return jitter_scores


def detect_spikes(values, threshold_multiplier, min_std=SPIKE_DETECTION_MIN_STD):
    """
    Detect spikes using statistical outlier detection.

    Formula: spike if value > mean + (threshold_multiplier × std)

    This identifies outliers by flagging values that deviate significantly
    from the mean. The threshold_multiplier determines sensitivity:
    - 3.0σ = ~99.7% of normal data excluded (very conservative)
    - 2.5σ = ~98.8% of normal data excluded (moderately conservative)

    Args:
        values: 1D array of scalar values
        threshold_multiplier: Number of standard deviations for outlier
        min_std: Minimum std to avoid false positives on constant data

    Returns:
        Array of frame indices where spikes occur
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    mean = np.mean(values)
    std = np.std(values)

    # Handle constant or near-constant values
    if std < min_std:
        return np.array([], dtype=int)  # No spikes in constant data

    threshold = mean + (threshold_multiplier * std)
    spike_indices = np.where(values > threshold)[0]
    return spike_indices


def compute_jitter_score(values, window_size=JITTER_WINDOW_SIZE):
    """
    Compute jitter score using high-frequency variation.

    Formula: mean(variance(sliding_window))

    Measures local variability by computing variance within a sliding window
    and averaging across all windows. This captures high-frequency noise
    that wouldn't be visible in global statistics.

    Higher scores indicate more jitter/noise. Scores are NOT normalized,
    so interpretation depends on the units of the input values.

    Args:
        values: 1D array of scalar values
        window_size: Size of sliding window (default: JITTER_WINDOW_SIZE)

    Returns:
        float: Jitter score (0 = perfectly smooth, higher = more jitter)
    """
    if len(values) < window_size:
        return 0.0

    # Compute local variance using sliding window
    local_variances = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        local_variances.append(np.var(window))

    # Jitter score is mean of local variances
    jitter_score = np.mean(local_variances)

    return jitter_score


def compute_smoothness_score(jerk_magnitude, scale_factor=SMOOTHNESS_SCALE_TRANSLATIONAL):
    """
    Compute smoothness score from jerk magnitude.

    Formula: 1 / (1 + mean_jerk × scale_factor)

    This is a monotonically decreasing function that maps jerk to smoothness:
    - jerk = 0 → smoothness = 1.0 (perfectly smooth)
    - jerk → ∞ → smoothness → 0.0 (infinitely jerky)

    The scale_factor controls sensitivity:
    - SMOOTHNESS_SCALE_TRANSLATIONAL (0.1): For position-based jerk (units/s³)
    - SMOOTHNESS_SCALE_ROTATIONAL (0.01): For rotation-based jerk (degrees/s³)
      Uses lower scale because rotational jerk values are typically larger

    The formula provides a soft, continuous mapping rather than hard thresholds,
    making scores comparable across different animation types and frame rates.

    Args:
        jerk_magnitude: Array of jerk magnitudes
        scale_factor: Scaling factor to normalize jerk range (default: translational)

    Returns:
        float: Smoothness score in range [0, 1], where higher = smoother
    """
    mean_jerk = np.mean(jerk_magnitude)

    # Avoid division by zero (perfect smoothness if no jerk)
    if mean_jerk == 0:
        return 1.0

    # Normalize using scaled inverse
    smoothness = 1.0 / (1.0 + mean_jerk * scale_factor)

    return smoothness


def detect_frozen_frames(velocity_magnitude, threshold=FROZEN_FRAME_VELOCITY_THRESHOLD):
    """
    Detect frames where joint is essentially stationary.

    Identifies consecutive frames where velocity falls below threshold,
    indicating potential stuck/frozen animation or deliberate pauses.

    Args:
        velocity_magnitude: Array of velocity magnitudes
        threshold: Velocity threshold for "frozen" classification (default: FROZEN_FRAME_VELOCITY_THRESHOLD)

    Returns:
        List of (start_frame, end_frame) tuples for frozen segments
    """
    frozen_mask = velocity_magnitude < threshold

    # Find consecutive frozen frames
    frozen_segments = []
    in_frozen = False
    start_frame = 0

    for i, is_frozen in enumerate(frozen_mask):
        if is_frozen and not in_frozen:
            start_frame = i
            in_frozen = True
        elif not is_frozen and in_frozen:
            frozen_segments.append((start_frame, i - 1))
            in_frozen = False

    # Handle case where animation ends while frozen
    if in_frozen:
        frozen_segments.append((start_frame, len(frozen_mask) - 1))

    return frozen_segments


def compute_smoothing_parameters(jitter_score, smoothness_score, frame_rate, jitter_thresholds=None):
    """
    Compute recommended smoothing filter parameters based on jitter and smoothness.

    Provides filter configuration for common smoothing techniques:
    - Gaussian blur (kernel_size, gaussian_sigma)
    - Butterworth low-pass filter (cutoff_frequency_hz, butterworth_order)
    - Savitzky-Golay filter (savgol_window, savgol_polyorder)

    Recommendations are tiered based on jitter severity:
    - HIGH (jitter > adaptive threshold): Aggressive smoothing
    - MEDIUM (jitter > adaptive threshold): Moderate smoothing
    - LOW/NONE: Minimal smoothing

    Args:
        jitter_score: Jitter score from compute_jitter_score
        smoothness_score: Smoothness score from compute_smoothness_score (currently unused)
        frame_rate: Animation frame rate
        jitter_thresholds: Optional dict from compute_adaptive_jitter_thresholds()
                           If None, uses deprecated constants

    Returns:
        dict: Smoothing recommendations with filter parameters
    """
    # Use adaptive thresholds if provided, otherwise fall back to constants
    if jitter_thresholds is not None:
        jitter_high_threshold = jitter_thresholds["jitter_high_threshold"]
        jitter_medium_threshold = jitter_thresholds["jitter_medium_threshold"]
    else:
        jitter_high_threshold = JITTER_HIGH_THRESHOLD
        jitter_medium_threshold = JITTER_MEDIUM_THRESHOLD

    # Determine smoothing intensity based on jitter thresholds
    if jitter_score > jitter_high_threshold:
        intensity = "high"
        kernel_size = SMOOTHING_KERNEL_HIGH
        gaussian_sigma = GAUSSIAN_SIGMA_HIGH
        cutoff_fraction = CUTOFF_FRACTION_HIGH_JITTER
        filter_order = BUTTERWORTH_ORDER_HIGH
    elif jitter_score > jitter_medium_threshold:
        intensity = "medium"
        kernel_size = SMOOTHING_KERNEL_MEDIUM
        gaussian_sigma = GAUSSIAN_SIGMA_MEDIUM
        cutoff_fraction = CUTOFF_FRACTION_MEDIUM_JITTER
        filter_order = BUTTERWORTH_ORDER_LOW
    else:
        intensity = "none"
        kernel_size = SMOOTHING_KERNEL_LOW
        gaussian_sigma = GAUSSIAN_SIGMA_LOW
        cutoff_fraction = CUTOFF_FRACTION_LOW_JITTER
        filter_order = BUTTERWORTH_ORDER_LOW

    # Cutoff frequency for low-pass filter (Hz)
    # Higher jitter = lower cutoff frequency (more aggressive filtering)
    cutoff_frequency = frame_rate * cutoff_fraction

    # Savitzky-Golay polynomial order (higher for larger windows)
    savgol_polyorder = SAVGOL_POLYORDER_HIGH if kernel_size >= 5 else SAVGOL_POLYORDER_LOW

    return {
        "intensity": intensity,
        "kernel_size": kernel_size,
        "gaussian_sigma": gaussian_sigma,
        "cutoff_frequency_hz": cutoff_frequency,
        "butterworth_order": filter_order,
        "savgol_window": kernel_size,
        "savgol_polyorder": savgol_polyorder,
    }


def analyze_velocity(scene, output_dir="output/"):
    """
    Comprehensive velocity, acceleration, and jitter analysis.

    Args:
        scene: FBX scene object
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary statistics and analysis results
    """
    ensure_output_dir(output_dir)

    # Get scene metadata
    from fbx_tool.analysis.fbx_loader import get_scene_metadata

    metadata = get_scene_metadata(scene)

    if not metadata.get("has_animation", False):
        raise ValueError("No animation data found in scene")

    start_time = metadata["start_time"]
    stop_time = metadata["stop_time"]
    frame_rate = metadata["frame_rate"]
    duration = stop_time - start_time
    total_frames = int(duration * frame_rate) + 1  # +1 to include both start and end frames

    # Get root node
    root_node = scene.GetRootNode()

    # Collect all bones
    bones = []

    def collect_bones(node):
        if node.GetNodeAttribute():
            attr_type = node.GetNodeAttribute().GetAttributeType()
            # Check if node is a skeleton bone
            import fbx as fbx_module

            if attr_type == fbx_module.FbxNodeAttribute.EType.eSkeleton:
                bones.append(node)
        for i in range(node.GetChildCount()):
            collect_bones(node.GetChild(i))

    collect_bones(root_node)

    if not bones:
        raise ValueError("No bones found in scene")

    print(f"Analyzing velocity for {len(bones)} bones across {total_frames} frames...")

    # Results storage
    velocity_summary = []
    angular_velocity_summary = []
    acceleration_peaks = []
    jitter_analysis = []
    velocity_spikes = []
    jerk_spikes = []  # NEW: Track jerk spikes
    jitter_temporal = []  # NEW: Frame-by-frame jitter
    smoothness_temporal = []  # NEW: Frame-by-frame smoothness

    # Time span for animation
    import fbx as fbx_module

    # Get the current animation stack's time span
    anim_stack_count = scene.GetSrcObjectCount(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId))
    if anim_stack_count > 0:
        anim_stack = scene.GetSrcObject(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId), 0)
        time_span = anim_stack.GetLocalTimeSpan()
    else:
        raise ValueError("No animation stack found")

    start_time = time_span.GetStart()
    frame_duration = fbx_module.FbxTime()
    frame_duration.SetSecondDouble(1.0 / frame_rate)

    # ===== PASS 1: Collect jitter scores for adaptive threshold computation =====
    print("Pass 1: Computing jitter scores for adaptive thresholds...")
    all_jitter_scores = []

    for bone in bones:
        # Extract positions
        positions = []
        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame
            translation = bone.EvaluateGlobalTransform(current_time).GetT()
            positions.append([translation[0], translation[1], translation[2]])

        positions = np.array(positions)

        # Compute velocity
        velocity, _, _ = compute_derivatives(positions, frame_rate)
        velocity_mag = compute_magnitudes(velocity)

        # Compute jitter score
        jitter_score = compute_jitter_score(velocity_mag)
        all_jitter_scores.append(jitter_score)

    # Compute adaptive jitter thresholds from all bones
    jitter_thresholds = compute_adaptive_jitter_thresholds(np.array(all_jitter_scores))
    print(
        f"  Adaptive jitter thresholds: medium={jitter_thresholds['jitter_medium_threshold']:.4f}, high={jitter_thresholds['jitter_high_threshold']:.4f}"
    )

    # ===== PASS 2: Full analysis with adaptive thresholds =====
    print("Pass 2: Performing full velocity analysis...")

    # Process each bone
    for bone_idx, bone in enumerate(bones):
        bone_name = bone.GetName()

        # Extract POSITION and ROTATION data across all frames
        positions = []
        rotations = []

        for frame in range(total_frames):
            current_time = start_time + frame_duration * frame

            # Get global transform
            global_transform = bone.EvaluateGlobalTransform(current_time)

            # Get translation
            translation = global_transform.GetT()
            positions.append([translation[0], translation[1], translation[2]])

            # Get rotation (Euler angles in degrees)
            rotation = global_transform.GetR()
            rotations.append([rotation[0], rotation[1], rotation[2]])

        positions = np.array(positions)
        rotations = np.array(rotations)

        # ===== TRANSLATIONAL MOTION ANALYSIS =====
        # Compute derivatives
        velocity, acceleration, jerk = compute_derivatives(positions, frame_rate)

        # Compute magnitudes
        velocity_mag = compute_magnitudes(velocity)
        acceleration_mag = compute_magnitudes(acceleration)
        jerk_mag = compute_magnitudes(jerk)

        # Statistics
        mean_velocity = np.mean(velocity_mag)
        max_velocity = np.max(velocity_mag)
        mean_acceleration = np.mean(acceleration_mag)
        max_acceleration = np.max(acceleration_mag)
        mean_jerk = np.mean(jerk_mag)
        max_jerk = np.max(jerk_mag)

        # Jitter score (magnitude-based)
        jitter_score = compute_jitter_score(velocity_mag)

        # ENHANCEMENT B: Directional jitter analysis
        directional_jitter = compute_directional_jitter(velocity)

        # Smoothness score
        smoothness_score = compute_smoothness_score(jerk_mag)

        # Detect frozen frames
        frozen_segments = detect_frozen_frames(velocity_mag)
        frozen_frame_count = sum(end - start + 1 for start, end in frozen_segments)
        frozen_percentage = (frozen_frame_count / total_frames) * 100 if total_frames > 0 else 0

        # ENHANCEMENT D: Smoothing parameter recommendations (with adaptive thresholds)
        smoothing_params = compute_smoothing_parameters(jitter_score, smoothness_score, frame_rate, jitter_thresholds)

        # ===== ROTATIONAL MOTION ANALYSIS =====
        # Compute angular derivatives
        angular_velocity, angular_acceleration, angular_jerk = compute_angular_derivatives(rotations, frame_rate)

        # Compute angular magnitudes
        angular_velocity_mag = compute_magnitudes(angular_velocity)
        angular_acceleration_mag = compute_magnitudes(angular_acceleration)
        angular_jerk_mag = compute_magnitudes(angular_jerk)

        # Angular statistics
        mean_angular_velocity = np.mean(angular_velocity_mag)
        max_angular_velocity = np.max(angular_velocity_mag)
        mean_angular_acceleration = np.mean(angular_acceleration_mag)
        max_angular_acceleration = np.max(angular_acceleration_mag)
        mean_angular_jerk = np.mean(angular_jerk_mag)
        max_angular_jerk = np.max(angular_jerk_mag)

        # Angular jitter and smoothness
        angular_jitter_score = compute_jitter_score(angular_velocity_mag)
        angular_smoothness_score = compute_smoothness_score(angular_jerk_mag, scale_factor=SMOOTHNESS_SCALE_ROTATIONAL)

        # Directional angular jitter
        directional_angular_jitter = compute_directional_jitter(angular_velocity)

        # Translational velocity summary
        velocity_summary.append(
            {
                "bone_name": bone_name,
                "mean_velocity": mean_velocity,
                "max_velocity": max_velocity,
                "mean_acceleration": mean_acceleration,
                "max_acceleration": max_acceleration,
                "mean_jerk": mean_jerk,
                "max_jerk": max_jerk,
                "smoothness_score": smoothness_score,
                "frozen_frames": frozen_frame_count,
                "frozen_percentage": frozen_percentage,
            }
        )

        # Angular velocity summary (NEW)
        angular_velocity_summary.append(
            {
                "bone_name": bone_name,
                "mean_angular_velocity": mean_angular_velocity,
                "max_angular_velocity": max_angular_velocity,
                "mean_angular_acceleration": mean_angular_acceleration,
                "max_angular_acceleration": max_angular_acceleration,
                "mean_angular_jerk": mean_angular_jerk,
                "max_angular_jerk": max_angular_jerk,
                "angular_smoothness_score": angular_smoothness_score,
                "angular_jitter_score": angular_jitter_score,
            }
        )

        # Detect velocity spikes
        velocity_spike_frames = detect_spikes(velocity_mag, VELOCITY_SPIKE_THRESHOLD_SIGMA)
        for frame_idx in velocity_spike_frames:
            velocity_spikes.append(
                {
                    "bone_name": bone_name,
                    "frame": int(frame_idx),
                    "velocity": velocity_mag[frame_idx],
                    "threshold": np.mean(velocity_mag) + VELOCITY_SPIKE_THRESHOLD_SIGMA * np.std(velocity_mag),
                }
            )

        # Detect acceleration peaks
        accel_spike_frames = detect_spikes(acceleration_mag, ACCELERATION_SPIKE_THRESHOLD_SIGMA)
        for frame_idx in accel_spike_frames:
            acceleration_peaks.append(
                {
                    "bone_name": bone_name,
                    "frame": int(frame_idx),
                    "acceleration": acceleration_mag[frame_idx],
                    "severity": acceleration_mag[frame_idx] / mean_acceleration if mean_acceleration > 0 else 0,
                }
            )

        # ENHANCEMENT C: Detect jerk spikes (translational)
        jerk_spike_frames = detect_spikes(jerk_mag, JERK_SPIKE_THRESHOLD_SIGMA)
        for frame_idx in jerk_spike_frames:
            jerk_spikes.append(
                {
                    "bone_name": bone_name,
                    "frame": int(frame_idx),
                    "jerk": jerk_mag[frame_idx],
                    "type": "translational",
                    "severity": jerk_mag[frame_idx] / mean_jerk if mean_jerk > 0 else 0,
                }
            )

        # ENHANCEMENT C: Detect angular jerk spikes
        angular_jerk_spike_frames = detect_spikes(angular_jerk_mag, JERK_SPIKE_THRESHOLD_SIGMA)
        for frame_idx in angular_jerk_spike_frames:
            jerk_spikes.append(
                {
                    "bone_name": bone_name,
                    "frame": int(frame_idx),
                    "jerk": angular_jerk_mag[frame_idx],
                    "type": "rotational",
                    "severity": angular_jerk_mag[frame_idx] / mean_angular_jerk if mean_angular_jerk > 0 else 0,
                }
            )

        # ENHANCEMENT B: Enhanced jitter analysis with directional breakdown
        jitter_analysis.append(
            {
                "bone_name": bone_name,
                "jitter_score": jitter_score,
                "jitter_x": directional_jitter["x"],
                "jitter_y": directional_jitter["y"],
                "jitter_z": directional_jitter["z"],
                "angular_jitter_score": angular_jitter_score,
                "angular_jitter_x": directional_angular_jitter["x"],
                "angular_jitter_y": directional_angular_jitter["y"],
                "angular_jitter_z": directional_angular_jitter["z"],
                "smoothness_score": smoothness_score,
                "angular_smoothness_score": angular_smoothness_score,
                "recommended_smoothing": smoothing_params["intensity"],
                "gaussian_sigma": smoothing_params["gaussian_sigma"],
                "cutoff_frequency_hz": smoothing_params["cutoff_frequency_hz"],
                "savgol_window": smoothing_params["savgol_window"],
            }
        )

        # ENHANCEMENT A: Frame-by-frame temporal data
        for frame in range(total_frames):
            jitter_temporal.append(
                {
                    "bone_name": bone_name,
                    "frame": frame,
                    "velocity": velocity_mag[frame],
                    "acceleration": acceleration_mag[frame],
                    "jerk": jerk_mag[frame],
                    "angular_velocity": angular_velocity_mag[frame],
                    "angular_acceleration": angular_acceleration_mag[frame],
                    "angular_jerk": angular_jerk_mag[frame],
                }
            )

            smoothness_temporal.append(
                {
                    "bone_name": bone_name,
                    "frame": frame,
                    # ✅ FIXED: Use constants to match compute_smoothness_score() formula
                    "smoothness": 1.0 / (1.0 + jerk_mag[frame] * SMOOTHNESS_SCALE_TRANSLATIONAL),
                    "angular_smoothness": 1.0 / (1.0 + angular_jerk_mag[frame] * SMOOTHNESS_SCALE_ROTATIONAL),
                }
            )

    # Write translational velocity summary CSV
    velocity_csv_path = os.path.join(output_dir, "velocity_summary.csv")
    with open(velocity_csv_path, "w", newline="") as f:
        if velocity_summary:
            writer = csv.DictWriter(f, fieldnames=velocity_summary[0].keys())
            writer.writeheader()
            writer.writerows(velocity_summary)

    # Write angular velocity summary CSV (NEW)
    angular_velocity_csv_path = os.path.join(output_dir, "angular_velocity_summary.csv")
    with open(angular_velocity_csv_path, "w", newline="") as f:
        if angular_velocity_summary:
            writer = csv.DictWriter(f, fieldnames=angular_velocity_summary[0].keys())
            writer.writeheader()
            writer.writerows(angular_velocity_summary)

    # Write acceleration peaks CSV
    accel_csv_path = os.path.join(output_dir, "acceleration_peaks.csv")
    with open(accel_csv_path, "w", newline="") as f:
        if acceleration_peaks:
            writer = csv.DictWriter(f, fieldnames=acceleration_peaks[0].keys())
            writer.writeheader()
            writer.writerows(acceleration_peaks)

    # Write jerk spikes CSV (ENHANCEMENT C)
    jerk_spikes_csv_path = os.path.join(output_dir, "jerk_spikes.csv")
    with open(jerk_spikes_csv_path, "w", newline="") as f:
        if jerk_spikes:
            writer = csv.DictWriter(f, fieldnames=jerk_spikes[0].keys())
            writer.writeheader()
            writer.writerows(jerk_spikes)

    # Write enhanced jitter analysis CSV (ENHANCEMENT B + D)
    jitter_csv_path = os.path.join(output_dir, "jitter_analysis.csv")
    with open(jitter_csv_path, "w", newline="") as f:
        if jitter_analysis:
            writer = csv.DictWriter(f, fieldnames=jitter_analysis[0].keys())
            writer.writeheader()
            writer.writerows(jitter_analysis)

    # Write jitter temporal CSV (ENHANCEMENT A)
    jitter_temporal_csv_path = os.path.join(output_dir, "jitter_temporal.csv")
    with open(jitter_temporal_csv_path, "w", newline="") as f:
        if jitter_temporal:
            writer = csv.DictWriter(f, fieldnames=jitter_temporal[0].keys())
            writer.writeheader()
            writer.writerows(jitter_temporal)

    # Write smoothness temporal CSV (ENHANCEMENT A)
    smoothness_temporal_csv_path = os.path.join(output_dir, "smoothness_temporal.csv")
    with open(smoothness_temporal_csv_path, "w", newline="") as f:
        if smoothness_temporal:
            writer = csv.DictWriter(f, fieldnames=smoothness_temporal[0].keys())
            writer.writeheader()
            writer.writerows(smoothness_temporal)

    # Write velocity spikes CSV
    spikes_csv_path = os.path.join(output_dir, "velocity_spikes.csv")
    with open(spikes_csv_path, "w", newline="") as f:
        if velocity_spikes:
            writer = csv.DictWriter(f, fieldnames=velocity_spikes[0].keys())
            writer.writeheader()
            writer.writerows(velocity_spikes)

    # CHAIN-LEVEL ANALYSIS
    print("Computing chain-level metrics...")
    chain_velocity_data = analyze_chain_velocity(scene, bones, frame_rate, total_frames, output_dir)

    # HOLISTIC ANALYSIS
    print("Computing holistic motion metrics...")
    holistic_data = analyze_holistic_motion(
        bones, frame_rate, total_frames, output_dir, scene, start_time, frame_duration
    )

    print(f"✓ Velocity analysis complete:")
    print(f"  - {len(velocity_summary)} bones analyzed")
    print(f"  - {len(velocity_spikes)} velocity spikes detected")
    print(f"  - {len(acceleration_peaks)} acceleration peaks detected")
    print(f"  - {len(jerk_spikes)} jerk spikes detected")
    print(f"  - {len(chain_velocity_data)} chains analyzed")
    print(f"  - {len(jitter_temporal)} temporal jitter data points")

    # Count high jitter bones (translational or rotational)
    high_jitter_bones = sum(1 for j in jitter_analysis if j["recommended_smoothing"] == "high")

    # Return summary
    return {
        "total_bones": len(bones),
        "total_frames": total_frames,
        "velocity_spikes_count": len(velocity_spikes),
        "acceleration_peaks_count": len(acceleration_peaks),
        "jerk_spikes_count": len(jerk_spikes),
        "high_jitter_bones": high_jitter_bones,
        "chains_analyzed": len(chain_velocity_data),
        "holistic_metrics": holistic_data,
        "temporal_data_points": len(jitter_temporal),
    }


def analyze_chain_velocity(scene, bones, frame_rate, total_frames, output_dir):
    """
    Analyze velocity coordination along kinematic chains.

    Detects:
    - Chain coherence (smooth motion propagation)
    - Parent-child velocity correlation
    - Wave propagation quality

    Args:
        scene: FBX scene
        bones: List of bone nodes
        frame_rate: Animation frame rate
        total_frames: Total number of frames
        output_dir: Output directory

    Returns:
        list: Chain velocity analysis results
    """
    from fbx_tool.analysis.utils import build_bone_hierarchy, detect_chains_from_hierarchy

    # Build bone hierarchy and detect chains
    hierarchy = build_bone_hierarchy(scene)
    chain_names_map = detect_chains_from_hierarchy(hierarchy, min_chain_length=2)

    # Map bone names to bone nodes
    bone_dict = {bone.GetName(): bone for bone in bones}
    chains = {}
    for chain_name, bone_names in chain_names_map.items():
        chain_bones = [bone_dict[name] for name in bone_names if name in bone_dict]
        if chain_bones:
            chains[chain_name] = chain_bones

    chain_results = []

    # Time setup
    import fbx as fbx_module

    # Get the current animation stack's time span
    anim_stack_count = scene.GetSrcObjectCount(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId))
    if anim_stack_count > 0:
        anim_stack = scene.GetSrcObject(fbx_module.FbxCriteria.ObjectType(fbx_module.FbxAnimStack.ClassId), 0)
        time_span = anim_stack.GetLocalTimeSpan()
    else:
        raise ValueError("No animation stack found")

    start_time = time_span.GetStart()
    frame_duration = fbx_module.FbxTime()
    frame_duration.SetSecondDouble(1.0 / frame_rate)

    # ===== PASS 1: Collect coherence scores for adaptive threshold computation =====
    all_mean_coherence_scores = []
    chain_data_cache = {}  # Store computed data for reuse in pass 2

    for chain_name, chain_bones in chains.items():
        if len(chain_bones) < 2:
            continue  # Need at least 2 bones for chain analysis

        # Collect velocities for all bones in chain
        chain_velocities = []

        for bone in chain_bones:
            positions = []
            for frame in range(total_frames):
                current_time = start_time + frame_duration * frame
                translation = bone.EvaluateGlobalTransform(current_time).GetT()
                positions.append([translation[0], translation[1], translation[2]])

            positions = np.array(positions)
            velocity, _, _ = compute_derivatives(positions, frame_rate)
            velocity_mag = compute_magnitudes(velocity)
            chain_velocities.append(velocity_mag)

        chain_velocities = np.array(chain_velocities)  # Shape: (n_bones, n_frames)

        # Compute chain coherence: correlation between adjacent bones
        # ✅ FIXED: Filter NaN values to avoid propagation
        coherence_scores = []
        for i in range(len(chain_bones) - 1):
            correlation = np.corrcoef(chain_velocities[i], chain_velocities[i + 1])[0, 1]
            if np.isfinite(correlation):
                coherence_scores.append(correlation)

        if not coherence_scores:
            mean_coherence = 0.0
            print(f"⚠ Chain {chain_name}: No valid coherence scores - using 0.0")
        else:
            mean_coherence = np.mean(coherence_scores)

        # Compute velocity propagation delay (phase shift between parent and child)
        # Measure using cross-correlation
        propagation_delays = []
        for i in range(len(chain_bones) - 1):
            cross_corr = np.correlate(chain_velocities[i], chain_velocities[i + 1], mode="same")
            delay_frame = np.argmax(cross_corr) - len(cross_corr) // 2
            propagation_delays.append(abs(delay_frame))

        mean_delay = np.mean(propagation_delays) if propagation_delays else 0.0

        # Chain smoothness: average jitter across chain
        chain_jitter_scores = [compute_jitter_score(vel) for vel in chain_velocities]
        mean_chain_jitter = np.mean(chain_jitter_scores)

        # Cache data for pass 2
        chain_data_cache[chain_name] = {
            "bone_count": len(chain_bones),
            "coherence_score": mean_coherence,
            "propagation_delay_frames": mean_delay,
            "chain_jitter_score": mean_chain_jitter,
        }

        # Collect coherence score
        all_mean_coherence_scores.append(mean_coherence)

    # ===== Compute adaptive coherence thresholds =====
    if all_mean_coherence_scores:
        coherence_thresholds = compute_adaptive_coherence_thresholds(np.array(all_mean_coherence_scores))
        coherence_good_threshold = coherence_thresholds["coherence_good_threshold"]
        coherence_fair_threshold = coherence_thresholds["coherence_fair_threshold"]
        print(
            f"  Adaptive coherence thresholds: fair={coherence_fair_threshold:.4f}, good={coherence_good_threshold:.4f}"
        )
    else:
        # No chains - use deprecated constants
        coherence_good_threshold = COHERENCE_GOOD_THRESHOLD
        coherence_fair_threshold = COHERENCE_FAIR_THRESHOLD

    # ===== PASS 2: Build results with adaptive classification =====
    for chain_name, chain_data in chain_data_cache.items():
        mean_coherence = chain_data["coherence_score"]

        # Use adaptive thresholds for classification
        coordination_quality = (
            "good"
            if mean_coherence > coherence_good_threshold
            else ("fair" if mean_coherence > coherence_fair_threshold else "poor")
        )

        chain_results.append(
            {
                "chain_name": chain_name,
                "bone_count": chain_data["bone_count"],
                "coherence_score": mean_coherence,
                "propagation_delay_frames": chain_data["propagation_delay_frames"],
                "chain_jitter_score": chain_data["chain_jitter_score"],
                "coordination_quality": coordination_quality,
            }
        )

    # Write chain velocity CSV
    if chain_results:
        chain_csv_path = os.path.join(output_dir, "chain_velocity.csv")
        with open(chain_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=chain_results[0].keys())
            writer.writeheader()
            writer.writerows(chain_results)

    return chain_results


def analyze_holistic_motion(bones, frame_rate, total_frames, output_dir, scene, start_time, frame_duration):
    """
    Analyze whole-body motion quality metrics.

    Computes:
    - Center-of-mass velocity
    - Total kinetic energy
    - Global smoothness
    - Overall motion coherence

    Args:
        bones: List of bone nodes
        frame_rate: Animation frame rate
        total_frames: Total number of frames
        output_dir: Output directory
        scene: FBX scene
        start_time: Start time
        frame_duration: Frame duration

    Returns:
        dict: Holistic motion metrics
    """
    # Compute center of mass position over time
    com_positions = []

    for frame in range(total_frames):
        current_time = start_time + frame_duration * frame

        # Average position of all bones (simplified COM)
        positions = []
        for bone in bones:
            translation = bone.EvaluateGlobalTransform(current_time).GetT()
            positions.append([translation[0], translation[1], translation[2]])

        # Center of mass as average (in reality would be weighted by mass)
        com = np.mean(positions, axis=0)
        com_positions.append(com)

    com_positions = np.array(com_positions)

    # Compute COM velocity, acceleration, jerk
    com_velocity, com_acceleration, com_jerk = compute_derivatives(com_positions, frame_rate)

    com_velocity_mag = compute_magnitudes(com_velocity)
    com_acceleration_mag = compute_magnitudes(com_acceleration)
    com_jerk_mag = compute_magnitudes(com_jerk)

    # Global smoothness score
    global_smoothness = compute_smoothness_score(com_jerk_mag)

    # Global jitter score
    global_jitter = compute_jitter_score(com_velocity_mag)

    # Total kinetic energy (simplified, assuming unit mass)
    # KE = 0.5 * m * v^2, using m=1 for relative comparison
    kinetic_energy = 0.5 * (com_velocity_mag**2)
    mean_kinetic_energy = np.mean(kinetic_energy)
    max_kinetic_energy = np.max(kinetic_energy)

    # Energy variation (how much energy changes - should be smooth)
    energy_variation = np.std(kinetic_energy)

    # Write holistic motion CSV (one row with overall metrics)
    holistic_results = [
        {
            "mean_com_velocity": np.mean(com_velocity_mag),
            "max_com_velocity": np.max(com_velocity_mag),
            "mean_com_acceleration": np.mean(com_acceleration_mag),
            "max_com_acceleration": np.max(com_acceleration_mag),
            "global_smoothness_score": global_smoothness,
            "global_jitter_score": global_jitter,
            "mean_kinetic_energy": mean_kinetic_energy,
            "max_kinetic_energy": max_kinetic_energy,
            "energy_variation": energy_variation,
            "overall_quality": (
                "excellent"
                if global_smoothness > SMOOTHNESS_EXCELLENT_THRESHOLD
                else (
                    "good"
                    if global_smoothness > SMOOTHNESS_GOOD_THRESHOLD
                    else ("fair" if global_smoothness > SMOOTHNESS_FAIR_THRESHOLD else "poor")
                )
            ),
        }
    ]

    holistic_csv_path = os.path.join(output_dir, "holistic_motion.csv")
    with open(holistic_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=holistic_results[0].keys())
        writer.writeheader()
        writer.writerows(holistic_results)

    return holistic_results[0]
