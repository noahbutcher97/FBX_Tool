"""
Motion Transition Detection Module

Detects transitions between different types of motion (walk, run, idle, etc.).

While directional changes track WHERE the character is moving,
motion transitions track HOW the character is moving - changes in
locomotion style, speed, and activity type.

Key transition types:
- Idle → Walking (start moving)
- Walking → Running (speed increase)
- Running → Walking (speed decrease)
- Walking/Running → Idle (stop moving)
- Idle → Jumping (explosive movement)
- Landing → Idle/Walking (impact absorption)

Detection uses multi-modal analysis:
- Velocity magnitude changes
- Acceleration patterns
- Gait cycle rate changes (from gait_analysis)
- Foot contact patterns

This enables descriptions like:
"character stands idle, begins walking forward, accelerates into a run,
decelerates back to a walk, and comes to a stop"

Outputs:
- motion_transitions.csv: Detected transitions with timing and classification
- motion_states.csv: Continuous motion state segments
- transition_quality.csv: Smoothness metrics for each transition
"""

import csv
import os

import numpy as np

from fbx_tool.analysis.utils import ensure_output_dir, extract_root_trajectory

# ==============================================================================
# CONSTANTS - Motion State Classification
# ==============================================================================

# Velocity thresholds for motion state classification (units/second)
VELOCITY_IDLE_THRESHOLD = 5.0  # Below this = idle/stationary
VELOCITY_WALK_THRESHOLD = 50.0  # Below this = walking
VELOCITY_RUN_THRESHOLD = 150.0  # Below this = running
# Above RUN_THRESHOLD = sprinting

# Acceleration thresholds for explosive movements (units/second²)
ACCELERATION_JUMP_THRESHOLD = 200.0  # Sudden upward acceleration
ACCELERATION_LAND_THRESHOLD = -200.0  # Sudden downward deceleration

# Vertical velocity thresholds for aerial detection (units/second)
VERTICAL_VELOCITY_AIRBORNE_THRESHOLD = 10.0  # Positive Y velocity = jumping/falling

# ==============================================================================
# CONSTANTS - Transition Detection
# ==============================================================================

# NOTE: These constants are intended as TIME-BASED values, not frame counts.
# They should be converted to frames based on actual frame rate.

# State stability requirements (in seconds)
STATE_STABLE_DURATION_SECONDS = 0.15  # 150ms required to confirm state change

# Minimum duration for valid motion state segment (in seconds)
STATE_MIN_DURATION_SECONDS = 0.3  # 300ms minimum duration

# Transition smoothness classification (based on jerk magnitude)
# These are scale-invariant and will be computed adaptively
TRANSITION_JERK_SMOOTH = 50.0  # Below this = smooth transition
TRANSITION_JERK_MODERATE = 150.0  # Below this = moderate transition
# Above this = abrupt transition


def compute_frame_aware_constants(frame_rate: float) -> dict:
    """
    Compute frame-rate aware constants for temporal thresholds.

    PROCEDURAL: Converts time-based constants to frame counts based on actual frame rate.
    This ensures consistent behavior across animations with different frame rates.

    Args:
        frame_rate: Animation frame rate (frames per second)

    Returns:
        dict: Frame-aware constants
    """
    return {
        "state_stable_frames": max(1, int(STATE_STABLE_DURATION_SECONDS * frame_rate)),
        "state_min_duration_frames": max(1, int(STATE_MIN_DURATION_SECONDS * frame_rate)),
    }


# ==============================================================================
# ADAPTIVE THRESHOLD CALCULATION
# ==============================================================================


def calculate_adaptive_velocity_thresholds(velocity_magnitudes):
    """
    Calculate adaptive velocity thresholds for motion state classification.

    Uses percentile-based approach to find natural boundaries between
    idle, walking, running, and sprinting states.

    This makes the classification scale-invariant and works across
    different character sizes, unit systems, and animation styles.

    Args:
        velocity_magnitudes: Array of velocity magnitudes across all frames

    Returns:
        dict: Adaptive thresholds for each state boundary
    """
    # Remove NaN/inf values
    valid_velocities = velocity_magnitudes[np.isfinite(velocity_magnitudes)]

    if len(valid_velocities) == 0:
        # Fallback to hardcoded thresholds
        return {"idle": VELOCITY_IDLE_THRESHOLD, "walk": VELOCITY_WALK_THRESHOLD, "run": VELOCITY_RUN_THRESHOLD}

    # Handle edge case: very few samples (< 10 frames)
    if len(valid_velocities) < 10:
        # Use simpler logic for small datasets
        min_vel = np.min(valid_velocities)
        max_vel = np.max(valid_velocities)
        range_vel = max_vel - min_vel

        # If range is very small, all same state
        if range_vel < 1.0:
            # Determine which state based on absolute magnitude
            if max_vel < 10.0:
                # All idle
                return {"idle": max_vel + 1.0, "walk": max_vel + 2.0, "run": max_vel + 3.0}
            elif max_vel < 75.0:
                # All walking
                return {"idle": min_vel - 1.0 if min_vel > 1.0 else 1.0, "walk": max_vel + 1.0, "run": max_vel + 2.0}
            else:
                # Running/sprinting
                return {"idle": min_vel * 0.1, "walk": min_vel * 0.5, "run": max_vel * 0.8}

        # Divide range into thirds
        idle_threshold = min_vel + range_vel * 0.25
        walk_threshold = min_vel + range_vel * 0.6
        run_threshold = min_vel + range_vel * 0.85

        return {"idle": idle_threshold, "walk": walk_threshold, "run": run_threshold}

    # Sort velocities to analyze distribution
    sorted_velocities = np.sort(valid_velocities)

    # Check for constant or near-constant velocity
    velocity_range = sorted_velocities[-1] - sorted_velocities[0]
    velocity_std = np.std(sorted_velocities)

    if velocity_range < 1.0 or velocity_std < 0.5:
        # All frames have similar velocity - single state animation
        # Create thresholds that classify everything as the same state
        median_vel = np.median(sorted_velocities)

        # PROCEDURAL: Use median velocity to infer state boundaries without hardcoded constants
        # Create thresholds around the median to allow some variation
        return {
            "idle": median_vel * 0.3,  # 30% of median
            "walk": median_vel * 0.7,  # 70% of median
            "run": median_vel * 1.2,  # 120% of median (above current speed)
        }

    # PROCEDURAL: Use gap detection to find natural boundaries in velocity distribution
    # Calculate gaps between consecutive sorted velocities
    gaps = np.diff(sorted_velocities)

    # Find large gaps (potential state boundaries)
    # Large gap = above 90th percentile of all gaps
    gap_threshold = np.percentile(gaps, 90)
    large_gap_indices = np.where(gaps > gap_threshold)[0]

    # If we found at least 2 large gaps, use them as state boundaries
    if len(large_gap_indices) >= 2:
        # Use first two large gaps as idle/walk and walk/run boundaries
        idle_threshold = sorted_velocities[large_gap_indices[0]]
        walk_threshold = sorted_velocities[large_gap_indices[1]]

        # If there's a third gap, use it for run/sprint boundary
        if len(large_gap_indices) >= 3:
            run_threshold = sorted_velocities[large_gap_indices[2]]
        else:
            # Otherwise use 75th percentile for run threshold
            run_threshold = np.percentile(sorted_velocities, 75)
    else:
        # Fallback to percentile-based if gap detection doesn't find clear boundaries
        # Use 10th, 40th, 75th percentiles as in original implementation
        idle_threshold = np.percentile(sorted_velocities, 10)
        walk_threshold = np.percentile(sorted_velocities, 40)
        run_threshold = np.percentile(sorted_velocities, 75)

    # Ensure thresholds are strictly increasing with meaningful gaps
    # ADAPTIVE: Require larger gaps (15% of range) to avoid flickering states
    min_gap = velocity_range * 0.15  # At least 15% of range between thresholds

    # Check if thresholds are too close together (low variance animation)
    # If gaps are smaller than 15% of range, velocity is too consistent for multi-state classification
    threshold_span = run_threshold - idle_threshold
    median_vel = np.median(sorted_velocities)

    # Calculate coefficient of variation (CV) to detect low-variance animations
    mean_vel = np.mean(sorted_velocities)
    std_vel = np.std(sorted_velocities)
    cv = std_vel / mean_vel if mean_vel > 0 else 0  # Coefficient of variation

    # Debug logging
    print(
        f"    🔬 Velocity range: {velocity_range:.1f}, threshold span: {threshold_span:.1f} ({threshold_span/velocity_range*100:.1f}% of range), CV: {cv:.3f}"
    )

    # ADAPTIVE: Detect single-state animations using multiple criteria
    # 1. Threshold span < 40% of range (tight clustering)
    # 2. Coefficient of variation < 12% (low relative variance)
    if threshold_span < velocity_range * 0.4 or cv < 0.12:
        # Classify as single state based on velocity range
        # Set thresholds so ALL frames fall into the dominant state (running)
        min_vel = sorted_velocities[0]
        max_vel = sorted_velocities[-1]
        print(f"    ⚠️  Low variance detected (CV={cv:.3f}) - classifying all as single state")
        return {
            "idle": min_vel * 0.5,  # Well below minimum (nothing should be idle)
            "walk": min_vel * 0.9,  # Just below minimum (nothing should be walking)
            "run": max_vel * 1.1,  # Above maximum (everything classified as running)
        }

    if walk_threshold <= idle_threshold + min_gap:
        walk_threshold = idle_threshold + max(min_gap, idle_threshold * 0.5)
    if run_threshold <= walk_threshold + min_gap:
        run_threshold = walk_threshold + max(min_gap, walk_threshold * 0.5)

    return {"idle": idle_threshold, "walk": walk_threshold, "run": run_threshold}


def calculate_adaptive_vertical_thresholds(velocity_y_values, acceleration_y_values):
    """
    Calculate adaptive thresholds for aerial state detection.

    Analyzes vertical velocity and acceleration distributions to find
    natural boundaries for jumping, falling, and landing detection.

    Args:
        velocity_y_values: Array of vertical velocity components
        acceleration_y_values: Array of vertical acceleration components

    Returns:
        dict: Adaptive thresholds for aerial detection
    """
    # Remove NaN/inf values
    valid_vel_y = velocity_y_values[np.isfinite(velocity_y_values)]
    valid_acc_y = acceleration_y_values[np.isfinite(acceleration_y_values)]

    if len(valid_vel_y) == 0 or len(valid_acc_y) == 0:
        # Fallback to hardcoded thresholds
        return {
            "airborne_velocity": VERTICAL_VELOCITY_AIRBORNE_THRESHOLD,
            "jump_acceleration": ACCELERATION_JUMP_THRESHOLD,
            "land_acceleration": ACCELERATION_LAND_THRESHOLD,
        }

    # Airborne threshold: Use MAD (Median Absolute Deviation) to detect outliers
    # Most frames are grounded, so vertical velocity near zero
    median_vel_y = np.median(valid_vel_y)
    mad_vel_y = np.median(np.abs(valid_vel_y - median_vel_y))

    # Handle case where there's no vertical movement (MAD = 0)
    if mad_vel_y < 0.1:
        # Check if there's ANY vertical velocity variation
        vel_y_range = np.max(valid_vel_y) - np.min(valid_vel_y)

        if vel_y_range < 1.0:
            # No significant vertical movement - set high threshold to avoid false aerials
            airborne_threshold = abs(median_vel_y) + 10.0
        else:
            # Some variation but low MAD - use standard deviation
            airborne_threshold = max(3.0 * np.std(valid_vel_y), 1.0)
    else:
        # Airborne = significant deviation from median (3x MAD)
        airborne_threshold = max(3.0 * mad_vel_y, 1.0)  # At least 1.0 to avoid false positives

    # Jump/land acceleration thresholds: Use percentiles
    # But ensure they're meaningful (not just noise)
    acc_y_range = np.max(valid_acc_y) - np.min(valid_acc_y)

    if acc_y_range < 10.0:
        # Very little acceleration variation - probably all grounded
        # Set extreme thresholds to avoid false aerial detection
        land_threshold = np.min(valid_acc_y) - 100.0
        jump_threshold = np.max(valid_acc_y) + 100.0
    else:
        # Landing = sudden negative acceleration (bottom 5th percentile)
        land_threshold = np.percentile(valid_acc_y, 5)

        # Jumping = sudden positive acceleration (top 95th percentile)
        jump_threshold = np.percentile(valid_acc_y, 95)

        # Ensure thresholds are meaningful (not just noise)
        # Jump should be positive, land should be negative
        if jump_threshold < 10.0:
            jump_threshold = ACCELERATION_JUMP_THRESHOLD
        if land_threshold > -10.0:
            land_threshold = ACCELERATION_LAND_THRESHOLD

    return {
        "airborne_velocity": airborne_threshold,
        "jump_acceleration": jump_threshold,
        "land_acceleration": land_threshold,
    }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def classify_motion_state(
    velocity_magnitude, velocity_y, acceleration_y, velocity_thresholds=None, vertical_thresholds=None
):
    """
    Classify the current motion state based on velocity and acceleration.

    Uses adaptive thresholds if provided, otherwise falls back to hardcoded constants.

    Args:
        velocity_magnitude: Overall velocity magnitude (horizontal + vertical)
        velocity_y: Vertical velocity component
        acceleration_y: Vertical acceleration component
        velocity_thresholds: Dict with 'idle', 'walk', 'run' thresholds (optional)
        vertical_thresholds: Dict with 'airborne_velocity', 'jump_acceleration', 'land_acceleration' (optional)

    Returns:
        str: Motion state classification
    """
    # Use adaptive thresholds if provided, otherwise use hardcoded constants
    if velocity_thresholds is None:
        velocity_thresholds = {
            "idle": VELOCITY_IDLE_THRESHOLD,
            "walk": VELOCITY_WALK_THRESHOLD,
            "run": VELOCITY_RUN_THRESHOLD,
        }

    if vertical_thresholds is None:
        vertical_thresholds = {
            "airborne_velocity": VERTICAL_VELOCITY_AIRBORNE_THRESHOLD,
            "jump_acceleration": ACCELERATION_JUMP_THRESHOLD,
            "land_acceleration": ACCELERATION_LAND_THRESHOLD,
        }

    # Check for aerial states first (highest priority)
    if velocity_y > vertical_thresholds["airborne_velocity"]:
        if acceleration_y < vertical_thresholds["land_acceleration"]:
            return "landing"
        else:
            return "jumping"
    elif velocity_y < -vertical_thresholds["airborne_velocity"]:
        return "falling"

    # Ground-based locomotion states
    if velocity_magnitude < velocity_thresholds["idle"]:
        return "idle"
    elif velocity_magnitude < velocity_thresholds["walk"]:
        return "walking"
    elif velocity_magnitude < velocity_thresholds["run"]:
        return "running"
    else:
        return "sprinting"


def detect_motion_state_sequence(velocities, accelerations, frame_rate, use_adaptive=True):
    """
    Classify motion state for each frame using adaptive thresholds.

    Analyzes the entire velocity/acceleration distribution to compute
    data-driven thresholds, making classification scale-invariant.

    Args:
        velocities: Array of velocity vectors (n_frames, 3)
        accelerations: Array of acceleration vectors (n_frames, 3)
        frame_rate: Animation frame rate
        use_adaptive: Whether to use adaptive thresholds (default: True)

    Returns:
        list: Motion state classification per frame
    """
    if len(velocities) == 0:
        return []

    velocity_mags = np.linalg.norm(velocities, axis=1)
    velocity_y = velocities[:, 1] if velocities.shape[1] > 1 else np.zeros(len(velocities))
    acceleration_y = accelerations[:, 1] if accelerations.shape[1] > 1 else np.zeros(len(accelerations))

    # Calculate adaptive thresholds from data distribution
    if use_adaptive:
        velocity_thresholds = calculate_adaptive_velocity_thresholds(velocity_mags)
        vertical_thresholds = calculate_adaptive_vertical_thresholds(velocity_y, acceleration_y)

        # Log adaptive thresholds for debugging
        print(
            f"    Adaptive velocity thresholds: idle={velocity_thresholds['idle']:.1f}, "
            f"walk={velocity_thresholds['walk']:.1f}, run={velocity_thresholds['run']:.1f} units/sec"
        )
    else:
        velocity_thresholds = None
        vertical_thresholds = None

    motion_states = []
    for i in range(len(velocities)):
        state = classify_motion_state(
            velocity_mags[i],
            velocity_y[i],
            acceleration_y[i],
            velocity_thresholds=velocity_thresholds,
            vertical_thresholds=vertical_thresholds,
        )
        motion_states.append(state)

    return motion_states


def detect_state_transitions(motion_state_sequence, frame_rate):
    """
    Detect transitions between motion states.

    Filters out transient noise by requiring state to be stable
    for a frame-rate-aware duration before confirming a transition.

    Args:
        motion_state_sequence: List of motion state classifications per frame
        frame_rate: Animation frame rate

    Returns:
        list: State transition events with timing
    """
    # PROCEDURAL: Compute frame-aware constants based on frame rate
    frame_constants = compute_frame_aware_constants(frame_rate)
    state_stable_frames = frame_constants["state_stable_frames"]

    if len(motion_state_sequence) < state_stable_frames:
        return []

    transitions = []
    current_state = motion_state_sequence[0]
    current_start_frame = 0
    confirmed_state = current_state
    stable_count = 0

    for frame, state in enumerate(motion_state_sequence):
        if state == confirmed_state:
            # Continue current state
            stable_count = 0
        elif state == current_state:
            # Same as candidate state
            stable_count += 1
            if stable_count >= state_stable_frames:
                # Confirm state change
                if confirmed_state != current_state:
                    duration = (frame - current_start_frame) / frame_rate
                    transitions.append(
                        {
                            "from_state": confirmed_state,
                            "to_state": current_state,
                            "transition_frame": frame - state_stable_frames,
                            "from_state_start_frame": current_start_frame,
                            "from_state_end_frame": frame - state_stable_frames - 1,
                            "from_state_duration_seconds": duration,
                            "transition_type": classify_transition_type(confirmed_state, current_state),
                        }
                    )
                    confirmed_state = current_state
                    current_start_frame = frame - state_stable_frames
        else:
            # New candidate state
            current_state = state
            stable_count = 1

    return transitions


def classify_transition_type(from_state, to_state):
    """
    Classify the type of motion transition.

    Args:
        from_state: Starting motion state
        to_state: Ending motion state

    Returns:
        str: Transition classification
    """
    # Start/stop transitions
    if from_state == "idle" and to_state in ["walking", "running", "sprinting"]:
        return "start_moving"
    elif from_state in ["walking", "running", "sprinting"] and to_state == "idle":
        return "stop_moving"

    # Speed transitions
    elif from_state == "walking" and to_state in ["running", "sprinting"]:
        return "accelerate"
    elif from_state in ["running", "sprinting"] and to_state == "walking":
        return "decelerate"
    elif from_state == "running" and to_state == "sprinting":
        return "accelerate"
    elif from_state == "sprinting" and to_state == "running":
        return "decelerate"

    # Aerial transitions
    elif to_state == "jumping":
        return "takeoff"
    elif from_state in ["jumping", "falling"] and to_state in ["idle", "walking", "running"]:
        return "landing"
    elif from_state == "jumping" and to_state == "falling":
        return "apex"

    else:
        return "other"


def analyze_transition_smoothness(velocities, accelerations, jerks, transition_frame, window_size=10):
    """
    Analyze smoothness of a motion transition.

    Examines velocity, acceleration, and jerk in a window around
    the transition point to classify transition quality.

    Args:
        velocities: Array of velocity magnitudes
        accelerations: Array of acceleration magnitudes
        jerks: Array of jerk magnitudes
        transition_frame: Frame index of transition
        window_size: Frames before/after transition to analyze

    Returns:
        dict: Smoothness metrics
    """
    # Define analysis window
    start_frame = max(0, transition_frame - window_size)
    end_frame = min(len(jerks), transition_frame + window_size)

    # Extract window data
    window_jerks = jerks[start_frame:end_frame]
    window_accelerations = accelerations[start_frame:end_frame]

    # Compute smoothness metrics
    mean_jerk = np.mean(window_jerks)
    max_jerk = np.max(window_jerks)
    std_acceleration = np.std(window_accelerations)

    # Classify smoothness
    if mean_jerk < TRANSITION_JERK_SMOOTH:
        smoothness = "smooth"
    elif mean_jerk < TRANSITION_JERK_MODERATE:
        smoothness = "moderate"
    else:
        smoothness = "abrupt"

    return {
        "mean_jerk": mean_jerk,
        "max_jerk": max_jerk,
        "std_acceleration": std_acceleration,
        "smoothness": smoothness,
    }


def segment_by_motion_state(motion_state_sequence, frame_rate):
    """
    Segment animation into continuous motion state periods.

    Filters out very short segments to focus on sustained states.

    Args:
        motion_state_sequence: List of motion state classifications per frame
        frame_rate: Animation frame rate

    Returns:
        list: Motion state segments with timing
    """
    if not motion_state_sequence:
        return []

    # ADAPTIVE: Compute minimum duration as percentage of total animation length
    # For short animations (< 30 frames), use 15% minimum
    # For longer animations, use max(3 frames, 10% of length) to avoid noise
    total_frames = len(motion_state_sequence)
    if total_frames < 30:
        min_duration_frames = max(3, int(total_frames * 0.15))  # 15% for short clips
    else:
        min_duration_frames = max(5, int(total_frames * 0.10))  # 10% for longer clips

    print(
        f"    📏 Adaptive min duration: {min_duration_frames} frames ({min_duration_frames/total_frames*100:.1f}% of {total_frames} total)"
    )

    segments = []
    current_state = motion_state_sequence[0]
    segment_start = 0

    for frame, state in enumerate(motion_state_sequence):
        if state != current_state:
            # End current segment
            duration_frames = frame - segment_start
            if duration_frames >= min_duration_frames:
                segments.append(
                    {
                        "motion_state": current_state,
                        "start_frame": segment_start,
                        "end_frame": frame - 1,
                        "duration_frames": duration_frames,
                        "duration_seconds": duration_frames / frame_rate,
                    }
                )
            else:
                print(
                    f"    ⏭ Skipping {current_state} segment (frames {segment_start}-{frame-1}: "
                    f"{duration_frames} frames < {min_duration_frames} min)"
                )

            # Start new segment
            current_state = state
            segment_start = frame

    # Handle final segment
    duration_frames = len(motion_state_sequence) - segment_start
    if duration_frames >= min_duration_frames:
        segments.append(
            {
                "motion_state": current_state,
                "start_frame": segment_start,
                "end_frame": len(motion_state_sequence) - 1,
                "duration_frames": duration_frames,
                "duration_seconds": duration_frames / frame_rate,
            }
        )
    else:
        print(
            f"    ⏭ Skipping final {current_state} segment (frames {segment_start}-{len(motion_state_sequence)-1}: "
            f"{duration_frames} frames < {min_duration_frames} min)"
        )

    return segments


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================


def analyze_motion_transitions(scene, output_dir="output/"):
    """
    Detect and analyze motion state transitions.

    Processes root motion trajectory to identify changes in locomotion type
    (idle, walk, run, jump, etc.) and characterize transition smoothness.

    Args:
        scene: FBX scene object
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary of motion transitions and states
    """
    ensure_output_dir(output_dir)

    # Extract trajectory data using cached utility
    trajectory = extract_root_trajectory(scene)

    # Unpack trajectory data
    trajectory_data = trajectory["trajectory_data"]
    frame_rate = trajectory["frame_rate"]

    # Delegate to the analysis function (pass full trajectory for cached derivatives)
    return analyze_motion_transitions_from_trajectory(trajectory_data, frame_rate, output_dir, trajectory)


def analyze_motion_transitions_from_trajectory(trajectory_data, frame_rate, output_dir="output/", trajectory=None):
    """
    Detect and analyze motion state transitions from trajectory data.

    Processes trajectory data to identify changes in locomotion type
    (idle, walk, run, jump, etc.) and characterize transition smoothness.

    Args:
        trajectory_data: List of trajectory dictionaries (from root_motion_analysis)
        frame_rate: Animation frame rate
        output_dir: Output directory for CSV files
        trajectory: Optional full trajectory dict with cached derivatives (for performance)

    Returns:
        dict: Summary of motion transitions and states
    """
    ensure_output_dir(output_dir)

    if not trajectory_data:
        raise ValueError("No trajectory data provided")

    print("Analyzing motion state transitions...")

    # Extract velocity data from trajectory
    velocities = np.array([[d["velocity_x"], d["velocity_y"], d["velocity_z"]] for d in trajectory_data])

    # Handle edge case: single frame (can't compute derivatives)
    if len(velocities) < 2:
        # Return empty results for minimal trajectory
        return {
            "transitions_count": 0,
            "states_count": 0,
            "transition_type_distribution": {},
            "avg_state_durations": {},
            "transitions": [],
            "motion_states": [],
            "transition_quality": [],
        }

    # Use cached accelerations and jerks (no need to recompute!)
    # These are now cached in the trajectory for performance optimization
    accelerations = None
    jerks = None

    if trajectory is not None:
        accelerations = trajectory.get("accelerations")
        jerks = trajectory.get("jerks")

    # Fallback: If cache doesn't have them or trajectory not provided, compute
    if accelerations is None or jerks is None:
        from fbx_tool.analysis.velocity_analysis import compute_derivatives

        positions = np.array([[d["position_x"], d["position_y"], d["position_z"]] for d in trajectory_data])
        velocities_calc, accelerations, jerks = compute_derivatives(positions, frame_rate)

    # Classify motion state for each frame
    print(f"  🔍 Detecting motion states for {len(velocities)} frames...")
    motion_state_sequence = detect_motion_state_sequence(velocities, accelerations, frame_rate)

    # Log state distribution
    state_counts = {}
    for state in motion_state_sequence:
        state_counts[state] = state_counts.get(state, 0) + 1
    print(f"  📊 State distribution: {state_counts}")

    # Detect state transitions
    state_transitions = detect_state_transitions(motion_state_sequence, frame_rate)

    # Analyze transition smoothness
    velocity_mags = np.linalg.norm(velocities, axis=1)
    acceleration_mags = np.linalg.norm(accelerations, axis=1)
    jerk_mags = np.linalg.norm(jerks, axis=1)

    transition_quality_data = []
    for transition in state_transitions:
        smoothness_metrics = analyze_transition_smoothness(
            velocity_mags, acceleration_mags, jerk_mags, transition["transition_frame"]
        )

        transition_quality_data.append(
            {
                "transition_frame": transition["transition_frame"],
                "from_state": transition["from_state"],
                "to_state": transition["to_state"],
                "transition_type": transition["transition_type"],
                **smoothness_metrics,
            }
        )

    # Segment by motion state
    print(f"  🔧 Segmenting {len(motion_state_sequence)} states...")
    motion_state_segments = segment_by_motion_state(motion_state_sequence, frame_rate)
    print(f"  ✅ Created {len(motion_state_segments)} motion state segments")

    # Write motion transitions CSV
    if state_transitions:
        transitions_csv_path = os.path.join(output_dir, "motion_transitions.csv")
        with open(transitions_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=state_transitions[0].keys())
            writer.writeheader()
            writer.writerows(state_transitions)

    # Write motion states CSV
    if motion_state_segments:
        states_csv_path = os.path.join(output_dir, "motion_states.csv")
        with open(states_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=motion_state_segments[0].keys())
            writer.writeheader()
            writer.writerows(motion_state_segments)

    # Write transition quality CSV
    if transition_quality_data:
        quality_csv_path = os.path.join(output_dir, "transition_quality.csv")
        with open(quality_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=transition_quality_data[0].keys())
            writer.writeheader()
            writer.writerows(transition_quality_data)

    print(f"✓ Motion transition analysis complete:")
    print(f"  - {len(state_transitions)} state transitions detected")
    print(f"  - {len(motion_state_segments)} motion state segments identified")

    # Compute summary statistics
    transition_type_counts = {}
    for transition in state_transitions:
        trans_type = transition["transition_type"]
        transition_type_counts[trans_type] = transition_type_counts.get(trans_type, 0) + 1

    state_duration_stats = {}
    for segment in motion_state_segments:
        state = segment["motion_state"]
        if state not in state_duration_stats:
            state_duration_stats[state] = []
        state_duration_stats[state].append(segment["duration_seconds"])

    # Average duration per state
    avg_state_durations = {state: np.mean(durations) for state, durations in state_duration_stats.items()}

    return {
        "transitions_count": len(state_transitions),
        "states_count": len(motion_state_segments),
        "transition_type_distribution": transition_type_counts,
        "avg_state_durations": avg_state_durations,
        "transitions": state_transitions,
        "motion_states": motion_state_segments,
        "transition_quality": transition_quality_data,
    }
