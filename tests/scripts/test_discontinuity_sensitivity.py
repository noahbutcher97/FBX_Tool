"""
Test script to verify curve discontinuity detection isn't missing false negatives.
Creates synthetic curves with known discontinuities and tests detection.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from fbx_tool.analysis.constraint_violation_detection import detect_curve_discontinuities


def create_smooth_curve(n_frames=50):
    """Create a smooth sinusoidal curve."""
    t = np.linspace(0, 4 * np.pi, n_frames)
    return np.sin(t) * 10


def create_curve_with_jump(n_frames=50, jump_frame=25, jump_magnitude=20):
    """Create smooth curve with a single discontinuity."""
    curve = create_smooth_curve(n_frames)
    # Add a jump at jump_frame
    curve[jump_frame:] += jump_magnitude
    return curve


def create_curve_with_spike(n_frames=50, spike_frame=25, spike_magnitude=30):
    """Create smooth curve with a single-frame spike."""
    curve = create_smooth_curve(n_frames)
    # Add a spike (goes up and back down)
    curve[spike_frame] += spike_magnitude
    return curve


def analyze_acceleration_based_detection(curve, description):
    """Manually compute what our detection should find."""
    velocities = np.diff(curve)
    accelerations = np.diff(velocities)

    abs_accelerations = np.abs(accelerations)
    median_accel = np.median(abs_accelerations)
    mad = np.median(np.abs(abs_accelerations - median_accel))

    if mad > 0:
        threshold_mad = 3.5 * 1.4826 * mad
        threshold = median_accel + threshold_mad

        # Count outliers
        outliers = abs_accelerations > threshold
        extreme_outliers = abs_accelerations > (median_accel * 5.0)

        print(f"\n{description}:")
        print(f"  Median acceleration: {median_accel:.4f}")
        print(f"  MAD: {mad:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Outliers (>threshold): {np.sum(outliers)}")
        print(f"  Extreme outliers (>5x median): {np.sum(extreme_outliers)}")
        print(f"  Max acceleration: {np.max(abs_accelerations):.4f}")

        if np.sum(extreme_outliers) > 0:
            print(f"  [YES] Would detect discontinuity at frames: {np.where(extreme_outliers)[0] + 2}")
        else:
            print(
                f"  [NO] Would NOT detect discontinuity (max accel only {np.max(abs_accelerations)/median_accel:.1f}x median)"
            )


print("=" * 70)
print("DISCONTINUITY DETECTION SENSITIVITY TEST")
print("=" * 70)

# Test 1: Smooth curve (should detect 0)
print("\n--- Test 1: Smooth Curve (NO discontinuity) ---")
smooth = create_smooth_curve()
analyze_acceleration_based_detection(smooth, "Smooth sinusoid")

# Test 2: Large jump (should detect 1)
print("\n--- Test 2: Large Jump (jump_magnitude=20) ---")
jump = create_curve_with_jump(jump_magnitude=20)
analyze_acceleration_based_detection(jump, "Large jump")

# Test 3: Small jump (edge case - might not detect)
print("\n--- Test 3: Small Jump (jump_magnitude=5) ---")
small_jump = create_curve_with_jump(jump_magnitude=5)
analyze_acceleration_based_detection(small_jump, "Small jump")

# Test 4: Single-frame spike (should detect 2 - up and down)
print("\n--- Test 4: Single-Frame Spike (spike_magnitude=30) ---")
spike = create_curve_with_spike(spike_magnitude=30)
analyze_acceleration_based_detection(spike, "Single-frame spike")

# Test 5: Rapid smooth motion (should NOT detect)
print("\n--- Test 5: Rapid Smooth Motion (high frequency sine) ---")
t = np.linspace(0, 20 * np.pi, 50)  # High frequency
rapid_smooth = np.sin(t) * 20  # Large amplitude
analyze_acceleration_based_detection(rapid_smooth, "Rapid smooth motion")

# Test 6: Real walking foot motion (should NOT detect)
print("\n--- Test 6: Simulated Walking Foot (Z-axis) ---")
# Simulate foot swing: slow-fast-slow with smooth acceleration
t = np.linspace(0, np.pi, 50)
foot_z = -10 * np.cos(t) + 10  # 0 to 20 range, smooth acceleration curve
analyze_acceleration_based_detection(foot_z, "Walking foot Z-motion")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("[YES] Detection should flag: Test 2 (large jump), Test 4 (spike)")
print("[NO] Detection should NOT flag: Test 1, 5, 6 (smooth motion)")
print("[?] Test 3 (small jump) - depends on threshold sensitivity")
print("\nIf Test 5 or 6 are flagged, we have FALSE POSITIVES")
print("If Test 2 or 4 are NOT flagged, we have FALSE NEGATIVES")
