"""
Root Motion Analysis Module

Extracts and analyzes root bone movement to understand character locomotion.

Root motion describes the overall movement of a character through space,
separate from the internal skeletal animation. This is critical for:
- Understanding direction of travel (forward, backward, strafing)
- Measuring translational and rotational velocity
- Detecting turning behavior
- Analyzing overall displacement and trajectory

This module serves as the foundation for higher-level motion understanding,
feeding into directional change detection and motion classification.

Outputs:
- root_motion_summary.csv: Overall displacement, velocity, rotation metrics
- root_motion_trajectory.csv: Frame-by-frame position and rotation
- root_motion_direction.csv: Directional analysis (forward/backward/left/right)
"""

import csv
import os

import fbx
import numpy as np

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import ensure_output_dir, export_procedural_metadata, extract_root_trajectory

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================
# NOTE: Helper functions have been moved to utils.py to support caching
# and DRY principles across multiple analysis modules.


def analyze_root_motion(scene, output_dir="output/"):
    """
    Comprehensive root motion analysis for character locomotion.

    Extracts:
    - Position trajectory over time
    - Translational velocity and acceleration
    - Rotation trajectory (Euler angles)
    - Angular velocity (turning speed)
    - Direction of travel (forward/backward/strafing)
    - Overall displacement metrics

    Args:
        scene: FBX scene object
        output_dir: Output directory for CSV files

    Returns:
        dict: Summary statistics and trajectory data
    """
    ensure_output_dir(output_dir)

    # Extract trajectory data using cached utility
    # This handles all the heavy lifting: detection, extraction, classification
    trajectory = extract_root_trajectory(scene)

    # Export procedural metadata brain (coordinate system, adaptive thresholds, etc.)
    metadata_path = os.path.join(output_dir, "procedural_metadata.json")
    export_procedural_metadata(trajectory, metadata_path)

    # Unpack trajectory data for analysis
    trajectory_data = trajectory["trajectory_data"]
    frame_rate = trajectory["frame_rate"]
    total_frames = trajectory["total_frames"]
    root_bone_name = trajectory["root_bone_name"]
    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    rotations = trajectory["rotations"]
    angular_velocity_y = trajectory["angular_velocity_y"]
    velocity_mags = trajectory["velocity_mags"]

    # Get scene metadata for duration
    metadata = get_scene_metadata(scene)
    duration = metadata["stop_time"] - metadata["start_time"]

    # Extract direction classifications from trajectory data
    direction_classifications = [frame_data["direction"] for frame_data in trajectory_data]

    # Compute rotations_y for unwrapping (needed for total_rotation_y)
    rotations_y = rotations[:, 1]  # Extract Y-axis rotation (yaw)
    dt = 1.0 / frame_rate

    # Unwrap angles to handle 360° wrapping
    rotations_y_unwrapped = np.unwrap(np.radians(rotations_y))
    rotations_y_unwrapped = np.degrees(rotations_y_unwrapped)

    # Compute overall summary metrics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    displacement = np.linalg.norm(positions[-1] - positions[0])

    mean_velocity = np.mean(velocity_mags)
    max_velocity = np.max(velocity_mags)

    mean_angular_velocity = np.mean(np.abs(angular_velocity_y))
    max_angular_velocity = np.max(np.abs(angular_velocity_y))

    total_rotation_y = abs(rotations_y_unwrapped[-1] - rotations_y_unwrapped[0])

    # Direction distribution
    direction_counts = {}
    for direction in direction_classifications:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    # Dominant direction (most common)
    dominant_direction = max(direction_counts, key=direction_counts.get) if direction_counts else "unknown"

    # Write trajectory CSV
    trajectory_csv_path = os.path.join(output_dir, "root_motion_trajectory.csv")
    with open(trajectory_csv_path, "w", newline="") as f:
        if trajectory_data:
            writer = csv.DictWriter(f, fieldnames=trajectory_data[0].keys())
            writer.writeheader()
            writer.writerows(trajectory_data)

    # Write direction analysis CSV
    direction_data = []
    for direction, count in direction_counts.items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        direction_data.append({"direction": direction, "frame_count": count, "percentage": percentage})

    direction_csv_path = os.path.join(output_dir, "root_motion_direction.csv")
    with open(direction_csv_path, "w", newline="") as f:
        if direction_data:
            writer = csv.DictWriter(f, fieldnames=direction_data[0].keys())
            writer.writeheader()
            writer.writerows(direction_data)

    # Write summary CSV
    summary_data = [
        {
            "root_bone": root_bone_name,
            "duration_seconds": duration,
            "total_frames": total_frames,
            "total_distance": total_distance,
            "displacement": displacement,
            "mean_velocity": mean_velocity,
            "max_velocity": max_velocity,
            "mean_angular_velocity_y": mean_angular_velocity,
            "max_angular_velocity_y": max_angular_velocity,
            "total_rotation_y": total_rotation_y,
            "dominant_direction": dominant_direction,
            "start_position": f"({positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f})",
            "end_position": f"({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f})",
        }
    ]

    summary_csv_path = os.path.join(output_dir, "root_motion_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\n✓ Root motion analysis complete:")
    print(f"  - Total distance traveled: {total_distance:.2f} units")
    print(f"  - Displacement: {displacement:.2f} units")
    print(f"  - Mean velocity: {mean_velocity:.2f} units/s")
    print(f"  - Total rotation: {total_rotation_y:.2f}°")
    print(f"  - Dominant direction: {dominant_direction}")

    # Return summary and trajectory data
    return {
        "root_bone_name": root_bone_name,
        "total_distance": total_distance,
        "displacement": displacement,
        "mean_velocity": mean_velocity,
        "max_velocity": max_velocity,
        "total_rotation_y": total_rotation_y,
        "dominant_direction": dominant_direction,
        "direction_distribution": direction_counts,
        "trajectory_frames": len(trajectory_data),
        "trajectory_data": trajectory_data,
        "frame_rate": frame_rate,
    }
