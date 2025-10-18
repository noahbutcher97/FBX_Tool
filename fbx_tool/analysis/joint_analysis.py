"""Joint Analysis Module"""

import csv

import fbx
import numpy as np

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import (
    build_bone_hierarchy,
    compute_ik_suitability,
    fbx_vector_to_array,
    prepare_output_file,
)


def analyze_joints(scene, output_dir="output/"):
    """
    Extracts joint-level metrics and IK suitability.

    Args:
        scene: FBX scene object
        output_dir (str): Output directory path (default: "output/")

    Returns:
        dict: Joint summary data {(parent, child): (stability, range_score, ik_score)}
    """
    anim_info = get_scene_metadata(scene)
    start = anim_info['start_time']
    stop = anim_info['stop_time']
    rate = anim_info['frame_rate']
    frame_time = 1.0 / rate  # Compute frame time from frame rate
    hierarchy = build_bone_hierarchy(scene)

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

            # ✅ FIXED: Convert FbxVector4 to numpy arrays
            rT_arr = fbx_vector_to_array(rT)
            rR_arr = fbx_vector_to_array(rR)

            key = (parent if parent else "Root", child)
            joint_data.setdefault(key, []).append([
                rT_arr[0], rT_arr[1], rT_arr[2],
                rR_arr[0], rR_arr[1], rR_arr[2]
            ])
        current += frame_time

    enhanced = []
    joint_summary = {}
    for joint, vals in joint_data.items():
        arr = np.array(vals)
        rotation_data = arr[:, 3:6]  # Extract rotation columns (X, Y, Z Euler angles)

        # Compute IK suitability using shared function
        stab, range_score, ik_score = compute_ik_suitability(rotation_data)
        joint_summary[joint] = (stab, range_score, ik_score)

        # Get rotation range for CSV output
        min_r = np.min(rotation_data, axis=0)
        max_r = np.max(rotation_data, axis=0)
        enhanced.append([
            joint[0], joint[1],
            min_r[0], max_r[0], min_r[1], max_r[1], min_r[2], max_r[2],
            round(stab, 4), round(range_score, 4), round(ik_score, 4)
        ])

    output_path = output_dir + "joint_enhanced_relationships.csv"
    prepare_output_file(output_path)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Parent", "Child", "MinRotX", "MaxRotX", "MinRotY", "MaxRotY", "MinRotZ", "MaxRotZ", "Stability", "RangeScore", "IKSuitability"])
        w.writerows(enhanced)

    return joint_summary