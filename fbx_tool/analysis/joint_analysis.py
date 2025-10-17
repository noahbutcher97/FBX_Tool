"""Joint Analysis Module"""

import csv
import numpy as np
import fbx
from analysis_modules.utils import prepare_output_file, get_animation_info, build_bone_hierarchy

def fbx_vector_to_array(vec):
    """Convert FbxVector4 to numpy array"""
    return np.array([vec[0], vec[1], vec[2]])

def analyze_joints(scene, output_dir="output/"):
    """
    Extracts joint-level metrics and IK suitability.

    Args:
        scene: FBX scene object
        output_dir (str): Output directory path (default: "output/")

    Returns:
        dict: Joint summary data {(parent, child): (stability, range_score, ik_score)}
    """
    anim_info = get_animation_info(scene)
    start = anim_info['start']
    stop = anim_info['stop']
    rate = anim_info['frame_rate']
    frame_time = anim_info['frame_time']
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
        std_r = np.std(arr[:, 3:6], axis=0)
        min_r = np.min(arr[:, 3:6], axis=0)
        max_r = np.max(arr[:, 3:6], axis=0)
        rot_range = max_r - min_r
        range_score = np.exp(-np.var(arr[:, 3:6])) * np.clip(np.sum(rot_range) / 540, 0, 1)
        stab = 1 / (1 + np.linalg.norm(std_r))
        ik_score = stab * 0.6 + range_score * 0.4
        joint_summary[joint] = (stab, range_score, ik_score)
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