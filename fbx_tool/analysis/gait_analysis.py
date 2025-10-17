"""Gait Analysis Module"""

import csv
import numpy as np
import fbx
from fbx_tool.analysis.utils import prepare_output_file, get_animation_info, build_bone_hierarchy, get_standard_chains

def fbx_vector_to_array(vec):
    """Convert FbxVector4 to numpy array"""
    return np.array([vec[0], vec[1], vec[2]])

def analyze_gait(scene, output_dir="output/"):
    """
    Performs stride segmentation and gait phase analysis.

    Args:
        scene: FBX scene object
        output_dir (str): Output directory path (default: "output/")

    Returns:
        tuple: (stride_segments, gait_summary)
            stride_segments: List of stride data
            gait_summary: Dictionary with gait metrics
    """
    anim_info = get_animation_info(scene)
    start = anim_info['start']
    stop = anim_info['stop']
    rate = anim_info['frame_rate']
    frame_time = anim_info['frame_time']
    hierarchy = build_bone_hierarchy(scene)
    chains = get_standard_chains()

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
            rT = rel.GetT()

            # ✅ FIXED: Convert FbxVector4 to numpy array
            rT_arr = fbx_vector_to_array(rT)

            key = (parent if parent else "Root", child)
            joint_data.setdefault(key, []).append([rT_arr[0], rT_arr[1], rT_arr[2]])
        current += frame_time

    legs = ["LeftLeg", "RightLeg"]
    stride_segments = []
    summary_info = []

    for cname in legs:
        if cname not in chains:
            continue
        bones = chains[cname]
        foot = bones[-2]
        parent = hierarchy.get(foot)
        key = (parent, foot)
        if key not in joint_data:
            continue

        # Extract Y position (vertical axis)
        pos_data = np.array(joint_data[key])
        pos = pos_data[:, 1]  # Y component
        frames = np.arange(len(pos))

        # Calculate velocity
        vel = np.diff(pos, prepend=pos[0])

        # ✅ FIXED: Find foot contacts (velocity changes from negative to positive)
        contacts_idx = []
        for i in range(1, len(vel) - 1):
            if vel[i-1] <= 0 and vel[i] >= 0:
                contacts_idx.append(i)

        if len(contacts_idx) < 2:
            continue

        last = contacts_idx[0]
        for c in contacts_idx[1:]:
            segment_vel = vel[last:c]
            conf = np.mean(np.pi / (1 + np.abs(segment_vel)))
            stride_segments.append([
                cname, int(last), int(c), "Stride",
                float(len(frames[last:c]) / rate),
                round(float(conf), 4),
                float(pos[c] - pos[last]) if c > last else 0.0,
                0.0
            ])
            last = c

        # Cross-correlation between left and right foot
        if cname == "LeftLeg" and "RightLeg" in chains:
            right = chains["RightLeg"][-2]
            p2 = hierarchy.get(right)
            key_r = (p2, right)
            if key_r in joint_data:
                pL_data = np.array(joint_data[(parent, foot)])
                pR_data = np.array(joint_data[key_r])

                # Use Z position for phase analysis
                pL = pL_data[:, 2]
                pR = pR_data[:, 2]

                # Normalize and correlate
                pL_norm = pL - np.mean(pL)
                pR_norm = pR - np.mean(pR)
                corr = np.correlate(pL_norm, pR_norm, 'full')
                shift = (np.argmax(corr) - len(pL) + 1) / rate

                # Calculate stride metrics
                cycle_time = np.mean([s[4] for s in stride_segments if s[0] == cname]) if stride_segments else 0

                summary_info.append([
                    cname,
                    len(contacts_idx),
                    float(stop - start),
                    float(cycle_time),
                    float(np.mean(pos)),
                    round(float(abs(shift)), 4),
                    "GaitRun" if abs(shift) < 0.1 else "GaitWalk"
                ])

    # Write segments
    output_path = output_dir + "chain_gait_segments.csv"
    prepare_output_file(output_path)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Chain", "FrameStart", "FrameEnd", "Phase", "CycleTime", "Confidence", "StrideLength", "Asymmetry"])
        w.writerows(stride_segments)

    # Write summary
    output_path = output_dir + "gait_summary.csv"
    prepare_output_file(output_path)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Chain", "CycleRateHz", "Duration", "MeanCycleTime", "MeanStrideHeight", "LeftRightPhaseShift", "GaitType"])
        w.writerows(summary_info)

    # ✅ FIXED: Proper dict creation
    gait_summary = {
        "cycle_rate": float(summary_info[0][1] / summary_info[0][2]) if summary_info and summary_info[0][2] > 0 else 0.0,
        "confidence": float(np.mean([s[5] for s in stride_segments])) if stride_segments else 0.0,
        "gait_type": summary_info[0][6] if summary_info else "Unknown"
    }

    return stride_segments, gait_summary