"""Chain Analysis Module"""

import csv
import numpy as np
import fbx
from fbx_tool.analysis.utils import prepare_output_file, get_animation_info, build_bone_hierarchy, get_standard_chains

def fbx_vector_to_array(vec):
    """Convert FbxVector4 to numpy array"""
    return np.array([vec[0], vec[1], vec[2]])

def analyze_chains(scene, output_dir="output/"):
    """
    Performs chain-level IK suitability and cross-temporal coherence analysis.

    Args:
        scene: FBX scene object
        output_dir (str): Output directory path (default: "output/")

    Returns:
        dict: Chain confidence data {chain_name: (mean_ik, cross_temporal, confidence)}
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

    # Compute joint summary
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

    chain_results = []
    for cname, chain in chains.items():
        segs = [(b, hierarchy[b]) for b in chain if hierarchy.get(b) and (hierarchy[b], b) in joint_data]
        if len(segs) < 2:
            continue
        chain_iks = []
        for i in range(len(segs)):
            p_i_er = (hierarchy[segs[i][0]], segs[i][0])
            if p_i_er in joint_summary:
                chain_iks.append(joint_summary[p_i_er][2])  # Get ik_score
        mean_ik = np.mean(chain_iks) if chain_iks else 0

        base = chain[0]
        tip = chain[-1]
        if not (tip in hierarchy):
            continue

        key = (hierarchy.get(tip), tip)
        if key not in joint_data:
            continue

        frames = len(joint_data[key])
        if frames < 3:
            continue

        # Extract position data (first 3 columns)
        t_vecs = np.array(joint_data[key])[:, :3]
        window = int(rate * 0.25)
        corrs = []
        for i in range(0, frames - window, max(1, window // 2)):
            w1 = t_vecs[i:i + window].flatten()
            end_idx = i + window + window // 2
            if end_idx <= frames:
                w2 = t_vecs[i + window // 2:end_idx].flatten()
                if len(w1) == len(w2):
                    corrs.append(np.corrcoef(w1, w2)[0, 1])

        cross_t = np.nanmean(corrs) if len(corrs) > 0 else 0
        final_conf = np.clip(mean_ik * 0.7 + 0.3 * cross_t, 0, 1)
        chain_results.append([cname, round(float(mean_ik), 4), round(float(cross_t), 4), round(float(final_conf), 4)])

    output_path = output_dir + "chain_confidence.csv"
    prepare_output_file(output_path)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Chain", "MeanIKSuitability", "CrossTemporalCoherence", "ChainIKConfidence"])
        w.writerows(chain_results)

    chain_conf = {row[0]: {"mean_ik": row[1], "cross_temp": row[2], "confidence": row[3]} for row in chain_results}
    return chain_conf