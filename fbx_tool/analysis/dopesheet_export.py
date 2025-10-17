"""
Dopesheet Export Module
Exports animation data in proper dopesheet format: bones as rows, frames as columns.
"""
import csv
import fbx
from fbx_tool.analysis.utils import prepare_output_file, get_animation_info, collect_bone_names


def export_dopesheet(scene, output_path, frame_rate=None):
    """Export animation dopesheet in optimized tabular format (bones × frames)."""
    prepare_output_file(output_path)
    
    root = scene.GetRootNode()
    if not root:
        raise RuntimeError("No root node found in FBX scene.")
    
    anim_info = get_animation_info(scene)
    if frame_rate is None:
        frame_rate = anim_info["frame_rate"]
    
    frame_time = 1.0 / frame_rate
    start = anim_info["start"]
    stop = anim_info["stop"]

    # Collect frame times
    frame_times = []
    current_time = start
    while current_time <= stop:
        frame_times.append(current_time)
        current_time += frame_time

    # Collect bones
    bone_names = collect_bone_names(scene)

    # Build transform data
    bone_data = {}
    components = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Sx", "Sy", "Sz"]
    
    for bone_name in bone_names:
        bone_data[bone_name] = {comp: [] for comp in components}
    
    # Extract data for each frame
    for frame_time_val in frame_times:
        time_obj = fbx.FbxTime()
        time_obj.SetSecondDouble(frame_time_val)
        
        def extract_bone_data(node):
            transform = node.EvaluateGlobalTransform(time_obj)
            t = transform.GetT()
            r = transform.GetR()
            s = transform.GetS()
            bone_name = node.GetName()
            
            if bone_name in bone_data:
                bone_data[bone_name]["Tx"].append(f"{t[0]:.6f}")
                bone_data[bone_name]["Ty"].append(f"{t[1]:.6f}")
                bone_data[bone_name]["Tz"].append(f"{t[2]:.6f}")
                bone_data[bone_name]["Rx"].append(f"{r[0]:.6f}")
                bone_data[bone_name]["Ry"].append(f"{r[1]:.6f}")
                bone_data[bone_name]["Rz"].append(f"{r[2]:.6f}")
                bone_data[bone_name]["Sx"].append(f"{s[0]:.6f}")
                bone_data[bone_name]["Sy"].append(f"{s[1]:.6f}")
                bone_data[bone_name]["Sz"].append(f"{s[2]:.6f}")
            
            for i in range(node.GetChildCount()):
                extract_bone_data(node.GetChild(i))
        
        for i in range(root.GetChildCount()):
            extract_bone_data(root.GetChild(i))

    # Write to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Bone", "Component"] + [f"Frame_{i:03d}" for i in range(len(frame_times))]
        writer.writerow(header)
        
        for bone_name in bone_names:
            for component in components:
                row = [bone_name, component] + bone_data[bone_name][component]
                writer.writerow(row)
    
    print(f"Optimized dopesheet exported: {len(bone_names)} bones × {len(frame_times)} frames × {len(components)} components = {len(bone_names) * len(components)} rows")
