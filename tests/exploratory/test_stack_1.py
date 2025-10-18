"""Test animation from stack 1 (mixamo.com)."""
import sys

import fbx
import numpy as np

fbx_file = "C:\\Users\\posne\\Downloads\\Walking.fbx"

# Load FBX
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, "")
importer = fbx.FbxImporter.Create(manager, "")

if not importer.Initialize(fbx_file, -1):
    print(f"Failed to load: {fbx_file}")
    sys.exit(1)

importer.Import(scene)
importer.Destroy()

# Use stack 1 (mixamo.com) instead of stack 0
stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
print(f"Total stacks: {stack_count}\n")

for stack_idx in range(stack_count):
    stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), stack_idx)
    print(f"=== Testing Stack {stack_idx}: {stack.GetName()} ===")

    scene.SetCurrentAnimationStack(stack)

    time_span = stack.GetLocalTimeSpan()
    start = time_span.GetStart().GetSecondDouble()
    stop = time_span.GetStop().GetSecondDouble()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    frame_rate = fbx.FbxTime.GetFrameRate(time_mode)
    frame_time = 1.0 / frame_rate

    print(f"Duration: {start:.2f}s to {stop:.2f}s ({stop - start:.2f}s)")
    print(f"Frame rate: {frame_rate} fps")
    print(f"Frames: ~{int((stop - start) / frame_time)}\n")

    # Test Hips bone
    node = scene.FindNodeByName("mixamorig:Hips")
    if not node:
        print("Hips not found!\n")
        continue

    positions = []
    rotations = []

    current = start
    frame_idx = 0
    while current <= stop:
        t = fbx.FbxTime()
        t.SetSecondDouble(current)

        global_transform = node.EvaluateGlobalTransform(t)
        translation = global_transform.GetT()
        rotation = global_transform.GetQ()

        positions.append([translation[0], translation[1], translation[2]])
        rotations.append([rotation[0], rotation[1], rotation[2], rotation[3]])

        # Show first 3 frames
        if frame_idx < 3:
            print(f"Frame {frame_idx} (t={current:.3f}s):")
            print(f"  Pos: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
            print(f"  Rot: [{rotation[0]:.4f}, {rotation[1]:.4f}, {rotation[2]:.4f}, {rotation[3]:.4f}]")

        current += frame_time
        frame_idx += 1

    pos_arr = np.array(positions)
    rot_arr = np.array(rotations)

    pos_variance = np.var(pos_arr, axis=0)
    rot_variance = np.var(rot_arr, axis=0)

    print(f"\nVariance Analysis:")
    print(f"  Position: X={pos_variance[0]:.6f}, Y={pos_variance[1]:.6f}, Z={pos_variance[2]:.6f}")
    print(
        f"  Rotation: X={rot_variance[0]:.6f}, Y={rot_variance[1]:.6f}, Z={rot_variance[2]:.6f}, W={rot_variance[3]:.6f}"
    )

    if np.any(pos_variance > 0.01) or np.any(rot_variance > 0.0001):
        print("  STATUS: ANIMATED!\n")
    else:
        print("  STATUS: STATIC\n")

manager.Destroy()
