"""Test if animation data actually varies across frames."""
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

# Get animation info
anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
scene.SetCurrentAnimationStack(anim_stack)
time_span = anim_stack.GetLocalTimeSpan()
start = time_span.GetStart().GetSecondDouble()
stop = time_span.GetStop().GetSecondDouble()
time_mode = scene.GetGlobalSettings().GetTimeMode()
frame_rate = fbx.FbxTime.GetFrameRate(time_mode)
frame_time = 1.0 / frame_rate

# Test multiple bones
test_bones = [
    "mixamorig:Hips",
    "mixamorig:LeftUpLeg",
    "mixamorig:RightUpLeg",
    "mixamorig:LeftArm",
    "mixamorig:RightHand"
]

print(f"\n=== Testing Animation Variance ===")
print(f"Frames: {int((stop - start) / frame_time)}")
print(f"Duration: {stop - start:.2f}s")
print()

for bone_name in test_bones:
    node = scene.FindNodeByName(bone_name)
    if not node:
        print(f"✗ {bone_name}: NOT FOUND")
        continue

    positions = []
    rotations = []

    current = start
    while current <= stop:
        t = fbx.FbxTime()
        t.SetSecondDouble(current)

        global_transform = node.EvaluateGlobalTransform(t)
        translation = global_transform.GetT()
        rotation = global_transform.GetQ()

        positions.append([translation[0], translation[1], translation[2]])
        rotations.append([rotation[0], rotation[1], rotation[2], rotation[3]])

        current += frame_time

    pos_arr = np.array(positions)
    rot_arr = np.array(rotations)

    pos_variance = np.var(pos_arr, axis=0)
    rot_variance = np.var(rot_arr, axis=0)

    pos_range = np.ptp(pos_arr, axis=0)  # peak-to-peak (max - min)
    rot_range = np.ptp(rot_arr, axis=0)

    print(f"{bone_name}:")
    print(f"  Position variance: X={pos_variance[0]:.6f}, Y={pos_variance[1]:.6f}, Z={pos_variance[2]:.6f}")
    print(f"  Position range:    X={pos_range[0]:.2f}, Y={pos_range[1]:.2f}, Z={pos_range[2]:.2f}")
    print(f"  Rotation variance: X={rot_variance[0]:.6f}, Y={rot_variance[1]:.6f}, Z={rot_variance[2]:.6f}, W={rot_variance[3]:.6f}")
    print(f"  Rotation range:    X={rot_range[0]:.4f}, Y={rot_range[1]:.4f}, Z={rot_range[2]:.4f}, W={rot_range[3]:.4f}")

    if np.all(pos_variance < 0.01) and np.all(rot_variance < 0.0001):
        print(f"  ⚠ STATIC (no movement)")
    else:
        print(f"  ✓ ANIMATED")
    print()

manager.Destroy()
