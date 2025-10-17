"""Test animation data extraction."""
import fbx
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python test_animation_extraction.py <fbx_file>")
    sys.exit(1)

fbx_file = sys.argv[1]

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

print(f"\n=== Animation Info ===")
print(f"Start: {start}s")
print(f"Stop: {stop}s")
print(f"Duration: {stop - start}s")
print(f"Frame Rate: {frame_rate} fps")
print(f"Frame Time: {frame_time}s")
print(f"Expected Frames: ~{int((stop - start) / frame_time)}")

# Test extracting position data for a specific bone
test_bone = "mixamorig:Hips"
node = scene.FindNodeByName(test_bone)

if node:
    print(f"\n=== Testing bone: {test_bone} ===")
    positions = []

    current = start
    frame_idx = 0

    while current <= stop:
        t = fbx.FbxTime()
        t.SetSecondDouble(current)

        global_transform = node.EvaluateGlobalTransform(t)
        translation = global_transform.GetT()
        position = np.array([translation[0], translation[1], translation[2]])
        positions.append(position)

        if frame_idx < 5 or frame_idx >= int((stop - start) / frame_time) - 2:
            print(f"Frame {frame_idx} (t={current:.3f}s): {position}")

        current += frame_time
        frame_idx += 1

    print(f"\nTotal frames extracted: {frame_idx}")
    print(f"Positions array shape: {np.array(positions).shape}")

    # Check if animation actually changes
    positions_arr = np.array(positions)
    position_variance = np.var(positions_arr, axis=0)
    print(f"\nPosition variance (X, Y, Z): {position_variance}")
    print(f"Animation is {'STATIC' if np.all(position_variance < 0.01) else 'ANIMATED'}")

else:
    print(f"\n✗ Bone '{test_bone}' not found!")

manager.Destroy()
