"""Test script to check FBX file coordinate system and animation data."""
import sys

import fbx

if len(sys.argv) < 2:
    print("Usage: python test_fbx_coordinate_system.py <fbx_file>")
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

# Get axis system
axis_system = scene.GetGlobalSettings().GetAxisSystem()
up_vector, up_sign = axis_system.GetUpVector()
front_vector, front_sign = axis_system.GetFrontVector()
coord_system = axis_system.GetCoorSystem()

print("\n=== FBX File Information ===")
print(f"File: {fbx_file}")
print(f"\n--- Coordinate System ---")
print(f"Up Vector: {up_vector} (sign: {up_sign})")
print(f"  0 = X-up, 1 = Y-up, 2 = Z-up")
print(f"Front Vector: {front_vector} (sign: {front_sign})")
print(f"Coord System: {coord_system}")
print(f"  0 = Right-handed, 1 = Left-handed")

# Get animation info
anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
print(f"\n--- Animation ---")
print(f"Animation stacks: {anim_stack_count}")

if anim_stack_count > 0:
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    scene.SetCurrentAnimationStack(anim_stack)

    time_span = anim_stack.GetLocalTimeSpan()
    start = time_span.GetStart().GetSecondDouble()
    stop = time_span.GetStop().GetSecondDouble()
    duration = stop - start

    mode = scene.GetGlobalSettings().GetTimeMode()
    frame_rate = fbx.FbxTime.GetFrameRate(mode)

    print(f"Animation duration: {duration:.2f}s ({start:.2f} to {stop:.2f})")
    print(f"Frame rate: {frame_rate} fps")
    print(f"Total frames: ~{int(duration * frame_rate)}")
else:
    print("No animation data found!")

# Get skeleton info
root = scene.GetRootNode()
bone_count = 0


def count_bones(node):
    global bone_count
    if node.GetNodeAttribute():
        attr_type = node.GetNodeAttribute().GetAttributeType()
        if attr_type == fbx.FbxNodeAttribute.eSkeleton:
            bone_count += 1

    for i in range(node.GetChildCount()):
        count_bones(node.GetChild(i))


count_bones(root)

print(f"\n--- Skeleton ---")
print(f"Bones found: {bone_count}")

print("\n=== Recommendation ===")
if up_vector == 1:  # Y-up
    print("✓ This FBX uses Y-up (standard for Mixamo/Unity)")
    print("  Coordinate conversion should be DISABLED")
elif up_vector == 2:  # Z-up
    print("✓ This FBX uses Z-up (standard for Blender/3ds Max)")
    print("  Coordinate conversion should be ENABLED")
else:  # X-up
    print("⚠ This FBX uses X-up (uncommon)")
    print("  May need custom conversion")

manager.Destroy()
