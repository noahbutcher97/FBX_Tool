"""Inspect animation layers and curves in FBX file."""
import sys
import fbx

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

print("\n=== Animation Stacks ===")
stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
print(f"Animation stacks: {stack_count}")

for i in range(stack_count):
    stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
    print(f"\nStack {i}: {stack.GetName()}")

    time_span = stack.GetLocalTimeSpan()
    start = time_span.GetStart().GetSecondDouble()
    stop = time_span.GetStop().GetSecondDouble()
    print(f"  Duration: {start:.2f}s to {stop:.2f}s ({stop - start:.2f}s)")

    # Get animation layers
    layer_count = stack.GetMemberCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId))
    print(f"  Animation layers: {layer_count}")

    for j in range(layer_count):
        layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), j)
        print(f"    Layer {j}: {layer.GetName()}")

# Check if specific nodes have animation curves
print("\n=== Checking Animation Curves ===")
test_node_name = "mixamorig:Hips"
node = scene.FindNodeByName(test_node_name)

if node:
    print(f"Node: {test_node_name}")

    # Check all animation curve nodes
    curve_node_count = node.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId))
    print(f"  Animation curve nodes: {curve_node_count}")

    for i in range(curve_node_count):
        curve_node = node.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId), i)
        print(f"    Curve node {i}: {curve_node.GetName()}")

        # Check channels
        channel_count = curve_node.GetChannelsCount()
        print(f"      Channels: {channel_count}")

        for ch in range(channel_count):
            curve = curve_node.GetCurve(ch)
            if curve:
                key_count = curve.KeyGetCount()
                print(f"        Channel {ch}: {key_count} keyframes")
                if key_count > 0:
                    # Show first few keyframes
                    for k in range(min(3, key_count)):
                        time = curve.KeyGetTime(k).GetSecondDouble()
                        value = curve.KeyGetValue(k)
                        print(f"          Key {k}: t={time:.3f}s, value={value:.4f}")
else:
    print(f"Node '{test_node_name}' not found!")

manager.Destroy()
