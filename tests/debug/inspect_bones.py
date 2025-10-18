"""Inspect bone names and hierarchy in FBX file."""
import sys

import fbx

if len(sys.argv) < 2:
    print("Usage: python inspect_bones.py <fbx_file>")
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

# Print all bone names and hierarchy
root = scene.GetRootNode()


def print_hierarchy(node, depth=0):
    indent = "  " * depth
    attr = node.GetNodeAttribute()
    attr_type = ""
    if attr:
        type_id = attr.GetAttributeType()
        type_map = {
            fbx.FbxNodeAttribute.EType.eSkeleton: "[BONE]",
            fbx.FbxNodeAttribute.EType.eMesh: "[MESH]",
            fbx.FbxNodeAttribute.EType.eNull: "[NULL]",
        }
        attr_type = type_map.get(type_id, f"[{type_id}]")

    print(f"{indent}{node.GetName()} {attr_type}")

    for i in range(node.GetChildCount()):
        print_hierarchy(node.GetChild(i), depth + 1)


print("\n=== Bone Hierarchy ===")
print_hierarchy(root)

# Count bone types
bone_count = 0


def count_bones(node):
    global bone_count
    if node.GetNodeAttribute():
        if node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
            bone_count += 1
    for i in range(node.GetChildCount()):
        count_bones(node.GetChild(i))


count_bones(root)
print(f"\n=== Summary ===")
print(f"Total skeleton bones: {bone_count}")

manager.Destroy()
