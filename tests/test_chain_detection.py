"""Test dynamic chain detection."""
import sys
sys.path.insert(0, 'C:\\Users\\posne\\Projects\\FBX_Tool')

import fbx
from fbx_tool.analysis.utils import build_bone_hierarchy, detect_chains_from_hierarchy

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

# Build hierarchy and detect chains
hierarchy = build_bone_hierarchy(scene)
chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

print(f"\n=== Detected Chains ===")
print(f"Total chains found: {len(chains)}\n")

for chain_name, bones in chains.items():
    print(f"{chain_name}:")
    for bone in bones:
        print(f"  - {bone}")
    print(f"  Total bones: {len(bones)}\n")

manager.Destroy()
