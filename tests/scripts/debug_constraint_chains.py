"""Debug script to check why constraint detection isn't finding chains."""

import sys

sys.path.insert(0, ".")

# Fix encoding
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from fbx_tool.analysis.constraint_violation_detection import _extract_hierarchy, _get_all_bones
from fbx_tool.analysis.scene_manager import get_scene_manager
from fbx_tool.analysis.utils import detect_chains_from_hierarchy

fbx_path = "assets/Test/FBX/Female Walk.fbx"

with get_scene_manager().get_scene(fbx_path) as scene_ref:
    scene = scene_ref.scene

    # Step 1: Get bones
    bones = _get_all_bones(scene)
    print(f"✓ Found {len(bones)} bones")

    # Step 2: Extract hierarchy
    hierarchy = _extract_hierarchy(bones)
    print(f"✓ Hierarchy has {len(hierarchy)} entries")

    # Step 3: Show sample
    print("\nSample hierarchy (first 10):")
    for i, (child, parent) in enumerate(list(hierarchy.items())[:10]):
        print(f"  {child} -> {parent}")

    # Step 4: Try to detect chains
    chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)
    print(f"\n✓ Detected {len(chains)} chains")

    if chains:
        print("\nChains found:")
        for chain in chains:
            print(f"  {' -> '.join(chain)}")
    else:
        print("\n❌ No chains found!")
        print("\nDebugging hierarchy structure...")
        roots = [k for k, v in hierarchy.items() if v is None]
        print(f"  Roots: {roots}")

        # Count children
        children_count = {}
        for child, parent in hierarchy.items():
            if parent:
                children_count[parent] = children_count.get(parent, 0) + 1

        print(f"  Bones with children: {len(children_count)}")
        top_parents = sorted(children_count.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top parents: {top_parents}")
