
import os
import re
from pathlib import Path

root = Path.cwd()

# Files to fix and their import changes
files_to_fix = {
    "fbx_tool/visualization/opengl_viewer.py": {
        "from analysis_modules.utils": "from fbx_tool.analysis.utils",
        "from analysis_modules.": "from fbx_tool.analysis.",
    },
    "fbx_tool/visualization/matplotlib_viewer.py": {
        "from analysis_modules.utils": "from fbx_tool.analysis.utils",
        "from analysis_modules.": "from fbx_tool.analysis.",
    },
    "fbx_tool/gui/main_window.py": {
        "from analysis_modules.": "from fbx_tool.analysis.",
        "from skeleton_visualizer": "from fbx_tool.visualization.matplotlib_viewer",
        "from opengl_viewer": "from fbx_tool.visualization.opengl_viewer",
    },
    "examples/visualize.py": {
        "from analysis_modules.fbx_loader": "from fbx_tool.analysis.fbx_loader",
        "from skeleton_visualizer": "from fbx_tool.visualization.matplotlib_viewer",
        "from opengl_viewer": "from fbx_tool.visualization.opengl_viewer",
    },
    "examples/run_analysis.py": {
        "from analysis_modules.": "from fbx_tool.analysis.",
    }
}

print("Fixing imports...")
print()

for filepath, replacements in files_to_fix.items():
    full_path = root / filepath

    if not full_path.exists():
        print(f"⚠️  File not found: {filepath}")
        continue

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
    else:
        print(f"  Skipped (no changes): {filepath}")

print()
print("Done! All imports fixed.")
