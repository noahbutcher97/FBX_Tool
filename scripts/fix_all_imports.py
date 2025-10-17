
import os
from pathlib import Path

root = Path.cwd()

print("Fixing ALL imports in fbx_tool/analysis/...")
print()

# Fix all files in fbx_tool/analysis/
analysis_dir = root / "fbx_tool" / "analysis"

if analysis_dir.exists():
    for py_file in analysis_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Fix imports
        content = content.replace(
            "from analysis_modules.", 
            "from fbx_tool.analysis."
        )
        content = content.replace(
            "import analysis_modules.",
            "import fbx_tool.analysis."
        )

        if content != original:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {py_file.name}")
        else:
            print(f"  OK: {py_file.name}")

print()
print("Done! All analysis module imports fixed.")
print()
print("Now try: python -m fbx_tool")
