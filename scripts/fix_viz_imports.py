from pathlib import Path

root = Path.cwd()

files = [
    "fbx_tool/visualization/matplotlib_viewer.py",
    "fbx_tool/visualization/opengl_viewer.py"
]

for filepath in files:
    full_path = root / filepath
    
    if not full_path.exists():
        print(f"⚠️  File not found: {filepath}")
        continue
    
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix imports
    content = content.replace(
        "from analysis_modules.", 
        "from fbx_tool.analysis."
    )
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed: {filepath}")

print("\nDone!")
