import re
from pathlib import Path

filepath = Path("fbx_tool/visualization/opengl_viewer.py")

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Better initial camera angle (look down from above)
content = content.replace(
    "self.camera_rotation_x = 30.0",
    "self.camera_rotation_x = -45.0  # Look down from above"
)

# Fix 2: Start further back
content = content.replace(
    "self.camera_distance = 200.0",
    "self.camera_distance = 400.0  # Start further back"
)

# Fix 3: Y-up coordinate (already correct, but ensure it)
# The grid is already on Y=0 which is correct

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("✓ Fixed camera angles!")
print("Restart the viewer to see changes.")
