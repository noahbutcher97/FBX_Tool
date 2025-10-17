from pathlib import Path
import re

filepath = Path("fbx_tool/visualization/opengl_viewer.py")

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Add coordinate system conversion in __init__
init_section = """        # Animation data
        self.current_frame = 0
        self.total_frames = 0
        self.bone_transforms = {}
        
        # Camera settings
        self.camera_distance = 400.0
        self.camera_rotation_x = -45.0
        self.camera_rotation_y = 45.0
        self.camera_target = np.array([0.0, 0.0, 0.0])"""

replacement = """        # Animation data
        self.current_frame = 0
        self.total_frames = 0
        self.bone_transforms = {}
        
        # Camera settings (orbit camera)
        self.camera_distance = 400.0
        self.camera_azimuth = 45.0    # Horizontal rotation
        self.camera_elevation = 20.0  # Vertical rotation
        self.camera_target = np.array([0.0, 0.0, 0.0])"""

content = content.replace(init_section, replacement)

# Fix 2: Update _extract_transforms to convert Z-up to Y-up
old_extract = """                global_transform = node.EvaluateGlobalTransform(t)
                translation = global_transform.GetT()
                position = np.array([translation[0], translation[1], translation[2]])"""

new_extract = """                global_transform = node.EvaluateGlobalTransform(t)
                translation = global_transform.GetT()
                # Convert from FBX coordinate system (Z-up) to Y-up
                # Swap Y and Z, negate new Z
                position = np.array([translation[0], translation[2], -translation[1]])"""

content = content.replace(old_extract, new_extract)

# Fix 3: Better orbit camera in paintGL
old_camera = """        # Set up camera
        cam_x = self.camera_target[0] + self.camera_distance * np.sin(np.radians(self.camera_rotation_y)) * np.cos(np.radians(self.camera_rotation_x))
        cam_y = self.camera_target[1] + self.camera_distance * np.sin(np.radians(self.camera_rotation_x))
        cam_z = self.camera_target[2] + self.camera_distance * np.cos(np.radians(self.camera_rotation_y)) * np.cos(np.radians(self.camera_rotation_x))
        
        gluLookAt(
            cam_x, cam_y, cam_z,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0.0, 1.0, 0.0
        )"""

new_camera = """        # Orbit camera (azimuth/elevation style)
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        
        cam_x = self.camera_target[0] + self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = self.camera_target[1] + self.camera_distance * np.sin(elevation_rad)
        cam_z = self.camera_target[2] + self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        
        gluLookAt(
            cam_x, cam_y, cam_z,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0.0, 1.0, 0.0  # Y-up
        )"""

content = content.replace(old_camera, new_camera)

# Fix 4: Update mouse controls for orbit
old_mouse = """        if self.mouse_button == Qt.MouseButton.LeftButton:
            # Rotate camera
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
            self.camera_rotation_x = max(-89, min(89, self.camera_rotation_x))"""

new_mouse = """        if self.mouse_button == Qt.MouseButton.LeftButton:
            # Orbit camera
            self.camera_azimuth += dx * 0.5
            self.camera_elevation -= dy * 0.5
            self.camera_elevation = max(-89, min(89, self.camera_elevation))"""

content = content.replace(old_mouse, new_mouse)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("✓ Fixed coordinate system and orbit camera!")
print("  - Character now Y-up (standing upright)")
print("  - Camera orbits naturally around character")
print("  - Mouse left-drag to orbit, right-drag to pan, wheel to zoom")
