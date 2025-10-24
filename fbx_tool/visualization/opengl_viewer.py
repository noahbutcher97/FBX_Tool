"""
OpenGL 3D Skeleton Viewer

Real-time interactive 3D visualization using PyQt6 and OpenGL.
Features camera controls, animation playback, and shader-based rendering.
"""

from datetime import datetime

import fbx
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.foot_contact_analysis import (
    calculate_adaptive_height_threshold,
    calculate_adaptive_velocity_threshold,
)
from fbx_tool.analysis.utils import build_bone_hierarchy, detect_full_coordinate_system


class SkeletonGLWidget(QOpenGLWidget):
    """
    OpenGL widget for rendering 3D skeleton.
    """

    def __init__(self, scene, parent=None, contact_data=None):
        super().__init__(parent)
        self.scene = scene
        self.anim_info = get_scene_metadata(scene)
        self.hierarchy = build_bone_hierarchy(scene)

        # Detect coordinate system using FBX SDK (simple check)
        # We'll use procedural detection after extracting some sample data
        axis_system = scene.GetGlobalSettings().GetAxisSystem()
        up_vector, _ = axis_system.GetUpVector()
        self.is_y_up = up_vector == fbx.FbxAxisSystem.EUpVector.eYAxis
        print(f"FBX Coordinate System (preliminary): {'Y-up' if self.is_y_up else 'Z-up (converting to Y-up)'}")

        # Animation data
        self.current_frame = 0
        self.total_frames = 0
        self.bone_transforms = {}

        # Coordinate system info (will be populated after transform extraction)
        self.coord_system = None  # Full coordinate system from procedural detection
        self.up_axis = 1  # Default to Y-up (index 1) until procedurally detected

        # Camera settings
        self.camera_distance = 400.0  # Start further back
        self.camera_azimuth = 45.0  # Look down from above
        self.camera_elevation = 20.0
        self.camera_target = np.array([0.0, 100.0, 0.0])  # Look at character height

        # Store default camera for reset
        self.default_camera = {
            "distance": 400.0,
            "azimuth": 45.0,
            "elevation": 20.0,
            "target": np.array([0.0, 100.0, 0.0]),
        }

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_button = None

        # Display options
        self.show_grid = True
        self.show_axes = True
        self.show_bone_names = False
        self.wireframe_mode = False
        self.show_foot_contacts = False  # Foot contact visualization toggle

        # Foot contact data from analysis
        # Format: {"bone_name": {"contact_segments": [(start, end), ...], "ground_height": float}}
        self.contact_data = contact_data or {}

        # Callbacks for parent widget communication
        self.on_frame_changed = None
        self.on_display_option_changed = None

        # Extract animation data
        self._extract_transforms()

    def _extract_transforms(self):
        """Extract bone transforms for all frames."""
        transforms = {}

        current = self.anim_info["start_time"]
        frame_idx = 0
        frame_time = 1.0 / self.anim_info["frame_rate"]

        while current <= self.anim_info["stop_time"]:
            t = fbx.FbxTime()
            t.SetSecondDouble(current)

            for bone_name in self.hierarchy.keys():
                node = self.scene.FindNodeByName(bone_name)
                if not node:
                    continue

                # Get global transform (includes rotation and translation)
                global_transform = node.EvaluateGlobalTransform(t)
                translation = global_transform.GetT()
                rotation = global_transform.GetQ()  # Get quaternion rotation

                # Convert FBX types to numpy arrays
                # Apply coordinate conversion only if needed
                if self.is_y_up:
                    # Already Y-up (Mixamo, Unity, etc.) - use as-is
                    position = np.array([translation[0], translation[1], translation[2]])
                    # Quaternion: (x, y, z, w)
                    quat = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])
                else:
                    # Z-up (Blender, 3ds Max, etc.) - convert to Y-up
                    # Swap Y and Z, negate new Z
                    position = np.array([translation[0], translation[2], -translation[1]])
                    # Also swap quaternion axes
                    quat = np.array([rotation[0], rotation[2], -rotation[1], rotation[3]])

                if bone_name not in transforms:
                    transforms[bone_name] = []

                # Store both position and rotation for each frame
                transforms[bone_name].append({"position": position, "rotation": quat})

            current += frame_time
            frame_idx += 1

        self.bone_transforms = transforms
        self.total_frames = frame_idx

        # Calculate center for camera target using first frame positions
        if transforms:
            all_positions = []
            for bone_data in transforms.values():
                if bone_data:
                    all_positions.append(bone_data[0]["position"])
            if all_positions:
                all_positions = np.array(all_positions)
                self.camera_target = np.mean(all_positions, axis=0)

        # PROCEDURAL COORDINATE SYSTEM DETECTION
        # Use root bone motion to detect coordinate system empirically
        # This replaces hardcoded Y-axis assumptions with data-driven detection
        if transforms:
            # Find root bone (bone with no parent)
            root_bone = None
            for child, parent in self.hierarchy.items():
                if not parent:
                    root_bone = child
                    break

            if root_bone and root_bone in transforms:
                # Extract positions and velocities for root bone
                root_positions = np.array([frame_data["position"] for frame_data in transforms[root_bone]])
                root_velocities = np.diff(root_positions, axis=0)

                # Detect full coordinate system from scene metadata + empirical data
                self.coord_system = detect_full_coordinate_system(self.scene, root_positions, root_velocities)
                self.up_axis = self.coord_system["up_axis"]

                print(
                    f"  [PROCEDURAL] Coordinate system detected: up_axis={self.up_axis} "
                    f"(0=X, 1=Y, 2=Z), confidence={self.coord_system['confidence']:.2f}"
                )

    def initializeGL(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.15, 1.0)

        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])

    def resizeGL(self, w, h):
        """Handle window resize."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h > 0 else 1, 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        # Orbit camera
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)

        cam_x = self.camera_target[0] + self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = self.camera_target[1] + self.camera_distance * np.sin(elevation_rad)
        cam_z = self.camera_target[2] + self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

        gluLookAt(
            cam_x, cam_y, cam_z, self.camera_target[0], self.camera_target[1], self.camera_target[2], 0.0, 1.0, 0.0
        )

        # Draw grid
        if self.show_grid:
            self._draw_grid()

        # Draw axes
        if self.show_axes:
            self._draw_axes()

        # Draw skeleton
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self._draw_skeleton()
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw foot contacts
        if self.show_foot_contacts:
            self._draw_foot_contacts()

    def _draw_grid(self):
        """Draw ground grid."""
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)

        grid_size = 500
        grid_step = 50

        for i in range(-grid_size, grid_size + grid_step, grid_step):
            # Lines parallel to X axis
            glVertex3f(-grid_size, 0, i)
            glVertex3f(grid_size, 0, i)
            # Lines parallel to Z axis
            glVertex3f(i, 0, -grid_size)
            glVertex3f(i, 0, grid_size)

        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_axes(self):
        """Draw coordinate axes (X=Red, Y=Green, Z=Blue)."""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)

        axis_length = 100.0

        # X axis (Red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)

        # Y axis (Green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)

        # Z axis (Blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)

        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_root_motion_trail(self, joint_transforms):
        """Draw root motion trajectory on ground plane."""
        # Find root bone (bone with no parent)
        root_bone = None
        for child, parent in self.hierarchy.items():
            if not parent:
                root_bone = child
                break

        if not root_bone or root_bone not in self.bone_transforms:
            return

        # Draw trail from start to current frame
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        # Draw trail line
        glColor3f(1.0, 1.0, 0.0)  # Yellow trail
        glBegin(GL_LINE_STRIP)

        for frame_idx in range(min(self.current_frame + 1, len(self.bone_transforms[root_bone]))):
            if frame_idx < len(self.bone_transforms[root_bone]):
                pos = self.bone_transforms[root_bone][frame_idx]["position"]
                # Project onto ground plane (Y=0)
                glVertex3f(pos[0], 0.1, pos[2])  # Slightly above ground

        glEnd()

        # Draw markers at key positions
        glPointSize(6.0)
        glColor3f(1.0, 0.5, 0.0)  # Orange markers
        glBegin(GL_POINTS)

        # Start position
        if len(self.bone_transforms[root_bone]) > 0:
            start_pos = self.bone_transforms[root_bone][0]["position"]
            glVertex3f(start_pos[0], 0.1, start_pos[2])

        # Current position
        if self.current_frame < len(self.bone_transforms[root_bone]):
            current_pos = self.bone_transforms[root_bone][self.current_frame]["position"]
            glVertex3f(current_pos[0], 0.1, current_pos[2])

        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_skeleton(self):
        """Draw skeleton bones and joints."""
        if self.current_frame >= self.total_frames:
            return

        # Get current frame data
        joint_transforms = {}
        for bone_name, transforms in self.bone_transforms.items():
            if self.current_frame < len(transforms):
                joint_transforms[bone_name] = transforms[self.current_frame]

        # Draw bones
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 1.0)  # Cyan
        glLineWidth(3.0)
        glBegin(GL_LINES)

        for child, parent in self.hierarchy.items():
            if child not in joint_transforms:
                continue

            child_pos = joint_transforms[child]["position"]

            if parent and parent in joint_transforms:
                # Regular bone: draw line from parent to child
                parent_pos = joint_transforms[parent]["position"]
                glVertex3f(parent_pos[0], parent_pos[1], parent_pos[2])
                glVertex3f(child_pos[0], child_pos[1], child_pos[2])
            elif not parent:
                # Root bone: draw line from ground (origin) to root
                glColor3f(1.0, 0.5, 0.0)  # Orange for root
                glVertex3f(child_pos[0], 0.0, child_pos[2])  # Ground point (Y=0)
                glVertex3f(child_pos[0], child_pos[1], child_pos[2])  # Root position
                glColor3f(0.0, 1.0, 1.0)  # Back to cyan

        glEnd()

        # Draw root motion trail on ground plane
        self._draw_root_motion_trail(joint_transforms)

        # Draw joints as spheres
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0])  # Red
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.0, 0.0, 1.0])

        for transform_data in joint_transforms.values():
            position = transform_data["position"]
            glPushMatrix()
            glTranslatef(position[0], position[1], position[2])

            # Draw sphere using GLU
            quadric = gluNewQuadric()
            gluSphere(quadric, 3.0, 16, 16)
            gluDeleteQuadric(quadric)

            glPopMatrix()

    def _get_bone_descendants(self, bone_name):
        """
        Get all descendant bones (children, grandchildren, etc.) of a given bone.

        Args:
            bone_name: Name of the parent bone

        Returns:
            list: List of descendant bone names (including the bone itself)
        """
        descendants = [bone_name]

        # Recursively collect all children
        def collect_children(parent):
            for child, child_parent in self.hierarchy.items():
                if child_parent == parent:
                    descendants.append(child)
                    collect_children(child)  # Recurse for grandchildren

        collect_children(bone_name)
        return descendants

    def _draw_foot_contacts(self):
        """
        Draw foot contact visualization with hierarchical lighting and ground contact lines.

        Features:
        - Uses actual contact analysis data when available (from contact_data parameter)
        - Falls back to adaptive height-based detection if no data provided
        - Lights up entire foot hierarchy (root + all children) during contact
        - Draws ground contact line indicator beneath foot showing contact footprint
        - Green bones = contact, Red bones = airborne
        """
        if self.current_frame >= self.total_frames:
            return

        # Get current frame joint transforms
        joint_transforms = {}
        for bone_name, transforms in self.bone_transforms.items():
            if self.current_frame < len(transforms):
                joint_transforms[bone_name] = transforms[self.current_frame]

        # Identify foot bones with priority (prefer "foot" over "ankle" over "toe")
        # This prevents showing multiple bones per foot
        foot_candidates = {"left": [], "right": []}

        for bone_name in joint_transforms.keys():
            name_lower = bone_name.lower()

            # Determine side
            side = None
            if any(kw in name_lower for kw in ["left", "l_", "_l", "l."]):
                side = "left"
            elif any(kw in name_lower for kw in ["right", "r_", "_r", "r."]):
                side = "right"

            # Check if it's a foot-related bone
            if "foot" in name_lower and "ball" not in name_lower:
                priority = 0  # Highest priority
            elif "ankle" in name_lower:
                priority = 1
            elif "toe" in name_lower and "tip" not in name_lower:
                priority = 2
            else:
                continue  # Not a foot bone

            if side:
                foot_candidates[side].append((priority, bone_name))

        # Select best bone for each side (lowest priority number = highest priority)
        foot_root_bones = []
        for side, candidates in foot_candidates.items():
            if candidates:
                candidates.sort()  # Sort by priority
                foot_root_bones.append(candidates[0][1])  # Take bone name with highest priority

        if not foot_root_bones:
            return  # No foot bones detected

        # Determine ground height ONCE (use contact_data if available, otherwise adaptive)
        # Cache it to avoid recalculation every frame
        if not hasattr(self, "_cached_ground_height"):
            if self.contact_data and any(bone in self.contact_data for bone in foot_root_bones):
                # Use ground height from contact analysis data
                ground_height = next(
                    (self.contact_data[bone]["ground_height"] for bone in foot_root_bones if bone in self.contact_data),
                    None,
                )
                if ground_height is None:
                    # Fall back to adaptive if data incomplete
                    ground_height = self._compute_adaptive_ground_height(foot_root_bones)
            else:
                # Adaptive ground height from animation data
                ground_height = self._compute_adaptive_ground_height(foot_root_bones)

            self._cached_ground_height = ground_height
            print(f"  [CACHE] Ground height calculated once: {ground_height:.2f}")
        else:
            ground_height = self._cached_ground_height

        # For each foot, calculate PROCEDURAL contact thresholds from actual data
        for foot_root in foot_root_bones:
            # Get all descendant bones (foot + toes + toe tips)
            foot_hierarchy = self._get_bone_descendants(foot_root)

            # Collect all height data for the ROOT FOOT BONE ONLY across the animation
            # This is used to calculate an adaptive threshold
            # NOTE: We use ONLY the root bone (heel) because child bones (toes) may have
            # invalid/stuck transforms (Y=0.00) that pollute the dataset
            # NOTE: Positions are already converted to Y-up in _extract_transforms()
            foot_heights_above_ground = []
            foot_velocity_magnitudes = []

            if foot_root in self.bone_transforms:
                for i, frame_data in enumerate(self.bone_transforms[foot_root]):
                    height_above_ground = frame_data["position"][1] - ground_height  # Y=1 (always up after conversion)
                    foot_heights_above_ground.append(height_above_ground)

                    # Calculate velocity magnitude (need at least 2 frames)
                    if i > 0:
                        prev_pos = self.bone_transforms[foot_root][i - 1]["position"]
                        curr_pos = frame_data["position"]
                        displacement = curr_pos - prev_pos
                        velocity = np.linalg.norm(displacement) * self.anim_info["frame_rate"]  # Convert to units/sec
                        foot_velocity_magnitudes.append(velocity)

            if not foot_heights_above_ground:
                continue  # No data for this foot

            # Calculate adaptive height threshold using foot_contact_analysis module
            # This automatically finds the separation between stance and aerial phases
            adaptive_height_threshold = calculate_adaptive_height_threshold(foot_heights_above_ground)
            contact_height_threshold = ground_height + adaptive_height_threshold

            # Calculate adaptive velocity threshold
            # This separates low-velocity stance from high-velocity aerial movement
            if len(foot_velocity_magnitudes) > 0:
                adaptive_velocity_threshold = calculate_adaptive_velocity_threshold(foot_velocity_magnitudes)
            else:
                adaptive_velocity_threshold = 10.0  # Fallback if only 1 frame

            # Debug: Print height and velocity distribution on frame 0 for ALL feet
            if self.current_frame == 0:
                sorted_heights = sorted(foot_heights_above_ground)
                sorted_velocities = sorted(foot_velocity_magnitudes) if foot_velocity_magnitudes else []
                print(f"  [DEBUG] Contact thresholds for {foot_root}:")
                print(f"    Height range: {sorted_heights[0]:.2f} to {sorted_heights[-1]:.2f}")
                print(f"    Height adaptive threshold: {adaptive_height_threshold:.2f}")
                print(f"    Height contact threshold: {contact_height_threshold:.2f}")
                if sorted_velocities:
                    print(f"    Velocity range: {sorted_velocities[0]:.2f} to {sorted_velocities[-1]:.2f} units/sec")
                    print(f"    Velocity adaptive threshold: {adaptive_velocity_threshold:.2f} units/sec")
                    print(f"    Velocity 50th percentile: {np.percentile(sorted_velocities, 50):.2f} units/sec")

            # Identify bones with stuck/invalid transforms
            # These are bones that never move significantly above ground level
            # Common in FBX files with partial animations or IK issues
            stuck_bones = set()
            for bone_name in foot_hierarchy:
                if bone_name in self.bone_transforms:
                    all_y_values = [frame_data["position"][1] for frame_data in self.bone_transforms[bone_name]]

                    # Debug: Print Y value stats for all feet
                    if self.current_frame == 0:
                        y_min, y_max, y_mean = min(all_y_values), max(all_y_values), np.mean(all_y_values)
                        y_range = y_max - y_min
                        always_below_ground = all(y <= ground_height + 1.0 for y in all_y_values)
                        print(
                            f"    [STUCK CHECK] {bone_name}: min={y_min:.4f}, max={y_max:.4f}, "
                            f"range={y_range:.4f}, always_below={always_below_ground}"
                        )

                    # A bone is "stuck" if it NEVER rises significantly above ground
                    # This catches bones stuck at/near ground level (Y=0-2) across all frames
                    if all(y <= ground_height + 1.0 for y in all_y_values):
                        stuck_bones.add(bone_name)

            # Find the LOWEST bone for CONTACT DETECTION (excluding stuck bones)
            # This determines WHEN contact occurs
            lowest_valid_bone = None
            lowest_valid_height = float("inf")

            for bone_name in foot_hierarchy:
                if bone_name in joint_transforms and bone_name not in stuck_bones:
                    bone_height = joint_transforms[bone_name]["position"][1]
                    if bone_height < lowest_valid_height:
                        lowest_valid_height = bone_height
                        lowest_valid_bone = bone_name

            # Calculate current frame velocity for the lowest valid bone
            current_velocity = 0.0
            if lowest_valid_bone and self.current_frame > 0 and lowest_valid_bone in self.bone_transforms:
                if self.current_frame < len(self.bone_transforms[lowest_valid_bone]):
                    prev_pos = self.bone_transforms[lowest_valid_bone][self.current_frame - 1]["position"]
                    curr_pos = joint_transforms[lowest_valid_bone]["position"]
                    displacement = curr_pos - prev_pos
                    current_velocity = np.linalg.norm(displacement) * self.anim_info["frame_rate"]  # units/sec

            # Contact requires BOTH height AND velocity criteria
            # This prevents false positives when foot is passing quickly through ground level
            height_criterion = lowest_valid_height <= contact_height_threshold
            velocity_criterion = current_velocity <= adaptive_velocity_threshold
            is_foot_in_contact = (height_criterion and velocity_criterion) if lowest_valid_bone else False

            # Find the LOWEST bone for SENSOR LINE POSITION (including stuck bones)
            # This determines WHERE to draw the line (actual ground level)
            lowest_bone_for_line = None
            lowest_height_for_line = float("inf")

            for bone_name in foot_hierarchy:
                if bone_name in joint_transforms:
                    bone_height = joint_transforms[bone_name]["position"][1]
                    if bone_height < lowest_height_for_line:
                        lowest_height_for_line = bone_height
                        lowest_bone_for_line = bone_name

            # Debug output for all feet on frame 0
            if self.current_frame == 0:
                print(f"\n  ═══ Contact Detection Debug ({foot_root}) ═══")
                print(f"    Ground height: {ground_height:.2f}")
                print(f"    Height threshold: {adaptive_height_threshold:.2f}")
                print(f"    Contact height threshold: {contact_height_threshold:.2f}")
                print(f"    Velocity threshold: {adaptive_velocity_threshold:.2f} units/sec")
                print(f"    Foot hierarchy: {foot_hierarchy}")
                print(f"    Stuck bones (excluded): {stuck_bones}")
                print(f"    Current frame {self.current_frame}:")
                for bone_name in foot_hierarchy:
                    if bone_name in joint_transforms:
                        pos = joint_transforms[bone_name]["position"]
                        is_lowest_valid = bone_name == lowest_valid_bone
                        is_lowest_line = bone_name == lowest_bone_for_line
                        markers = []
                        if is_lowest_valid:
                            markers.append("CONTACT CHECK")
                        if is_lowest_line:
                            markers.append("LINE POS")
                        marker_str = f" [{', '.join(markers)}]" if markers else ""
                        print(f"      {bone_name}: Y={pos[1]:.2f}{marker_str}")
                print(f"    Lowest valid bone: {lowest_valid_bone} at Y={lowest_valid_height:.2f}")
                print(f"    Current velocity: {current_velocity:.2f} units/sec")
                print(
                    f"    Height criterion: {height_criterion} (Y={lowest_valid_height:.2f} <= {contact_height_threshold:.2f})"
                )
                print(
                    f"    Velocity criterion: {velocity_criterion} (vel={current_velocity:.2f} <= {adaptive_velocity_threshold:.2f})"
                )
                print(f"    CONTACT: {'✓ YES (both criteria met)' if is_foot_in_contact else '✗ NO'}")
                if not is_foot_in_contact:
                    if not height_criterion:
                        print(f"      → Reason: Foot too high (height criterion failed)")
                    elif not velocity_criterion:
                        print(f"      → Reason: Moving too fast (velocity criterion failed)")
                print(f"  ═══════════════════════════════════════════\n")

            # Determine which bones to light up based on contact state
            # Only light up bones when the lowest bone is touching ground
            bones_to_light = foot_hierarchy if is_foot_in_contact else []

            # Draw spheres on all foot bones with individual contact states
            glEnable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            for bone_name in foot_hierarchy:
                if bone_name in joint_transforms:
                    position = joint_transforms[bone_name]["position"]

                    # Determine color based on foot contact state
                    if bone_name in bones_to_light:
                        # Green for contact
                        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 1.0, 0.0, 0.8])  # Bright green
                        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.4, 0.0, 0.8])
                    else:
                        # Red for airborne
                        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 0.6])  # Red
                        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.4, 0.0, 0.0, 0.6])

                    glPushMatrix()
                    glTranslatef(position[0], position[1], position[2])

                    quadric = gluNewQuadric()
                    gluSphere(quadric, 3.5, 16, 16)  # Slightly larger for visibility
                    gluDeleteQuadric(quadric)

                    glPopMatrix()

            # Draw ground contact line ONLY if foot is touching ground
            # Line is drawn at the LOWEST bone's height (actual contact point from stuck/valid bones)
            if is_foot_in_contact and lowest_bone_for_line:
                self._draw_ground_contact_line(foot_root, foot_hierarchy, joint_transforms, lowest_height_for_line)

        glDisable(GL_BLEND)

    def _compute_adaptive_ground_height(self, foot_bones):
        """
        Compute ground height adaptively from foot bone positions.

        Args:
            foot_bones: List of foot bone names

        Returns:
            float: Estimated ground height (5th percentile of foot Y positions)
        """
        all_foot_heights = []
        for bone_name in foot_bones:
            if bone_name in self.bone_transforms:
                for i in range(0, len(self.bone_transforms[bone_name]), 10):
                    pos = self.bone_transforms[bone_name][i]["position"]
                    all_foot_heights.append(pos[1])  # Y is typically up

        if not all_foot_heights:
            return 0.0

        # Ground height = 5th percentile (robust to noise)
        return np.percentile(all_foot_heights, 5)

    def _draw_ground_contact_line(self, foot_root, foot_hierarchy, joint_transforms, ground_height):
        """
        Draw a line on the ground beneath the foot showing contact footprint.

        The line is drawn at ground height and positioned to appear beneath the foot,
        extending from the heel to the toe tip to show the contact "footprint".

        Args:
            foot_root: Name of the root foot bone
            foot_hierarchy: List of all bones in foot (including root and descendants)
            joint_transforms: Dictionary of current frame transforms
            ground_height: Height of the ground plane
        """
        # Find the heel (most backward bone) and toe (most forward bone) in the foot hierarchy
        # This gives us the contact line endpoints regardless of foot orientation

        foot_bone_positions = []
        for bone_name in foot_hierarchy:
            if bone_name in joint_transforms:
                pos = joint_transforms[bone_name]["position"]
                foot_bone_positions.append((bone_name, pos))

        if len(foot_bone_positions) < 2:
            # Not enough bones to draw a meaningful line
            return

        # Find heel (backward-most) and toe (forward-most) based on Z-axis
        # For feet, the heel is typically negative Z, toe is positive Z (in local space)
        # We'll use the actual 3D distance from foot root to find extremes

        foot_root_pos = joint_transforms[foot_root]["position"]

        # Find the bone closest to heel (minimum distance from root in backward direction)
        # and farthest from heel (toe tip - maximum distance from root)
        min_bone = foot_root
        max_bone = foot_root
        min_dist = 0.0
        max_dist = 0.0

        for bone_name, pos in foot_bone_positions:
            # Calculate horizontal distance (XZ plane only, ignore Y)
            horizontal_dist = np.sqrt((pos[0] - foot_root_pos[0]) ** 2 + (pos[2] - foot_root_pos[2]) ** 2)

            # Find direction vector in XZ plane
            if horizontal_dist > 1e-6:  # Avoid division by zero
                # Calculate signed distance along the foot's forward direction
                # Use Z-axis as primary forward direction
                signed_dist = pos[2] - foot_root_pos[2]

                if signed_dist < min_dist:
                    min_dist = signed_dist
                    min_bone = bone_name
                if signed_dist > max_dist or horizontal_dist > max_dist:
                    max_dist = max(signed_dist, horizontal_dist)
                    max_bone = bone_name

        # Get positions for heel and toe
        heel_pos = joint_transforms[min_bone]["position"] if min_bone in joint_transforms else foot_root_pos
        toe_pos = joint_transforms[max_bone]["position"] if max_bone in joint_transforms else foot_root_pos

        # If we didn't find distinct heel/toe, fall back to root and farthest bone
        if min_bone == max_bone:
            # Find farthest bone by pure distance
            max_horizontal_dist = 0.0
            for bone_name, pos in foot_bone_positions:
                horizontal_dist = np.sqrt((pos[0] - foot_root_pos[0]) ** 2 + (pos[2] - foot_root_pos[2]) ** 2)
                if horizontal_dist > max_horizontal_dist:
                    max_horizontal_dist = horizontal_dist
                    max_bone = bone_name

            heel_pos = foot_root_pos
            toe_pos = joint_transforms[max_bone]["position"]

        # Draw line FLUSH with ground plane (at ground_height + slight offset for visibility)
        glDisable(GL_LIGHTING)
        glLineWidth(5.0)  # Thicker line for better visibility
        glColor3f(0.0, 1.0, 0.0)  # Bright green

        glBegin(GL_LINES)
        # Draw at ground height (slightly above to prevent z-fighting with grid)
        glVertex3f(heel_pos[0], ground_height + 0.1, heel_pos[2])  # Heel position (XZ), ground height (Y)
        glVertex3f(toe_pos[0], ground_height + 0.1, toe_pos[2])  # Toe position (XZ), ground height (Y)
        glEnd()

        glEnable(GL_LIGHTING)

    def set_frame(self, frame):
        """Set current animation frame."""
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update()

        # Notify parent widget of frame change
        if self.on_frame_changed:
            self.on_frame_changed(self.current_frame)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.last_mouse_pos = event.pos()
        self.mouse_button = event.button()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera control."""
        if self.last_mouse_pos is None:
            return

        dx = event.pos().x() - self.last_mouse_pos.x()
        dy = event.pos().y() - self.last_mouse_pos.y()

        if self.mouse_button == Qt.MouseButton.LeftButton:
            # Orbit camera
            self.camera_azimuth += dx * 0.5
            self.camera_elevation -= dy * 0.5
            self.camera_elevation = max(-89, min(89, self.camera_elevation))

        elif self.mouse_button == Qt.MouseButton.RightButton:
            # Pan camera
            move_speed = 0.5
            self.camera_target[0] -= dx * move_speed
            self.camera_target[1] += dy * move_speed

        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.last_mouse_pos = None
        self.mouse_button = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.1
        self.camera_distance = max(10, min(1000, self.camera_distance))
        self.update()

    def keyPressEvent(self, event):
        """Handle keyboard input."""
        from PyQt6.QtCore import Qt

        key = event.key()

        # Frame navigation
        if key == Qt.Key.Key_Left:
            self.set_frame(max(0, self.current_frame - 1))
        elif key == Qt.Key.Key_Right:
            self.set_frame(min(self.total_frames - 1, self.current_frame + 1))
        elif key == Qt.Key.Key_Home:
            self.set_frame(0)
        elif key == Qt.Key.Key_End:
            self.set_frame(self.total_frames - 1)

        # Camera controls
        elif key == Qt.Key.Key_R:
            self.reset_camera()
        elif key == Qt.Key.Key_F:
            self.set_camera_preset("front")
        elif key == Qt.Key.Key_S:
            self.set_camera_preset("side")
        elif key == Qt.Key.Key_T:
            self.set_camera_preset("top")

        # Display toggles
        elif key == Qt.Key.Key_G:
            self.show_grid = not self.show_grid
            self.update()
        elif key == Qt.Key.Key_A:
            self.show_axes = not self.show_axes
            self.update()
        elif key == Qt.Key.Key_N:
            self.show_bone_names = not self.show_bone_names
            self.update()
        elif key == Qt.Key.Key_W:
            self.wireframe_mode = not self.wireframe_mode
            self.update()
        elif key == Qt.Key.Key_C:
            self.show_foot_contacts = not self.show_foot_contacts
            self.update()
            # Notify parent to sync checkbox state
            if self.on_display_option_changed:
                self.on_display_option_changed("foot_contacts", self.show_foot_contacts)

    def reset_camera(self):
        """Reset camera to default position."""
        self.camera_distance = self.default_camera["distance"]
        self.camera_azimuth = self.default_camera["azimuth"]
        self.camera_elevation = self.default_camera["elevation"]
        self.camera_target = self.default_camera["target"].copy()
        self.update()

    def set_camera_preset(self, preset):
        """Set camera to predefined viewpoint."""
        if preset == "front":
            self.camera_azimuth = 0.0
            self.camera_elevation = 0.0
        elif preset == "side":
            self.camera_azimuth = 90.0
            self.camera_elevation = 0.0
        elif preset == "top":
            self.camera_azimuth = 0.0
            self.camera_elevation = 89.0
        elif preset == "back":
            self.camera_azimuth = 180.0
            self.camera_elevation = 0.0
        self.update()

    def capture_screenshot(self):
        """Capture current OpenGL framebuffer as QImage."""
        # Force a render
        self.update()
        self.makeCurrent()

        # Get framebuffer dimensions
        width = self.width()
        height = self.height()

        # Read pixels from framebuffer
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

        # Create QImage from pixel data
        image = QImage(pixels, width, height, QImage.Format.Format_RGBA8888)

        # Flip vertically (OpenGL origin is bottom-left, QImage is top-left)
        image = image.mirrored(False, True)

        self.doneCurrent()
        return image


class SkeletonViewerWidget(QWidget):
    """
    Complete skeleton viewer widget with controls.
    """

    def __init__(self, scene, parent=None, fbx_files=None, scene_manager=None):
        super().__init__(parent)
        self.scene = scene

        # Scene manager for reference-counted lifecycle
        from fbx_tool.analysis.scene_manager import get_scene_manager

        self.scene_manager = scene_manager if scene_manager else get_scene_manager()

        # Multiple file support
        self.fbx_files = fbx_files if fbx_files else []
        self.current_file_index = 0
        self.scene_refs = {}  # {index: FBXSceneReference} - visualizer holds refs

        # Get reference for initial scene (file at index 0)
        if self.fbx_files:
            self.scene_refs[0] = self.scene_manager.get_scene(self.fbx_files[0])

        self.gl_widget = SkeletonGLWidget(scene)
        self.playing = False
        self.playback_speed = 1.0  # Normal speed
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)

        # Visualization preferences (persistent across file switches)
        self.viz_prefs = {
            "grid": True,
            "axes": True,
            "wireframe": False,
            "foot_contacts": False,
        }

        # Connect GL widget callbacks
        self.gl_widget.on_frame_changed = self._on_gl_frame_changed
        self.gl_widget.on_display_option_changed = self._on_display_option_changed

        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # File switcher (if multiple files)
        if len(self.fbx_files) > 1:
            file_switcher_group = QGroupBox("Animation Files")
            file_switcher_layout = QHBoxLayout()

            self.prev_file_btn = QPushButton("◀ Previous")
            self.prev_file_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent stealing focus
            self.prev_file_btn.clicked.connect(self._load_previous_file)
            file_switcher_layout.addWidget(self.prev_file_btn)

            self.file_label = QLabel(self._get_current_filename())
            self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.file_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
            file_switcher_layout.addWidget(self.file_label, stretch=1)

            self.next_file_btn = QPushButton("Next ▶")
            self.next_file_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent stealing focus
            self.next_file_btn.clicked.connect(self._load_next_file)
            file_switcher_layout.addWidget(self.next_file_btn)

            file_switcher_group.setLayout(file_switcher_layout)
            layout.addWidget(file_switcher_group)

        # Add GL widget
        layout.addWidget(self.gl_widget, stretch=1)

        # Playback Controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent stealing focus from spacebar
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setMinimumWidth(80)
        playback_layout.addWidget(self.play_button)

        # Frame label
        self.frame_label = QLabel(f"Frame: 0 / {self.gl_widget.total_frames}")
        self.frame_label.setMinimumWidth(120)
        playback_layout.addWidget(self.frame_label)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.gl_widget.total_frames - 1)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        playback_layout.addWidget(self.frame_slider, stretch=1)

        # Speed label
        playback_layout.addWidget(QLabel("Speed:"))

        # Speed slider
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)  # 0.1x speed
        self.speed_slider.setMaximum(40)  # 4.0x speed
        self.speed_slider.setValue(10)  # 1.0x speed (normal)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        playback_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(40)
        playback_layout.addWidget(self.speed_label)

        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)

        # Display Options and Camera Presets
        options_layout = QHBoxLayout()

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout()

        self.grid_checkbox = QCheckBox("Grid (G)")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(lambda: self._toggle_display("grid"))
        display_layout.addWidget(self.grid_checkbox, 0, 0)

        self.axes_checkbox = QCheckBox("Axes (A)")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.stateChanged.connect(lambda: self._toggle_display("axes"))
        display_layout.addWidget(self.axes_checkbox, 0, 1)

        self.wireframe_checkbox = QCheckBox("Wireframe (W)")
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.stateChanged.connect(lambda: self._toggle_display("wireframe"))
        display_layout.addWidget(self.wireframe_checkbox, 1, 0)

        self.foot_contacts_checkbox = QCheckBox("Foot Contacts (C)")
        self.foot_contacts_checkbox.setChecked(False)
        self.foot_contacts_checkbox.stateChanged.connect(lambda: self._toggle_display("foot_contacts"))
        display_layout.addWidget(self.foot_contacts_checkbox, 1, 1)

        display_group.setLayout(display_layout)
        options_layout.addWidget(display_group)

        # Camera presets group
        camera_group = QGroupBox("Camera Presets")
        camera_layout = QGridLayout()

        front_btn = QPushButton("Front (F)")
        front_btn.clicked.connect(lambda: self.gl_widget.set_camera_preset("front"))
        camera_layout.addWidget(front_btn, 0, 0)

        side_btn = QPushButton("Side (S)")
        side_btn.clicked.connect(lambda: self.gl_widget.set_camera_preset("side"))
        camera_layout.addWidget(side_btn, 0, 1)

        top_btn = QPushButton("Top (T)")
        top_btn.clicked.connect(lambda: self.gl_widget.set_camera_preset("top"))
        camera_layout.addWidget(top_btn, 1, 0)

        reset_btn = QPushButton("Reset (R)")
        reset_btn.clicked.connect(self.gl_widget.reset_camera)
        camera_layout.addWidget(reset_btn, 1, 1)

        camera_group.setLayout(camera_layout)
        options_layout.addWidget(camera_group)

        # Screenshot group
        screenshot_group = QGroupBox("Screenshot")
        screenshot_layout = QVBoxLayout()

        screenshot_btn = QPushButton("📷 Capture Screenshot")
        screenshot_btn.clicked.connect(self._save_screenshot)
        screenshot_layout.addWidget(screenshot_btn)

        screenshot_group.setLayout(screenshot_layout)
        options_layout.addWidget(screenshot_group)

        # Keyboard shortcuts info
        shortcuts_group = QGroupBox("Keyboard Shortcuts")
        shortcuts_layout = QVBoxLayout()
        shortcuts_text = QLabel(
            "Arrow Keys: Navigate frames\n"
            "Home/End: First/Last frame\n"
            "P: Save screenshot\n"
            "Left Mouse: Rotate camera\n"
            "Right Mouse: Pan camera\n"
            "Mouse Wheel: Zoom"
        )
        shortcuts_text.setStyleSheet("font-size: 9pt;")
        shortcuts_layout.addWidget(shortcuts_text)
        shortcuts_group.setLayout(shortcuts_layout)
        options_layout.addWidget(shortcuts_group, stretch=1)

        layout.addLayout(options_layout)

        self.setLayout(layout)
        self.setWindowTitle("FBX Tool - 3D Skeleton Viewer")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _toggle_playback(self):
        """Toggle animation playback."""
        self.playing = not self.playing

        if self.playing:
            self.play_button.setText("Pause")
            fps = self.gl_widget.anim_info["frame_rate"] * self.playback_speed
            self.timer.start(int(1000 / fps))
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def _on_speed_change(self, value):
        """Handle playback speed change."""
        self.playback_speed = value / 10.0  # Convert slider value to speed multiplier
        self.speed_label.setText(f"{self.playback_speed:.1f}x")

        # Update timer if playing
        if self.playing:
            fps = self.gl_widget.anim_info["frame_rate"] * self.playback_speed
            self.timer.setInterval(int(1000 / fps))

    def _toggle_display(self, option):
        """Toggle display options and persist preference."""
        if option == "grid":
            self.gl_widget.show_grid = self.grid_checkbox.isChecked()
            self.viz_prefs["grid"] = self.grid_checkbox.isChecked()
        elif option == "axes":
            self.gl_widget.show_axes = self.axes_checkbox.isChecked()
            self.viz_prefs["axes"] = self.axes_checkbox.isChecked()
        elif option == "wireframe":
            self.gl_widget.wireframe_mode = self.wireframe_checkbox.isChecked()
            self.viz_prefs["wireframe"] = self.wireframe_checkbox.isChecked()
        elif option == "foot_contacts":
            self.gl_widget.show_foot_contacts = self.foot_contacts_checkbox.isChecked()
            self.viz_prefs["foot_contacts"] = self.foot_contacts_checkbox.isChecked()
        self.gl_widget.update()

    def _on_display_option_changed(self, option, value):
        """Handle display option changes from GL widget (e.g., keyboard shortcuts)."""
        # Update checkbox to match GL widget state
        if option == "foot_contacts":
            # Block signals to prevent circular updates
            self.foot_contacts_checkbox.blockSignals(True)
            self.foot_contacts_checkbox.setChecked(value)
            self.foot_contacts_checkbox.blockSignals(False)
            # Update preference
            self.viz_prefs["foot_contacts"] = value

    def keyPressEvent(self, event):
        """Forward keyboard events to GL widget."""
        if event.key() == Qt.Key.Key_Space:
            self._toggle_playback()
        elif event.key() == Qt.Key.Key_P:
            # P for Photo/Picture
            self._save_screenshot()
        else:
            self.gl_widget.keyPressEvent(event)
        super().keyPressEvent(event)

    def _advance_frame(self):
        """Advance to next frame."""
        current = self.gl_widget.current_frame
        next_frame = (current + 1) % self.gl_widget.total_frames

        self.gl_widget.set_frame(next_frame)
        self.frame_slider.setValue(next_frame)
        self.frame_label.setText(f"Frame: {next_frame} / {self.gl_widget.total_frames}")

    def _on_slider_change(self, value):
        """Handle slider change."""
        self.gl_widget.set_frame(value)
        self.frame_label.setText(f"Frame: {value} / {self.gl_widget.total_frames}")

    def _on_gl_frame_changed(self, frame):
        """Handle frame changes from GL widget (e.g., keyboard navigation)."""
        # Block signals to prevent circular updates
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {frame} / {self.gl_widget.total_frames}")

    def _save_screenshot(self):
        """Capture and save screenshot with smart defaults."""
        import os

        # Get current animation name
        if self.fbx_files and self.current_file_index < len(self.fbx_files):
            current_file = self.fbx_files[self.current_file_index]
            animation_name = os.path.splitext(os.path.basename(current_file))[0]
        else:
            animation_name = "animation"

        # Build screenshot directory: output/[animation_name]/screenshots/
        screenshot_dir = os.path.join("output", animation_name, "screenshots")

        # Create directory if it doesn't exist
        os.makedirs(screenshot_dir, exist_ok=True)

        # Generate default filename: [animation_name]_Frame[x]_Screenshot.png
        frame_num = self.gl_widget.current_frame
        default_filename = f"{animation_name}_Frame{frame_num:04d}_Screenshot.png"

        # Full default path
        default_path = os.path.join(screenshot_dir, default_filename)

        # Open save dialog starting in the screenshot directory
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default_path, "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if filename:
            # Capture screenshot
            image = self.gl_widget.capture_screenshot()

            # Save to file
            if image.save(filename):
                print(f"✓ Screenshot saved: {filename}")
            else:
                print(f"✗ Failed to save screenshot: {filename}")

    def _get_current_filename(self):
        """Get current file name with index."""
        if self.fbx_files:
            import os

            filename = os.path.basename(self.fbx_files[self.current_file_index])
            return f"[{self.current_file_index + 1}/{len(self.fbx_files)}] {filename}"
        return "No file"

    def _load_previous_file(self):
        """Load previous file in list."""
        if len(self.fbx_files) <= 1:
            return

        self.current_file_index = (self.current_file_index - 1) % len(self.fbx_files)
        self._switch_to_file(self.current_file_index)

    def _load_next_file(self):
        """Load next file in list."""
        if len(self.fbx_files) <= 1:
            return

        self.current_file_index = (self.current_file_index + 1) % len(self.fbx_files)
        self._switch_to_file(self.current_file_index)

    def _switch_to_file(self, index):
        """Switch to a different FBX file using scene manager with smart caching."""
        # Stop playback
        if self.playing:
            self._toggle_playback()

        # SMART CACHING: Keep only current + previous + next files cached
        # This prevents memory bloat when switching through large batches
        files_to_keep = {index}
        if index > 0:
            files_to_keep.add(index - 1)  # Previous file for quick back navigation
        if index < len(self.fbx_files) - 1:
            files_to_keep.add(index + 1)  # Next file for quick forward navigation

        # Release references to files not in the keep set
        for file_index in list(self.scene_refs.keys()):
            if file_index not in files_to_keep:
                self.scene_refs[file_index].release()
                del self.scene_refs[file_index]
                print(f"📉 Released scene cache for file {file_index} (smart caching)")

        # Get scene reference from scene manager (loads if needed, or cache hit if GUI has it)
        if index not in self.scene_refs:
            print(f"Loading {self.fbx_files[index]}...")
            self.scene_refs[index] = self.scene_manager.get_scene(self.fbx_files[index])

        # Get scene from reference
        self.scene = self.scene_refs[index].scene

        # Recreate GL widget with new scene
        old_widget = self.gl_widget
        self.gl_widget = SkeletonGLWidget(self.scene)
        self.gl_widget.on_frame_changed = self._on_gl_frame_changed
        self.gl_widget.on_display_option_changed = self._on_display_option_changed

        # Apply persistent visualization preferences
        self.gl_widget.show_grid = self.viz_prefs["grid"]
        self.gl_widget.show_axes = self.viz_prefs["axes"]
        self.gl_widget.wireframe_mode = self.viz_prefs["wireframe"]
        self.gl_widget.show_foot_contacts = self.viz_prefs["foot_contacts"]

        # Update UI checkboxes to match preferences
        self.grid_checkbox.setChecked(self.viz_prefs["grid"])
        self.axes_checkbox.setChecked(self.viz_prefs["axes"])
        self.wireframe_checkbox.setChecked(self.viz_prefs["wireframe"])
        self.foot_contacts_checkbox.setChecked(self.viz_prefs["foot_contacts"])

        # Replace widget in layout
        layout = self.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() == old_widget:
                layout.replaceWidget(old_widget, self.gl_widget)
                old_widget.deleteLater()
                break

        # Reset controls
        self.frame_slider.setMaximum(self.gl_widget.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame: 0 / {self.gl_widget.total_frames}")

        # Update file label
        if hasattr(self, "file_label"):
            self.file_label.setText(self._get_current_filename())

        print(f"Switched to: {self._get_current_filename()}")

    def closeEvent(self, event):
        """Handle window close - release all scene references."""
        # Release all scene references held by visualizer
        for index, scene_ref in self.scene_refs.items():
            scene_ref.release()
            print(f"Visualizer released scene reference for file {index}")
        self.scene_refs.clear()
        event.accept()


def launch_skeleton_viewer(scene, fbx_files=None, scene_manager=None):
    """
    Launch standalone skeleton viewer with scene manager support.

    Args:
        scene: FBX scene object
        fbx_files: Optional list of FBX file paths for multi-file switching
        scene_manager: Optional scene manager instance (uses global if None)

    Returns:
        SkeletonViewerWidget: The viewer widget instance
    """
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    created_app = False

    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    viewer = SkeletonViewerWidget(scene, fbx_files=fbx_files, scene_manager=scene_manager)
    viewer.resize(1200, 800)
    viewer.show()

    # Only start event loop if we created the application
    # If called from existing app, just return the viewer
    if created_app:
        app.exec()

    return viewer
