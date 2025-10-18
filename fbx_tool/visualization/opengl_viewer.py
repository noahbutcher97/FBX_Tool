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
from fbx_tool.analysis.utils import build_bone_hierarchy


class SkeletonGLWidget(QOpenGLWidget):
    """
    OpenGL widget for rendering 3D skeleton.
    """

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.anim_info = get_scene_metadata(scene)
        self.hierarchy = build_bone_hierarchy(scene)

        # Detect coordinate system
        axis_system = scene.GetGlobalSettings().GetAxisSystem()
        up_vector, _ = axis_system.GetUpVector()
        self.is_y_up = up_vector == fbx.FbxAxisSystem.EUpVector.eYAxis
        print(f"FBX Coordinate System: {'Y-up' if self.is_y_up else 'Z-up (converting to Y-up)'}")

        # Animation data
        self.current_frame = 0
        self.total_frames = 0
        self.bone_transforms = {}

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

        # Callback for frame changes (set by parent widget)
        self.on_frame_changed = None

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

        # Connect GL widget frame change callback
        self.gl_widget.on_frame_changed = self._on_gl_frame_changed

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
        """Toggle display options."""
        if option == "grid":
            self.gl_widget.show_grid = self.grid_checkbox.isChecked()
        elif option == "axes":
            self.gl_widget.show_axes = self.axes_checkbox.isChecked()
        elif option == "wireframe":
            self.gl_widget.wireframe_mode = self.wireframe_checkbox.isChecked()
        self.gl_widget.update()

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
