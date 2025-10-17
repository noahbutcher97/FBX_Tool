"""
OpenGL 3D Skeleton Viewer

Real-time interactive 3D visualization using PyQt6 and OpenGL.
Features camera controls, animation playback, and shader-based rendering.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import fbx
from analysis_modules.utils import get_animation_info, build_bone_hierarchy


class SkeletonGLWidget(QOpenGLWidget):
    """
    OpenGL widget for rendering 3D skeleton.
    """

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.anim_info = get_animation_info(scene)
        self.hierarchy = build_bone_hierarchy(scene)

        # Animation data
        self.current_frame = 0
        self.total_frames = 0
        self.bone_transforms = {}

        # Camera settings
        self.camera_distance = 200.0
        self.camera_rotation_x = 30.0
        self.camera_rotation_y = 45.0
        self.camera_target = np.array([0.0, 0.0, 0.0])

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_button = None

        # Extract animation data
        self._extract_transforms()

    def _extract_transforms(self):
        """Extract bone transforms for all frames."""
        transforms = {}

        current = self.anim_info['start']
        frame_idx = 0

        while current <= self.anim_info['stop']:
            t = fbx.FbxTime()
            t.SetSecondDouble(current)

            for bone_name in self.hierarchy.keys():
                node = self.scene.FindNodeByName(bone_name)
                if not node:
                    continue

                global_transform = node.EvaluateGlobalTransform(t)
                translation = global_transform.GetT()
                position = np.array([translation[0], translation[1], translation[2]])

                if bone_name not in transforms:
                    transforms[bone_name] = []

                transforms[bone_name].append(position)

            current += self.anim_info['frame_time']
            frame_idx += 1

        self.bone_transforms = transforms
        self.total_frames = frame_idx

        # Calculate center for camera target
        if transforms:
            all_positions = []
            for positions in transforms.values():
                all_positions.extend(positions)
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
        cam_x = self.camera_target[0] + self.camera_distance * np.sin(np.radians(self.camera_rotation_y)) * np.cos(np.radians(self.camera_rotation_x))
        cam_y = self.camera_target[1] + self.camera_distance * np.sin(np.radians(self.camera_rotation_x))
        cam_z = self.camera_target[2] + self.camera_distance * np.cos(np.radians(self.camera_rotation_y)) * np.cos(np.radians(self.camera_rotation_x))

        gluLookAt(
            cam_x, cam_y, cam_z,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0.0, 1.0, 0.0
        )

        # Draw grid
        self._draw_grid()

        # Draw skeleton
        self._draw_skeleton()

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

    def _draw_skeleton(self):
        """Draw skeleton bones and joints."""
        if self.current_frame >= self.total_frames:
            return

        # Get current frame data
        joint_positions = {}
        for bone_name, positions in self.bone_transforms.items():
            if self.current_frame < len(positions):
                joint_positions[bone_name] = positions[self.current_frame]

        # Draw bones
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 1.0)  # Cyan
        glLineWidth(3.0)
        glBegin(GL_LINES)

        for child, parent in self.hierarchy.items():
            if parent and child in joint_positions and parent in joint_positions:
                child_pos = joint_positions[child]
                parent_pos = joint_positions[parent]

                glVertex3f(parent_pos[0], parent_pos[1], parent_pos[2])
                glVertex3f(child_pos[0], child_pos[1], child_pos[2])

        glEnd()

        # Draw joints as spheres
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0])  # Red
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.0, 0.0, 1.0])

        for position in joint_positions.values():
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
            # Rotate camera
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
            self.camera_rotation_x = max(-89, min(89, self.camera_rotation_x))

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


class SkeletonViewerWidget(QWidget):
    """
    Complete skeleton viewer widget with controls.
    """

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.gl_widget = SkeletonGLWidget(scene)
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)

        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # Add GL widget
        layout.addWidget(self.gl_widget, stretch=1)

        # Controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.play_button)

        # Frame label
        self.frame_label = QLabel(f"Frame: 0 / {self.gl_widget.total_frames}")
        controls_layout.addWidget(self.frame_label)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.gl_widget.total_frames - 1)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        controls_layout.addWidget(self.frame_slider, stretch=1)

        layout.addLayout(controls_layout)

        self.setLayout(layout)
        self.setWindowTitle("FBX Tool - 3D Skeleton Viewer")

    def _toggle_playback(self):
        """Toggle animation playback."""
        self.playing = not self.playing

        if self.playing:
            self.play_button.setText("Pause")
            fps = self.gl_widget.anim_info['frame_rate']
            self.timer.start(int(1000 / fps))
        else:
            self.play_button.setText("Play")
            self.timer.stop()

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


def launch_skeleton_viewer(scene):
    """
    Launch standalone skeleton viewer.

    Args:
        scene: FBX scene object
    """
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = SkeletonViewerWidget(scene)
    viewer.resize(1200, 800)
    viewer.show()

    if app:
        app.exec()
