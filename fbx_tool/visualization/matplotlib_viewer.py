"""
Skeleton Visualization Module

Provides 3D visualization of FBX skeleton animations.
Renders bones as lines, joints as spheres, and supports animation playback.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import fbx
from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.utils import build_bone_hierarchy


class SkeletonVisualizer:
    """
    3D skeleton visualization using matplotlib.

    Features:
    - Bones rendered as lines
    - Joints rendered as spheres
    - Animation playback
    - Interactive 3D camera
    - Frame scrubbing
    """

    def __init__(self, scene):
        """
        Initialize visualizer with FBX scene.

        Args:
            scene: FBX scene object
        """
        self.scene = scene
        self.anim_info = get_scene_metadata(scene)
        self.hierarchy = build_bone_hierarchy(scene)

        self.start = self.anim_info['start_time']
        self.stop = self.anim_info['stop_time']
        self.rate = self.anim_info['frame_rate']
        self.frame_time = 1.0 / self.rate

        # Cache all bone transforms
        self.bone_transforms = self._extract_transforms()

        # Visualization settings
        self.bone_color = 'cyan'
        self.joint_color = 'red'
        self.joint_size = 50
        self.bone_width = 2

    def _extract_transforms(self):
        """
        Extract global transforms for all bones across all frames.

        Returns:
            dict: {bone_name: [(frame, position), ...]}
        """
        transforms = {}

        current = self.start
        frame_idx = 0

        while current <= self.stop:
            t = fbx.FbxTime()
            t.SetSecondDouble(current)

            for bone_name in self.hierarchy.keys():
                node = self.scene.FindNodeByName(bone_name)
                if not node:
                    continue

                # Get global transform
                global_transform = node.EvaluateGlobalTransform(t)
                translation = global_transform.GetT()

                # Convert to numpy array
                position = np.array([translation[0], translation[1], translation[2]])

                if bone_name not in transforms:
                    transforms[bone_name] = []

                transforms[bone_name].append((frame_idx, position))

            current += self.frame_time
            frame_idx += 1

        return transforms

    def get_frame_data(self, frame_idx):
        """
        Get bone positions for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            tuple: (joint_positions, bone_connections)
                joint_positions: dict {bone_name: position}
                bone_connections: list [(parent_pos, child_pos), ...]
        """
        joint_positions = {}
        bone_connections = []

        for bone_name, frames in self.bone_transforms.items():
            if frame_idx < len(frames):
                _, position = frames[frame_idx]
                joint_positions[bone_name] = position

        # Build bone connections
        for child, parent in self.hierarchy.items():
            if parent and child in joint_positions and parent in joint_positions:
                child_pos = joint_positions[child]
                parent_pos = joint_positions[parent]
                bone_connections.append((parent_pos, child_pos))

        return joint_positions, bone_connections

    def visualize_frame(self, frame_idx=0, save_path=None):
        """
        Visualize a single frame.

        Args:
            frame_idx: Frame to visualize
            save_path: Optional path to save image
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        joint_positions, bone_connections = self.get_frame_data(frame_idx)

        # Draw bones
        for parent_pos, child_pos in bone_connections:
            ax.plot3D(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                color=self.bone_color,
                linewidth=self.bone_width
            )

        # Draw joints
        if joint_positions:
            positions = np.array(list(joint_positions.values()))
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=self.joint_color,
                s=self.joint_size,
                alpha=0.8
            )

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Skeleton - Frame {frame_idx}')

        # Set equal aspect ratio
        self._set_axes_equal(ax, joint_positions)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        else:
            plt.show()

        plt.close()

    def animate(self, start_frame=0, end_frame=None, interval=33, save_path=None):
        """
        Create animated visualization.

        Args:
            start_frame: Starting frame
            end_frame: Ending frame (None = all frames)
            interval: Milliseconds between frames (33 = ~30fps)
            save_path: Optional path to save animation (.mp4 or .gif)
        """
        if end_frame is None:
            end_frame = len(next(iter(self.bone_transforms.values())))

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Initialize plot elements
        bone_lines = []
        joint_scatter = None

        def init():
            """Initialize animation."""
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            return []

        def update(frame):
            """Update animation frame."""
            nonlocal joint_scatter, bone_lines

            # Clear previous frame
            for line in bone_lines:
                line.remove()
            bone_lines = []

            if joint_scatter:
                joint_scatter.remove()

            # Get frame data
            joint_positions, bone_connections = self.get_frame_data(frame)

            # Draw bones
            for parent_pos, child_pos in bone_connections:
                line = ax.plot3D(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]],
                    color=self.bone_color,
                    linewidth=self.bone_width
                )[0]
                bone_lines.append(line)

            # Draw joints
            if joint_positions:
                positions = np.array(list(joint_positions.values()))
                joint_scatter = ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color=self.joint_color,
                    s=self.joint_size,
                    alpha=0.8
                )

            ax.set_title(f'Skeleton Animation - Frame {frame}')

            # Set axes limits (keep constant for smooth animation)
            if frame == start_frame:
                self._set_axes_equal(ax, joint_positions)

            return bone_lines + [joint_scatter] if joint_scatter else bone_lines

        anim = FuncAnimation(
            fig,
            update,
            frames=range(start_frame, end_frame),
            init_func=init,
            interval=interval,
            blit=False
        )

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            else:
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            print(f"✓ Saved animation: {save_path}")
        else:
            plt.show()

        plt.close()

    def _set_axes_equal(self, ax, joint_positions):
        """
        Set equal aspect ratio for 3D plot.

        Args:
            ax: Matplotlib 3D axis
            joint_positions: Dict of joint positions
        """
        if not joint_positions:
            return

        positions = np.array(list(joint_positions.values()))

        # Get bounds
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_skeleton_frame(scene, frame_idx=0, save_path=None):
    """
    Quick function to visualize a single frame.

    Args:
        scene: FBX scene object
        frame_idx: Frame index to visualize
        save_path: Optional path to save image
    """
    visualizer = SkeletonVisualizer(scene)
    visualizer.visualize_frame(frame_idx, save_path)


def create_skeleton_animation(scene, output_path="skeleton_animation.mp4", 
                              start_frame=0, end_frame=None, fps=30):
    """
    Quick function to create animation.

    Args:
        scene: FBX scene object
        output_path: Path to save animation
        start_frame: Starting frame
        end_frame: Ending frame (None = all)
        fps: Frames per second
    """
    visualizer = SkeletonVisualizer(scene)
    interval = int(1000 / fps)
    visualizer.animate(start_frame, end_frame, interval, output_path)
