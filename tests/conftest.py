"""
Pytest configuration and shared fixtures for FBX Tool tests

Fixtures:
- mock_scene: Mock FBX scene with bones
- sample_positions: Sample bone position data
- sample_rotations: Sample bone rotation data
- temp_output_dir: Temporary directory for test outputs
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

# ========== Mock FBX SDK Objects ==========


@pytest.fixture
def mock_fbx_bone():
    """Create a mock FBX bone node."""
    bone = Mock()
    bone.GetName.return_value = "TestBone"
    bone.GetChildCount.return_value = 0

    # Mock node attribute
    attr = Mock()
    attr.GetAttributeType.return_value = Mock(eAttributeType=True)
    bone.GetNodeAttribute.return_value = attr

    # Mock transform
    transform = Mock()
    transform.GetT.return_value = [0.0, 0.0, 0.0]
    transform.GetR.return_value = [0.0, 0.0, 0.0]
    bone.EvaluateGlobalTransform.return_value = transform

    return bone


@pytest.fixture
def mock_scene():
    """Create a mock FBX scene with basic structure."""
    scene = Mock()

    # Mock root node
    root = Mock()
    root.GetChildCount.return_value = 0
    scene.GetRootNode.return_value = root

    # Mock time settings
    time_span = Mock()
    time_span.GetStart.return_value = Mock()

    global_settings = Mock()
    global_settings.GetTimeSpan.return_value = time_span
    scene.GetGlobalSettings.return_value = global_settings

    # Mock FBX time
    scene.FbxTime = Mock
    scene.FbxSkeleton = Mock()
    scene.FbxSkeleton.eAttributeType = "Skeleton"

    return scene


@pytest.fixture
def mock_bone_hierarchy():
    """Create a mock bone hierarchy (spine, arms, legs)."""
    bones = {}

    # Spine
    bones["hips"] = Mock(name="Hips")
    bones["spine"] = Mock(name="Spine")
    bones["chest"] = Mock(name="Chest")

    # Left arm
    bones["left_shoulder"] = Mock(name="LeftShoulder")
    bones["left_elbow"] = Mock(name="LeftElbow")
    bones["left_wrist"] = Mock(name="LeftWrist")

    # Right arm
    bones["right_shoulder"] = Mock(name="RightShoulder")
    bones["right_elbow"] = Mock(name="RightElbow")
    bones["right_wrist"] = Mock(name="RightWrist")

    # Left leg
    bones["left_hip"] = Mock(name="LeftHip")
    bones["left_knee"] = Mock(name="LeftKnee")
    bones["left_foot"] = Mock(name="LeftFoot")

    # Right leg
    bones["right_hip"] = Mock(name="RightHip")
    bones["right_knee"] = Mock(name="RightKnee")
    bones["right_foot"] = Mock(name="RightFoot")

    # Configure mock methods
    for bone_name, bone in bones.items():
        bone.GetName.return_value = bone_name
        attr = Mock()
        attr.GetAttributeType.return_value = Mock(eAttributeType=True)
        bone.GetNodeAttribute.return_value = attr

    return bones


# ========== Sample Data Fixtures ==========


@pytest.fixture
def sample_positions():
    """Generate sample position data for testing (walking motion)."""
    frames = 100
    t = np.linspace(0, 2 * np.pi, frames)

    # Simple sinusoidal walk pattern
    positions = np.zeros((frames, 3))
    positions[:, 0] = t  # X: forward movement
    positions[:, 1] = np.abs(np.sin(t * 4)) * 10 + 5  # Y: up/down (foot height)
    positions[:, 2] = np.sin(t * 2) * 2  # Z: side-to-side sway

    return positions


@pytest.fixture
def sample_rotations():
    """Generate sample rotation data for testing."""
    frames = 100
    t = np.linspace(0, 2 * np.pi, frames)

    # Simple rotation pattern
    rotations = np.zeros((frames, 3))
    rotations[:, 0] = np.sin(t) * 45  # X: pitch
    rotations[:, 1] = np.cos(t) * 30  # Y: yaw
    rotations[:, 2] = np.sin(t * 2) * 15  # Z: roll

    return rotations


@pytest.fixture
def sample_velocities():
    """Generate sample velocity data for testing."""
    frames = 100
    t = np.linspace(0, 2 * np.pi, frames)

    velocities = np.zeros((frames, 3))
    velocities[:, 0] = np.ones(frames) * 5  # Constant forward velocity
    velocities[:, 1] = np.cos(t * 4) * 10  # Vertical velocity
    velocities[:, 2] = np.cos(t * 2) * 2  # Lateral velocity

    return velocities


@pytest.fixture
def sample_frame_rate():
    """Standard frame rate for testing."""
    return 30.0


@pytest.fixture
def sample_total_frames():
    """Standard frame count for testing."""
    return 100


# ========== File System Fixtures ==========


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory that is cleaned up after test."""
    temp_dir = tempfile.mkdtemp(prefix="fbx_tool_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return [
        {"bone_name": "Bone1", "value1": 1.0, "value2": 2.0},
        {"bone_name": "Bone2", "value1": 3.0, "value2": 4.0},
        {"bone_name": "Bone3", "value1": 5.0, "value2": 6.0},
    ]


# ========== Test Markers ==========


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gui: GUI tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "fbx: Tests requiring FBX SDK")


# ========== Test Utilities ==========


@pytest.fixture
def assert_csv_exists():
    """Helper to assert CSV file exists and is valid."""

    def _assert_csv(filepath, min_rows=1):
        import csv

        assert Path(filepath).exists(), f"CSV file not found: {filepath}"

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= min_rows, f"Expected at least {min_rows} rows, got {len(rows)}"

        return rows

    return _assert_csv


@pytest.fixture
def numpy_arrays_equal():
    """Helper to compare numpy arrays with tolerance."""

    def _arrays_equal(arr1, arr2, rtol=1e-5, atol=1e-8):
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol)

    return _arrays_equal
