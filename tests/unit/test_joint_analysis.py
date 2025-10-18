"""
Unit tests for joint_analysis module.

Tests joint-level IK suitability analysis and rotation range computation.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestFbxVectorConversion:
    """Test FBX vector to numpy array conversion."""

    def test_fbx_vector_to_array_basic(self):
        """Test basic FbxVector4 to array conversion."""
        from fbx_tool.analysis.utils import fbx_vector_to_array

        # Create mock FbxVector4 with mData attribute
        mock_vec = Mock()
        mock_vec.mData = [1.0, 2.0, 3.0, 0.0]

        result = fbx_vector_to_array(mock_vec)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # FbxVector4 has 4 components
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 0.0])

    def test_fbx_vector_to_array_negative_values(self):
        """Test conversion with negative values."""
        from fbx_tool.analysis.utils import fbx_vector_to_array

        mock_vec = Mock()
        mock_vec.mData = [-5.5, 10.2, -3.7, 1.0]

        result = fbx_vector_to_array(mock_vec)

        np.testing.assert_array_almost_equal(result, [-5.5, 10.2, -3.7, 1.0])

    def test_fbx_vector_to_array_zero_values(self):
        """Test conversion with all zeros."""
        from fbx_tool.analysis.utils import fbx_vector_to_array

        mock_vec = Mock()
        mock_vec.mData = [0.0, 0.0, 0.0, 0.0]

        result = fbx_vector_to_array(mock_vec)

        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])

    def test_fbx_vector_to_array_without_mdata(self):
        """Test conversion for FbxDouble3 without mData attribute."""
        from fbx_tool.analysis.utils import fbx_vector_to_array

        # Create a mock that acts like a list for __getitem__ but has no mData
        class MockFbxDouble3:
            def __getitem__(self, index):
                return [1.5, 2.5, 3.5][index]

        mock_vec = MockFbxDouble3()

        result = fbx_vector_to_array(mock_vec)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1.5, 2.5, 3.5])


class TestJointIKSuitabilityComputation:
    """Test IK suitability score computation."""

    def test_ik_suitability_stable_joint(self):
        """Test IK suitability for a stable joint with minimal rotation."""
        # Joint with very small rotation variance (good for IK)
        rotation_data = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [-0.1, -0.1, -0.1], [0.0, 0.0, 0.0]]
        )

        # Compute IK metrics manually
        std_r = np.std(rotation_data, axis=0)
        stability = 1.0 / (1.0 + np.linalg.norm(std_r))

        # Stable joint should have high stability score (close to 1)
        assert stability > 0.9

    def test_ik_suitability_unstable_joint(self):
        """Test IK suitability for unstable joint with high variance."""
        # Joint with high rotation variance (bad for IK)
        rotation_data = np.array(
            [[0.0, 0.0, 0.0], [45.0, 30.0, 20.0], [-30.0, 50.0, -40.0], [60.0, -20.0, 35.0], [-50.0, 40.0, -30.0]]
        )

        std_r = np.std(rotation_data, axis=0)
        stability = 1.0 / (1.0 + np.linalg.norm(std_r))

        # Unstable joint should have low stability score
        assert stability < 0.5

    def test_ik_range_score_sufficient_range(self):
        """Test range score for joint with sufficient rotation range."""
        # Joint with good rotation range (not too much, not too little)
        rotation_data = np.array(
            [[0.0, 0.0, 0.0], [30.0, 20.0, 15.0], [60.0, 40.0, 30.0], [30.0, 20.0, 15.0], [0.0, 0.0, 0.0]]
        )

        min_r = np.min(rotation_data, axis=0)
        max_r = np.max(rotation_data, axis=0)
        rot_range = max_r - min_r
        total_range = np.sum(rot_range)

        # 60+40+30 = 130 degrees total range
        assert total_range == 130.0

        # Normalized range (130/540 ≈ 0.24)
        normalized_range = np.clip(total_range / 540.0, 0, 1)
        assert 0.2 < normalized_range < 0.3

    def test_ik_range_score_excessive_range(self):
        """Test range score for joint with excessive rotation range."""
        # Joint with excessive rotation (clamped to 1.0)
        rotation_data = np.array([[-180.0, -180.0, -180.0], [180.0, 180.0, 180.0]])

        min_r = np.min(rotation_data, axis=0)
        max_r = np.max(rotation_data, axis=0)
        rot_range = max_r - min_r
        total_range = np.sum(rot_range)

        # 360+360+360 = 1080 degrees (exceeds full range)
        assert total_range == 1080.0

        # Should be clamped to 1.0
        normalized_range = np.clip(total_range / 540.0, 0, 1)
        assert normalized_range == 1.0

    def test_ik_combined_score_weights(self):
        """Test that IK score uses correct weighting (60% stability, 40% range)."""
        stability = 0.8
        range_score = 0.5

        expected_ik_score = stability * 0.6 + range_score * 0.4

        assert expected_ik_score == pytest.approx(0.68, rel=1e-3)


class TestJointAnalysisEdgeCases:
    """Test edge cases in joint analysis."""

    def test_joint_with_zero_rotation(self):
        """Test joint that never rotates (locked joint)."""
        # Completely static joint
        rotation_data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        std_r = np.std(rotation_data, axis=0)
        stability = 1.0 / (1.0 + np.linalg.norm(std_r))

        # Zero variance = perfect stability
        assert stability == 1.0

        rot_range = np.max(rotation_data, axis=0) - np.min(rotation_data, axis=0)
        assert np.all(rot_range == 0)

    def test_joint_with_single_frame(self):
        """Test joint with only one frame of data."""
        rotation_data = np.array([[10.0, 20.0, 30.0]])

        std_r = np.std(rotation_data, axis=0)

        # Single frame = zero variance
        assert np.all(std_r == 0)

    def test_rotation_range_negative_to_positive(self):
        """Test rotation range calculation across negative and positive values."""
        rotation_data = np.array([[-45.0, -30.0, -15.0], [45.0, 30.0, 15.0]])

        min_r = np.min(rotation_data, axis=0)
        max_r = np.max(rotation_data, axis=0)
        rot_range = max_r - min_r

        np.testing.assert_array_equal(rot_range, [90.0, 60.0, 30.0])


class TestAnalyzeJointsFunction:
    """Test the main analyze_joints() function."""

    @patch("fbx_tool.analysis.joint_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.joint_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.joint_analysis.prepare_output_file")
    @patch("builtins.open", create=True)
    def test_analyze_joints_basic(self, mock_open, mock_prepare, mock_hierarchy, mock_anim_info):
        """Test basic joint analysis execution."""
        from fbx_tool.analysis.joint_analysis import analyze_joints

        # Mock animation info
        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 0.1, "frame_rate": 30.0}

        # Mock bone hierarchy
        mock_hierarchy.return_value = {"ChildBone": "ParentBone"}

        # Mock FBX scene
        mock_scene = MagicMock()
        mock_node = MagicMock()
        mock_parent_node = MagicMock()

        # Mock transforms with mData attribute for FBX vectors
        mock_transform = MagicMock()
        mock_t_vec = Mock()
        mock_t_vec.mData = [0.0, 1.0, 0.0, 0.0]
        mock_r_vec = Mock()
        mock_r_vec.mData = [10.0, 20.0, 30.0, 0.0]
        mock_transform.GetT.return_value = mock_t_vec
        mock_transform.GetR.return_value = mock_r_vec

        mock_node.EvaluateGlobalTransform.return_value = mock_transform
        mock_parent_node.EvaluateGlobalTransform.return_value = mock_transform

        mock_scene.FindNodeByName.side_effect = lambda name: mock_node if name == "ChildBone" else mock_parent_node

        # Execute
        result = analyze_joints(mock_scene, output_dir="test_output/")

        # Verify
        assert isinstance(result, dict)
        mock_prepare.assert_called_once()

    @patch("fbx_tool.analysis.joint_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.joint_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.joint_analysis.prepare_output_file")
    @patch("builtins.open", create=True)
    def test_analyze_joints_returns_summary(self, mock_open, mock_prepare, mock_hierarchy, mock_anim_info):
        """Test that analyze_joints returns correct summary structure."""
        from fbx_tool.analysis.joint_analysis import analyze_joints

        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 0.1, "frame_rate": 30.0}

        mock_hierarchy.return_value = {"Bone1": "Root", "Bone2": "Bone1"}

        mock_scene = MagicMock()
        mock_node = MagicMock()

        mock_transform = MagicMock()
        mock_t_vec = Mock()
        mock_t_vec.mData = [0.0, 0.0, 0.0, 0.0]
        mock_r_vec = Mock()
        mock_r_vec.mData = [0.0, 0.0, 0.0, 0.0]
        mock_transform.GetT.return_value = mock_t_vec
        mock_transform.GetR.return_value = mock_r_vec
        mock_transform.Inverse.return_value = mock_transform
        mock_transform.__mul__ = Mock(return_value=mock_transform)

        mock_node.EvaluateGlobalTransform.return_value = mock_transform
        mock_scene.FindNodeByName.return_value = mock_node

        result = analyze_joints(mock_scene)

        # Verify result structure
        assert isinstance(result, dict)

        # Each joint should have (stability, range_score, ik_score) tuple
        for joint_key, metrics in result.items():
            assert isinstance(joint_key, tuple)
            assert len(joint_key) == 2  # (parent, child)
            assert isinstance(metrics, tuple)
            assert len(metrics) == 3  # (stability, range_score, ik_score)

            stability, range_score, ik_score = metrics
            assert 0.0 <= stability <= 1.0
            assert 0.0 <= range_score <= 1.0
            assert 0.0 <= ik_score <= 1.0

    @patch("fbx_tool.analysis.joint_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.joint_analysis.build_bone_hierarchy")
    def test_analyze_joints_handles_missing_nodes(self, mock_hierarchy, mock_anim_info):
        """Test that analyze_joints gracefully handles missing nodes."""
        from fbx_tool.analysis.joint_analysis import analyze_joints

        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 0.033, "frame_rate": 30.0}

        mock_hierarchy.return_value = {"ExistingBone": "Root", "MissingBone": "Root"}

        mock_scene = MagicMock()

        # Only ExistingBone returns a valid node
        def find_node_side_effect(name):
            if name == "ExistingBone" or name == "Root":
                mock_node = MagicMock()
                mock_transform = MagicMock()
                mock_t_vec = Mock()
                mock_t_vec.mData = [0.0, 0.0, 0.0, 0.0]
                mock_r_vec = Mock()
                mock_r_vec.mData = [0.0, 0.0, 0.0, 0.0]
                mock_transform.GetT.return_value = mock_t_vec
                mock_transform.GetR.return_value = mock_r_vec
                mock_transform.Inverse.return_value = mock_transform
                mock_transform.__mul__ = Mock(return_value=mock_transform)
                mock_node.EvaluateGlobalTransform.return_value = mock_transform
                return mock_node
            return None  # MissingBone returns None

        mock_scene.FindNodeByName.side_effect = find_node_side_effect

        with patch("fbx_tool.analysis.joint_analysis.prepare_output_file"):
            with patch("builtins.open", create=True):
                result = analyze_joints(mock_scene)

        # Should only include ExistingBone, not MissingBone
        bone_names = [joint[1] for joint in result.keys()]
        assert "ExistingBone" in bone_names
        assert "MissingBone" not in bone_names


class TestJointAnalysisCSVOutput:
    """Test CSV output formatting."""

    @patch("fbx_tool.analysis.joint_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.joint_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.joint_analysis.prepare_output_file")
    def test_csv_has_correct_headers(self, mock_prepare, mock_hierarchy, mock_anim_info):
        """Test that CSV output has correct column headers."""
        import io
        from unittest.mock import mock_open

        from fbx_tool.analysis.joint_analysis import analyze_joints

        mock_anim_info.return_value = {"start_time": 0.0, "stop_time": 0.033, "frame_rate": 30.0}

        mock_hierarchy.return_value = {"Bone1": "Root"}

        mock_scene = MagicMock()
        mock_node = MagicMock()
        mock_transform = MagicMock()
        mock_t_vec = Mock()
        mock_t_vec.mData = [1.0, 2.0, 3.0, 0.0]
        mock_r_vec = Mock()
        mock_r_vec.mData = [10.0, 20.0, 30.0, 0.0]
        mock_transform.GetT.return_value = mock_t_vec
        mock_transform.GetR.return_value = mock_r_vec
        mock_transform.Inverse.return_value = mock_transform
        mock_transform.__mul__ = Mock(return_value=mock_transform)
        mock_node.EvaluateGlobalTransform.return_value = mock_transform
        mock_scene.FindNodeByName.return_value = mock_node

        # Capture CSV output using mock_open
        m = mock_open()

        with patch("builtins.open", m):
            analyze_joints(mock_scene, output_dir="test/")

        # Get the written content from mock
        written_content = "".join(call.args[0] for call in m().write.call_args_list if call.args)

        # Verify header row
        expected_headers = [
            "Parent",
            "Child",
            "MinRotX",
            "MaxRotX",
            "MinRotY",
            "MaxRotY",
            "MinRotZ",
            "MaxRotZ",
            "Stability",
            "RangeScore",
            "IKSuitability",
        ]

        assert all(header in written_content for header in expected_headers)
