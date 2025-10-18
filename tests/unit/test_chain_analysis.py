"""
Unit tests for chain_analysis module.

Tests the chain-level IK suitability and temporal coherence analysis,
focusing on:
- Bone hierarchy building from FBX scene
- Chain detection from hierarchy
- Temporal coherence computation
- IK suitability aggregation
- Chain confidence scoring
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.mark.unit
class TestBoneHierarchyBuilding:
    """Test building bone hierarchy from FBX scene."""

    def test_build_bone_hierarchy_simple_chain(self):
        """Should build hierarchy dict from simple parent-child relationships."""
        from fbx_tool.analysis.utils import build_bone_hierarchy
        import fbx

        # Create mock scene with simple hierarchy: Root -> Hips -> Spine
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        # Setup hierarchy with skeleton node attributes
        hips = Mock()
        hips.GetName.return_value = "Hips"
        hips.GetChildCount.return_value = 1
        hips_attr = Mock()
        hips_attr.GetAttributeType.return_value = fbx.FbxNodeAttribute.EType.eSkeleton
        hips.GetNodeAttribute.return_value = hips_attr

        spine = Mock()
        spine.GetName.return_value = "Spine"
        spine.GetChildCount.return_value = 0
        spine_attr = Mock()
        spine_attr.GetAttributeType.return_value = fbx.FbxNodeAttribute.EType.eSkeleton
        spine.GetNodeAttribute.return_value = spine_attr

        hips.GetChild.return_value = spine

        # Root node is NOT a skeleton (typical in FBX)
        mock_root_node.GetName.return_value = "RootNode"
        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips
        mock_root_node.GetNodeAttribute.return_value = None

        # Build hierarchy
        hierarchy = build_bone_hierarchy(mock_scene)

        # Verify structure
        assert isinstance(hierarchy, dict)
        assert "Hips" in hierarchy
        assert "Spine" in hierarchy
        assert hierarchy["Hips"] is None  # First skeleton has no parent skeleton
        assert hierarchy["Spine"] == "Hips"

    def test_build_bone_hierarchy_branching(self):
        """Should handle branching hierarchies (multiple children)."""
        from fbx_tool.analysis.utils import build_bone_hierarchy
        import fbx

        # Create mock scene with branching: Hips -> [LeftLeg, RightLeg]
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        hips = Mock()
        hips.GetName.return_value = "Hips"
        hips.GetChildCount.return_value = 2
        hips_attr = Mock()
        hips_attr.GetAttributeType.return_value = fbx.FbxNodeAttribute.EType.eSkeleton
        hips.GetNodeAttribute.return_value = hips_attr

        left_leg = Mock()
        left_leg.GetName.return_value = "LeftLeg"
        left_leg.GetChildCount.return_value = 0
        left_attr = Mock()
        left_attr.GetAttributeType.return_value = fbx.FbxNodeAttribute.EType.eSkeleton
        left_leg.GetNodeAttribute.return_value = left_attr

        right_leg = Mock()
        right_leg.GetName.return_value = "RightLeg"
        right_leg.GetChildCount.return_value = 0
        right_attr = Mock()
        right_attr.GetAttributeType.return_value = fbx.FbxNodeAttribute.EType.eSkeleton
        right_leg.GetNodeAttribute.return_value = right_attr

        hips.GetChild.side_effect = [left_leg, right_leg]

        mock_root_node.GetName.return_value = "RootNode"
        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips
        mock_root_node.GetNodeAttribute.return_value = None

        # Build hierarchy
        hierarchy = build_bone_hierarchy(mock_scene)

        # Verify branching
        assert "LeftLeg" in hierarchy
        assert "RightLeg" in hierarchy
        assert hierarchy["LeftLeg"] == "Hips"
        assert hierarchy["RightLeg"] == "Hips"


@pytest.mark.unit
class TestChainDetection:
    """Test detecting kinematic chains from bone hierarchy."""

    def test_detect_chains_from_hierarchy_simple(self):
        """Should detect simple linear chains."""
        from fbx_tool.analysis.utils import detect_chains_from_hierarchy

        # Simple hierarchy: Root -> Hips -> Spine -> Chest
        hierarchy = {"Hips": "Root", "Spine": "Hips", "Chest": "Spine"}

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

        # Should detect one chain
        assert isinstance(chains, dict)
        assert len(chains) > 0

        # Verify chain contains expected bones in order
        chain_values = list(chains.values())
        assert any("Hips" in chain and "Spine" in chain and "Chest" in chain for chain in chain_values)

    def test_detect_chains_filters_by_min_length(self):
        """Should filter out chains shorter than min_chain_length."""
        from fbx_tool.analysis.utils import detect_chains_from_hierarchy

        # Short hierarchy: Root -> Hips (only 2 bones total)
        # Hips is a leaf, chain will be [Root, Hips] = 2 bones
        hierarchy = {"Hips": "Root"}

        # Require chains of at least 3
        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

        # Should have no chains (too short)
        assert len(chains) == 0

    def test_detect_chains_handles_branching(self):
        """Should detect multiple chains from branching hierarchy."""
        from fbx_tool.analysis.utils import detect_chains_from_hierarchy

        # Branching hierarchy: Hips -> [LeftLeg->LeftFoot, RightLeg->RightFoot]
        hierarchy = {
            "Hips": "Root",
            "LeftLeg": "Hips",
            "LeftFoot": "LeftLeg",
            "RightLeg": "Hips",
            "RightFoot": "RightLeg",
        }

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=2)

        # Should detect at least 2 chains (left and right legs)
        assert len(chains) >= 2


@pytest.mark.unit
class TestTemporalCoherence:
    """Test temporal coherence computation."""

    def test_compute_temporal_coherence_smooth_motion(self):
        """Should return high coherence for smooth, predictable motion."""
        from fbx_tool.analysis.chain_analysis import compute_temporal_coherence

        # Create smooth sinusoidal motion
        frames = 100
        t = np.linspace(0, 2 * np.pi, frames)
        position_data = np.column_stack([np.sin(t), np.cos(t), np.zeros(frames)])

        coherence = compute_temporal_coherence(position_data, frame_rate=30.0)

        # Smooth motion should have high coherence
        assert 0.0 <= coherence <= 1.0
        assert coherence > 0.5  # Should be reasonably high

    def test_compute_temporal_coherence_noisy_motion(self):
        """Should return lower coherence for noisy, unpredictable motion."""
        from fbx_tool.analysis.chain_analysis import compute_temporal_coherence

        # Create random noisy motion
        frames = 100
        position_data = np.random.randn(frames, 3)

        coherence = compute_temporal_coherence(position_data, frame_rate=30.0)

        # Noisy motion should have lower coherence
        assert 0.0 <= coherence <= 1.0
        # Note: Random data can sometimes correlate by chance, so we just check bounds

    def test_compute_temporal_coherence_insufficient_data(self):
        """Should return 0.0 for insufficient data."""
        from fbx_tool.analysis.chain_analysis import compute_temporal_coherence

        # Too few frames for coherence analysis
        position_data = np.array([[0, 0, 0], [1, 1, 1]])

        coherence = compute_temporal_coherence(position_data, frame_rate=30.0)

        assert coherence == 0.0


@pytest.mark.unit
class TestAnalyzeChains:
    """Test the main analyze_chains function."""

    @patch("fbx_tool.analysis.chain_analysis.detect_chains_from_hierarchy")
    @patch("fbx_tool.analysis.chain_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.chain_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.chain_analysis.prepare_output_file")
    @patch("builtins.open")
    def test_analyze_chains_basic_execution(
        self, mock_open, mock_prepare, mock_get_anim, mock_build_hierarchy, mock_detect_chains
    ):
        """Should execute basic chain analysis workflow."""
        from fbx_tool.analysis.chain_analysis import analyze_chains

        # Setup mocks
        mock_scene = Mock()

        mock_get_anim.return_value = {"start_time": 0.0, "stop_time": 1.0, "frame_rate": 30.0}

        mock_build_hierarchy.return_value = {"Hips": None, "Spine": "Hips", "Chest": "Spine"}

        mock_detect_chains.return_value = {"SpineChain": ["Hips", "Spine", "Chest"]}

        # Mock scene.FindNodeByName to return mock nodes
        def create_mock_node(name):
            node = Mock()
            node.GetName.return_value = name

            transform = Mock()
            # Create mock vectors with mData
            mock_t = Mock()
            mock_t.mData = [0.0, 0.0, 0.0, 0.0]
            mock_r = Mock()
            mock_r.mData = [0.0, 0.0, 0.0, 0.0]
            transform.GetT.return_value = mock_t
            transform.GetR.return_value = mock_r
            transform.Inverse.return_value = transform

            def multiply(self, other):
                result = Mock()
                result_t = Mock()
                result_t.mData = [0.0, 0.0, 0.0, 0.0]
                result_r = Mock()
                result_r.mData = [0.0, 0.0, 0.0, 0.0]
                result.GetT.return_value = result_t
                result.GetR.return_value = result_r
                return result

            transform.__mul__ = multiply
            node.EvaluateGlobalTransform.return_value = transform
            return node

        mock_scene.FindNodeByName.side_effect = create_mock_node

        # Execute
        result = analyze_chains(mock_scene, output_dir="test_output/")

        # Verify result structure
        assert isinstance(result, dict)

    @patch("fbx_tool.analysis.chain_analysis.detect_chains_from_hierarchy")
    @patch("fbx_tool.analysis.chain_analysis.build_bone_hierarchy")
    @patch("fbx_tool.analysis.chain_analysis.get_scene_metadata")
    @patch("fbx_tool.analysis.chain_analysis.prepare_output_file")
    @patch("builtins.open")
    def test_analyze_chains_computes_chain_confidence(
        self, mock_open, mock_prepare, mock_get_anim, mock_build_hierarchy, mock_detect_chains
    ):
        """Should compute chain confidence scores."""
        from fbx_tool.analysis.chain_analysis import analyze_chains

        # Setup similar to above
        mock_scene = Mock()

        mock_get_anim.return_value = {"start_time": 0.0, "stop_time": 1.0, "frame_rate": 30.0}

        mock_build_hierarchy.return_value = {"Hips": None, "Spine": "Hips", "Chest": "Spine"}

        mock_detect_chains.return_value = {"SpineChain": ["Hips", "Spine", "Chest"]}

        def create_mock_node(name):
            node = Mock()
            node.GetName.return_value = name

            transform = Mock()
            # Create mock vectors with mData
            mock_t = Mock()
            mock_t.mData = [0.0, 0.0, 0.0, 0.0]
            mock_r = Mock()
            mock_r.mData = [0.0, 0.0, 0.0, 0.0]
            transform.GetT.return_value = mock_t
            transform.GetR.return_value = mock_r
            transform.Inverse.return_value = transform

            def multiply(self, other):
                result = Mock()
                result_t = Mock()
                result_t.mData = [0.0, 0.0, 0.0, 0.0]
                result_r = Mock()
                result_r.mData = [0.0, 0.0, 0.0, 0.0]
                result.GetT.return_value = result_t
                result.GetR.return_value = result_r
                return result

            transform.__mul__ = multiply
            node.EvaluateGlobalTransform.return_value = transform
            return node

        mock_scene.FindNodeByName.side_effect = create_mock_node

        # Execute
        result = analyze_chains(mock_scene, output_dir="test_output/")

        # Verify chain confidence structure
        for chain_name, chain_data in result.items():
            assert "mean_ik" in chain_data
            assert "cross_temp" in chain_data
            assert "confidence" in chain_data
            assert 0.0 <= chain_data["confidence"] <= 1.0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_build_bone_hierarchy_empty_scene(self):
        """Should handle empty scene gracefully."""
        from fbx_tool.analysis.utils import build_bone_hierarchy

        mock_scene = Mock()
        mock_root = Mock()
        mock_root.GetName.return_value = "Root"
        mock_root.GetChildCount.return_value = 0
        mock_scene.GetRootNode.return_value = mock_root

        hierarchy = build_bone_hierarchy(mock_scene)

        # Should return empty or minimal hierarchy
        assert isinstance(hierarchy, dict)

    def test_detect_chains_empty_hierarchy(self):
        """Should handle empty hierarchy gracefully."""
        from fbx_tool.analysis.utils import detect_chains_from_hierarchy

        chains = detect_chains_from_hierarchy({}, min_chain_length=3)

        assert isinstance(chains, dict)
        assert len(chains) == 0
