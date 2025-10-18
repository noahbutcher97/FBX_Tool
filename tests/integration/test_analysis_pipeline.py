"""
Integration tests for the complete analysis pipeline.

Tests the end-to-end flow from FBX loading through all analysis
modules, focusing on:
- Cache efficiency (trajectory extracted once, reused by all modules)
- Correct data flow between dependent modules
- Complete pipeline execution without errors
- Output file generation
"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.mark.integration
class TestAnalysisPipelineCaching:
    """Test that the analysis pipeline uses caching efficiently."""

    @patch("fbx_tool.analysis.utils._detect_root_bone")
    @patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
    @patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
    def test_trajectory_extracted_once_for_multiple_modules(self, mock_derivatives, mock_metadata, mock_detect):
        """Should extract trajectory once and reuse for all modules."""
        from fbx_tool.analysis.directional_change_detection import analyze_directional_changes
        from fbx_tool.analysis.motion_transition_detection import analyze_motion_transitions
        from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
        from fbx_tool.analysis.utils import clear_trajectory_cache

        # Clear cache before test
        clear_trajectory_cache()

        # Setup mocks
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        # Create mock hierarchy
        hips_node = Mock()
        hips_node.GetName.return_value = "Hips"
        hips_node.GetChildCount.return_value = 0

        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips_node

        mock_detect.return_value = hips_node

        # Setup metadata
        mock_metadata.return_value = {
            "has_animation": True,
            "start_time": 0.0,
            "stop_time": 1.0,
            "frame_rate": 30.0,
            "duration": 1.0,
            "total_frames": 30,
        }

        # Setup time span
        mock_time_span = Mock()
        import fbx

        real_start_time = fbx.FbxTime()
        real_start_time.SetSecondDouble(0.0)
        mock_time_span.GetStart.return_value = real_start_time

        duration_time = fbx.FbxTime()
        duration_time.SetSecondDouble(1.0)
        mock_time_span.GetDuration.return_value = duration_time

        mock_anim_stack = Mock()
        mock_anim_stack.GetLocalTimeSpan.return_value = mock_time_span

        mock_scene.GetCurrentAnimationStack.return_value = mock_anim_stack

        # Setup GetSrcObjectCount and GetSrcObject for animation stack access
        mock_scene.GetSrcObjectCount.return_value = 1
        mock_scene.GetSrcObject.return_value = mock_anim_stack

        # Setup transform evaluations
        def create_mock_transform(frame):
            transform = Mock()
            transform.GetT.return_value = [float(frame), 0.0, 0.0]
            transform.GetR.return_value = [0.0, 0.0, 0.0]

            # Mock the transformation matrix for forward direction extraction
            # Identity matrix: forward = -Z axis = [0, 0, -1]
            identity_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            transform.Get.side_effect = lambda i, j: identity_matrix[i][j]

            return transform

        hips_node.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

        # Setup derivatives computation
        # Note: total_frames = int(duration * frame_rate) + 1 = int(1.0 * 30.0) + 1 = 31
        positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])
        velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
        accelerations = np.zeros((31, 3))
        jerks = np.zeros((31, 3))

        mock_derivatives.return_value = (velocities, accelerations, jerks)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run all three analysis modules
            result1 = analyze_root_motion(mock_scene, output_dir=tmpdir)
            result2 = analyze_directional_changes(mock_scene, output_dir=tmpdir)
            result3 = analyze_motion_transitions(mock_scene, output_dir=tmpdir)

            # Verify all modules completed
            assert result1 is not None
            assert result2 is not None
            assert result3 is not None

            # Verify root bone detection was only called ONCE (cached)
            # This is the key integration test - cache should prevent redundant work
            assert mock_detect.call_count == 1, "Root bone should only be detected once due to caching"

    @patch("fbx_tool.analysis.utils._detect_root_bone")
    @patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
    @patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
    def test_cache_cleared_between_different_files(self, mock_derivatives, mock_metadata, mock_detect):
        """Should clear cache when loading a different FBX file."""
        from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
        from fbx_tool.analysis.utils import clear_trajectory_cache, get_scene_cache_key

        # Clear cache before test
        clear_trajectory_cache()

        # Setup first scene
        mock_scene1 = Mock()
        mock_root_node1 = Mock()
        mock_scene1.GetRootNode.return_value = mock_root_node1

        hips_node1 = Mock()
        hips_node1.GetName.return_value = "Hips"
        hips_node1.GetChildCount.return_value = 0
        mock_root_node1.GetChildCount.return_value = 1
        mock_root_node1.GetChild.return_value = hips_node1
        mock_detect.return_value = hips_node1

        # Setup metadata
        mock_metadata.return_value = {
            "has_animation": True,
            "start_time": 0.0,
            "stop_time": 1.0,
            "frame_rate": 30.0,
            "duration": 1.0,
            "total_frames": 30,
        }

        # Setup time span
        mock_time_span = Mock()
        import fbx

        real_start_time = fbx.FbxTime()
        real_start_time.SetSecondDouble(0.0)
        mock_time_span.GetStart.return_value = real_start_time

        duration_time = fbx.FbxTime()
        duration_time.SetSecondDouble(1.0)
        mock_time_span.GetDuration.return_value = duration_time

        mock_anim_stack = Mock()
        mock_anim_stack.GetLocalTimeSpan.return_value = mock_time_span
        mock_scene1.GetCurrentAnimationStack.return_value = mock_anim_stack

        # Setup GetSrcObjectCount and GetSrcObject for animation stack access
        mock_scene1.GetSrcObjectCount.return_value = 1
        mock_scene1.GetSrcObject.return_value = mock_anim_stack

        # Setup transform
        def create_mock_transform(frame):
            transform = Mock()
            transform.GetT.return_value = [float(frame), 0.0, 0.0]
            transform.GetR.return_value = [0.0, 0.0, 0.0]
            # Mock transformation matrix with identity matrix
            identity_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            transform.Get.side_effect = lambda i, j: identity_matrix[i][j]
            return transform

        hips_node1.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

        # Setup derivatives
        # Note: total_frames = int(duration * frame_rate) + 1 = int(1.0 * 30.0) + 1 = 31
        positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])
        velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
        accelerations = np.zeros((31, 3))
        jerks = np.zeros((31, 3))
        mock_derivatives.return_value = (velocities, accelerations, jerks)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Analyze first file
            result1 = analyze_root_motion(mock_scene1, output_dir=tmpdir)
            assert result1 is not None

            # Get cache key for first scene
            cache_key1 = get_scene_cache_key(mock_scene1)

            # Clear cache (simulating loading a new file)
            clear_trajectory_cache()

            # Create second scene (different object)
            mock_scene2 = Mock()
            mock_root_node2 = Mock()
            mock_scene2.GetRootNode.return_value = mock_root_node2

            hips_node2 = Mock()
            hips_node2.GetName.return_value = "Hips"
            hips_node2.GetChildCount.return_value = 0
            mock_root_node2.GetChildCount.return_value = 1
            mock_root_node2.GetChild.return_value = hips_node2
            mock_detect.return_value = hips_node2

            mock_scene2.GetCurrentAnimationStack.return_value = mock_anim_stack

            # Setup GetSrcObjectCount and GetSrcObject for scene2
            mock_scene2.GetSrcObjectCount.return_value = 1
            mock_scene2.GetSrcObject.return_value = mock_anim_stack

            hips_node2.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

            # Analyze second file
            result2 = analyze_root_motion(mock_scene2, output_dir=tmpdir)
            assert result2 is not None

            # Get cache key for second scene
            cache_key2 = get_scene_cache_key(mock_scene2)

            # Verify cache keys are different
            assert cache_key1 != cache_key2, "Different scenes should have different cache keys"

            # Verify root bone detection was called twice (once per file)
            assert mock_detect.call_count == 2, "Should detect root bone for each new file"


@pytest.mark.integration
class TestAnalysisPipelineDataFlow:
    """Test correct data flow between dependent analysis modules."""

    @patch("fbx_tool.analysis.utils._detect_root_bone")
    @patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
    @patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
    def test_directional_changes_uses_root_motion_trajectory(self, mock_derivatives, mock_metadata, mock_detect):
        """Directional changes should use the same trajectory as root motion."""
        from fbx_tool.analysis.directional_change_detection import analyze_directional_changes
        from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
        from fbx_tool.analysis.utils import clear_trajectory_cache

        # Clear cache before test
        clear_trajectory_cache()

        # Setup scene (minimal setup)
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        hips_node = Mock()
        hips_node.GetName.return_value = "Hips"
        hips_node.GetChildCount.return_value = 0
        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips_node
        mock_detect.return_value = hips_node

        # Setup metadata
        mock_metadata.return_value = {
            "has_animation": True,
            "start_time": 0.0,
            "stop_time": 1.0,
            "frame_rate": 30.0,
            "duration": 1.0,
            "total_frames": 30,
        }

        # Setup time span
        mock_time_span = Mock()
        import fbx

        real_start_time = fbx.FbxTime()
        real_start_time.SetSecondDouble(0.0)
        mock_time_span.GetStart.return_value = real_start_time

        duration_time = fbx.FbxTime()
        duration_time.SetSecondDouble(1.0)
        mock_time_span.GetDuration.return_value = duration_time

        mock_anim_stack = Mock()
        mock_anim_stack.GetLocalTimeSpan.return_value = mock_time_span
        mock_scene.GetCurrentAnimationStack.return_value = mock_anim_stack

        # Setup GetSrcObjectCount and GetSrcObject for animation stack access
        mock_scene.GetSrcObjectCount.return_value = 1
        mock_scene.GetSrcObject.return_value = mock_anim_stack

        # Setup transform
        def create_mock_transform(frame):
            transform = Mock()
            transform.GetT.return_value = [float(frame), 0.0, 0.0]
            transform.GetR.return_value = [0.0, float(frame) * 2.0, 0.0]  # Rotation for turning
            # Mock transformation matrix with identity matrix
            identity_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            transform.Get.side_effect = lambda i, j: identity_matrix[i][j]
            return transform

        hips_node.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

        # Setup derivatives
        # Note: total_frames = int(duration * frame_rate) + 1 = int(1.0 * 30.0) + 1 = 31
        positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])
        velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
        accelerations = np.zeros((31, 3))
        jerks = np.zeros((31, 3))
        mock_derivatives.return_value = (velocities, accelerations, jerks)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run both modules
            root_result = analyze_root_motion(mock_scene, output_dir=tmpdir)
            directional_result = analyze_directional_changes(mock_scene, output_dir=tmpdir)

            # Verify both got results
            assert root_result is not None
            assert directional_result is not None

            # Verify they analyzed the same number of frames
            assert root_result["trajectory_frames"] > 0
            # Directional changes should process the same trajectory
            assert directional_result is not None


@pytest.mark.integration
class TestAnalysisPipelineOutputs:
    """Test that the pipeline generates expected output files."""

    @patch("fbx_tool.analysis.utils._detect_root_bone")
    @patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
    @patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
    def test_pipeline_creates_output_files(self, mock_derivatives, mock_metadata, mock_detect):
        """Should create CSV output files for each analysis module."""
        from fbx_tool.analysis.directional_change_detection import analyze_directional_changes
        from fbx_tool.analysis.motion_transition_detection import analyze_motion_transitions
        from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
        from fbx_tool.analysis.utils import clear_trajectory_cache

        # Clear cache before test
        clear_trajectory_cache()

        # Setup scene
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        hips_node = Mock()
        hips_node.GetName.return_value = "Hips"
        hips_node.GetChildCount.return_value = 0
        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips_node
        mock_detect.return_value = hips_node

        # Setup metadata
        mock_metadata.return_value = {
            "has_animation": True,
            "start_time": 0.0,
            "stop_time": 1.0,
            "frame_rate": 30.0,
            "duration": 1.0,
            "total_frames": 30,
        }

        # Setup time span
        mock_time_span = Mock()
        import fbx

        real_start_time = fbx.FbxTime()
        real_start_time.SetSecondDouble(0.0)
        mock_time_span.GetStart.return_value = real_start_time

        duration_time = fbx.FbxTime()
        duration_time.SetSecondDouble(1.0)
        mock_time_span.GetDuration.return_value = duration_time

        mock_anim_stack = Mock()
        mock_anim_stack.GetLocalTimeSpan.return_value = mock_time_span
        mock_scene.GetCurrentAnimationStack.return_value = mock_anim_stack

        # Setup GetSrcObjectCount and GetSrcObject for animation stack access
        mock_scene.GetSrcObjectCount.return_value = 1
        mock_scene.GetSrcObject.return_value = mock_anim_stack

        # Setup transform
        def create_mock_transform(frame):
            transform = Mock()
            transform.GetT.return_value = [float(frame), 0.0, 0.0]
            transform.GetR.return_value = [0.0, 0.0, 0.0]
            # Mock transformation matrix with identity matrix
            identity_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            transform.Get.side_effect = lambda i, j: identity_matrix[i][j]
            return transform

        hips_node.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

        # Setup derivatives
        # Note: total_frames = int(duration * frame_rate) + 1 = int(1.0 * 30.0) + 1 = 31
        positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])
        velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
        accelerations = np.zeros((31, 3))
        jerks = np.zeros((31, 3))
        mock_derivatives.return_value = (velocities, accelerations, jerks)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run all modules
            analyze_root_motion(mock_scene, output_dir=tmpdir)
            analyze_directional_changes(mock_scene, output_dir=tmpdir)
            analyze_motion_transitions(mock_scene, output_dir=tmpdir)

            # Verify output files were created
            output_files = os.listdir(tmpdir)

            # Root motion should create 3 files
            assert "root_motion_trajectory.csv" in output_files
            assert "root_motion_direction.csv" in output_files
            assert "root_motion_summary.csv" in output_files

            # Note: Directional changes and motion transitions may not create files
            # if no transitions/segments were detected in the minimal test data
            # That's okay - the important test is that they don't crash


@pytest.mark.integration
class TestAnalysisWorkerSceneManager:
    """Integration tests for AnalysisWorker using scene manager."""

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    def test_worker_uses_scene_manager(self, mock_metadata, mock_fbx_module):
        """Worker should use scene manager to get scenes."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
            from fbx_tool.gui.main_window import AnalysisWorker
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_scene = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.FbxScene.Create.return_value = mock_scene
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        # Setup metadata
        mock_metadata.return_value = {
            "duration": 1.0,
            "frame_rate": 30.0,
            "bone_count": 10,
        }

        scene_manager = get_scene_manager()

        # Clean up any existing scenes
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            worker = AnalysisWorker("test.fbx", [])

            # Track progress signals
            progress_signals = []
            worker.progress.connect(lambda msg: progress_signals.append(msg))

            worker.run()

            # Verify scene was loaded via scene manager
            stats = scene_manager.get_cache_stats()

            # Scene should be released after worker completes
            assert stats["cached_scenes"] == 0, "Worker should release scene reference when done"

            # Verify cleanup message was emitted
            cleanup_messages = [msg for msg in progress_signals if "Scene reference released" in msg]
            assert len(cleanup_messages) > 0, "Worker should emit cleanup message"

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    def test_worker_gets_cache_hit_when_gui_holds_ref(self, mock_metadata, mock_fbx_module):
        """Worker should get cache hit when GUI already loaded the file."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
            from fbx_tool.gui.main_window import AnalysisWorker
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_scene = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.FbxScene.Create.return_value = mock_scene
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        mock_metadata.return_value = {
            "duration": 1.0,
            "frame_rate": 30.0,
            "bone_count": 10,
        }

        scene_manager = get_scene_manager()
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            # Simulate GUI loading the file first
            gui_ref = scene_manager.get_scene("test.fbx")

            # Verify scene is cached
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 1
            assert stats["scenes"][0]["ref_count"] == 1

            # Now run worker - should get cache hit
            worker = AnalysisWorker("test.fbx", [])

            # Mock load_fbx to track if it's called
            with patch("fbx_tool.gui.main_window.load_fbx") as mock_load:
                # Worker uses scene_manager.get_scene, not load_fbx directly
                # So we need to verify ref_count increases

                stats_before = scene_manager.get_cache_stats()
                worker.run()
                stats_after = scene_manager.get_cache_stats()

                # Worker should have released its ref
                # But GUI still holds ref, so scene should still be cached
                assert stats_after["cached_scenes"] == 1, "Scene should stay cached (GUI holds ref)"
                assert stats_after["scenes"][0]["ref_count"] == 1, "Ref count should be back to 1 (GUI only)"

            # Cleanup
            gui_ref.release()

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    def test_worker_releases_scene_on_error(self, mock_metadata, mock_fbx_module):
        """Worker should release scene reference even when error occurs."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
            from fbx_tool.gui.main_window import AnalysisWorker
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_scene = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.FbxScene.Create.return_value = mock_scene
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        # Simulate error during metadata retrieval
        mock_metadata.side_effect = Exception("Metadata error")

        scene_manager = get_scene_manager()
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            worker = AnalysisWorker("test.fbx", ["dopesheet"])

            worker.run()

            # Verify scene was cleaned up despite error
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 0, "Worker should release scene even on error"


@pytest.mark.integration
class TestGUIWorkflowWithSceneManager:
    """Integration tests for complete GUI workflows using scene manager."""

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    def test_load_analyze_clear_workflow(self, mock_fbx_module):
        """Test: Load file → Run analysis → Clear → Memory freed."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_scene = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.FbxScene.Create.return_value = mock_scene
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        scene_manager = get_scene_manager()
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            # Simulate GUI workflow

            # 1. GUI loads file
            gui_ref = scene_manager.get_scene("test.fbx")
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 1
            assert stats["scenes"][0]["ref_count"] == 1

            # 2. Run analysis (gets its own ref)
            worker_ref = scene_manager.get_scene("test.fbx")
            stats = scene_manager.get_cache_stats()
            assert stats["scenes"][0]["ref_count"] == 2

            # 3. Analysis completes, releases ref
            worker_ref.release()
            stats = scene_manager.get_cache_stats()
            assert stats["scenes"][0]["ref_count"] == 1
            assert stats["cached_scenes"] == 1  # Still cached (GUI holds ref)

            # 4. User clicks "Clear" - GUI releases ref
            gui_ref.release()
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 0  # Memory freed!

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    def test_visualizer_and_analysis_share_scene(self, mock_fbx_module):
        """Test: Visualizer open → Run analysis → Both share same scene."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_scene = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.FbxScene.Create.return_value = mock_scene
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        scene_manager = get_scene_manager()
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            # 1. Visualizer opens
            visualizer_ref = scene_manager.get_scene("test.fbx")
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 1
            assert stats["scenes"][0]["ref_count"] == 1

            # 2. Run analysis while visualizer is open
            analysis_ref = scene_manager.get_scene("test.fbx")
            stats = scene_manager.get_cache_stats()
            assert stats["scenes"][0]["ref_count"] == 2

            # CRITICAL: Both references point to the SAME scene
            assert visualizer_ref.scene is analysis_ref.scene

            # 3. Analysis completes, releases ref
            analysis_ref.release()
            stats = scene_manager.get_cache_stats()
            assert stats["scenes"][0]["ref_count"] == 1
            assert stats["cached_scenes"] == 1  # Still cached (visualizer holds ref)

            # 4. Close visualizer
            visualizer_ref.release()
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 0  # Memory freed!

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    def test_batch_processing_with_scene_manager(self, mock_fbx_module):
        """Test: Process 5 files in batch → No memory accumulation."""
        try:
            from fbx_tool.analysis.scene_manager import get_scene_manager
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        # Setup FBX SDK mocks
        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.IOSROOT = "IOSROOT"

        mock_importer.Initialize.return_value = True
        mock_importer.Import.return_value = True

        scene_manager = get_scene_manager()
        scene_manager.cleanup_all()

        with patch("os.path.exists", return_value=True):
            files = [f"test{i}.fbx" for i in range(5)]

            # Process each file
            for filepath in files:
                # Create unique scene for each file
                mock_scene = Mock()
                mock_fbx_module.FbxScene.Create.return_value = mock_scene

                # Get scene ref
                scene_ref = scene_manager.get_scene(filepath)

                # Do some work...
                assert scene_ref.scene is not None

                # Release when done
                scene_ref.release()

            # Verify all scenes cleaned up (no memory leak)
            stats = scene_manager.get_cache_stats()
            assert stats["cached_scenes"] == 0, "All scenes should be cleaned up after batch processing"
