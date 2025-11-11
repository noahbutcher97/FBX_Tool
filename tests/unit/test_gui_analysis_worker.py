"""
Unit tests for GUI AnalysisWorker class.

Tests the background worker thread that runs analysis operations,
focusing on:
- Trajectory cache clearing before analysis starts
- Scene object passing to analysis functions
- Robust error handling for each step
- Progress signal emissions
- Results collection and aggregation
"""

from unittest.mock import Mock, patch

import pytest

# We need to test the AnalysisWorker but it requires PyQt6
# Skip tests if running in headless environment or if imports fail
try:
    from fbx_tool.gui.main_window import AnalysisWorker

    GUI_AVAILABLE = True
except (ImportError, Exception) as e:
    GUI_AVAILABLE = False
    GUI_IMPORT_ERROR = str(e)


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI not available in headless environment")
@pytest.mark.unit
class TestAnalysisWorkerCacheClearing:
    """Test that AnalysisWorker clears trajectory cache before analysis."""

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_worker_clears_cache_before_analysis(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_clear_cache
    ):
        """Should call clear_trajectory_cache before starting analysis."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}

        # Create worker with no operations to minimize execution
        worker = AnalysisWorker("test.fbx", [])

        # Run worker
        worker.run()

        # Verify cache was cleared before analysis
        mock_clear_cache.assert_called_once()


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI not available in headless environment")
@pytest.mark.unit
class TestAnalysisWorkerScenePassing:
    """Test that AnalysisWorker passes scene objects to analysis functions."""

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.gui.main_window.analyze_root_motion")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_root_motion_receives_scene_object(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_analyze, mock_clear_cache
    ):
        """Should pass scene object to analyze_root_motion."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}
        mock_analyze.return_value = {"total_distance": 10.0, "displacement": 5.0, "dominant_direction": "forward"}

        # Create worker with root_motion operation
        worker = AnalysisWorker("test.fbx", ["root_motion"])

        # Run worker
        worker.run()

        # Verify analyze_root_motion was called with scene
        mock_analyze.assert_called_once()
        call_args = mock_analyze.call_args
        assert call_args[0][0] is mock_scene  # First positional arg should be scene

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.gui.main_window.analyze_directional_changes")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_directional_changes_receives_scene_object(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_analyze, mock_clear_cache
    ):
        """Should pass scene object to analyze_directional_changes."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}
        mock_analyze.return_value = {"movement_segments": [], "turning_events": []}

        # Create worker with directional_changes operation
        worker = AnalysisWorker("test.fbx", ["directional_changes"])

        # Run worker
        worker.run()

        # Verify analyze_directional_changes was called with scene
        mock_analyze.assert_called_once()
        call_args = mock_analyze.call_args
        assert call_args[0][0] is mock_scene

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.gui.main_window.analyze_motion_transitions")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_motion_transitions_receives_scene_object(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_analyze, mock_clear_cache
    ):
        """Should pass scene object to analyze_motion_transitions."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}
        mock_analyze.return_value = {"motion_states": []}

        # Create worker with motion_transitions operation
        worker = AnalysisWorker("test.fbx", ["motion_transitions"])

        # Run worker
        worker.run()

        # Verify analyze_motion_transitions was called with scene
        mock_analyze.assert_called_once()
        call_args = mock_analyze.call_args
        assert call_args[0][0] is mock_scene


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI not available in headless environment")
@pytest.mark.unit
class TestAnalysisWorkerErrorHandling:
    """Test robust error handling in AnalysisWorker."""

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    def test_worker_handles_fbx_load_failure(self, mock_get_scene_mgr, mock_clear_cache):
        """Should emit error signal when FBX load fails."""
        # Setup scene manager to raise error
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.side_effect = Exception("Failed to load FBX")
        mock_get_scene_mgr.return_value = mock_scene_manager

        # Create worker
        worker = AnalysisWorker("bad.fbx", ["root_motion"])

        # Capture error signal
        error_received = []
        worker.error.connect(lambda msg: error_received.append(msg))

        # Run worker
        worker.run()

        # Verify error was emitted
        assert len(error_received) > 0
        assert "Failed to load FBX" in error_received[0]

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.gui.main_window.analyze_root_motion")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_worker_continues_after_individual_failure(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_analyze, mock_clear_cache
    ):
        """Should continue to other operations even if one fails."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}

        # First call fails, second should still be attempted
        mock_analyze.side_effect = [
            Exception("Analysis failed"),
            {"total_distance": 10.0, "displacement": 5.0, "dominant_direction": "forward"},
        ]

        # Create worker with two root_motion operations (simulated)
        worker = AnalysisWorker("test.fbx", ["root_motion"])

        # Run worker
        worker.run()

        # Verify errors list was populated
        assert len(worker.errors) > 0


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI not available in headless environment")
@pytest.mark.unit
class TestAnalysisWorkerProgressSignals:
    """Test progress signal emissions from AnalysisWorker."""

    @patch("fbx_tool.gui.main_window.clear_trajectory_cache")
    @patch("fbx_tool.gui.main_window.analyze_root_motion")
    @patch("fbx_tool.analysis.scene_manager.get_scene_manager")
    @patch("fbx_tool.gui.main_window.get_scene_metadata")
    @patch("fbx_tool.gui.main_window.ensure_output_dir")
    def test_worker_emits_progress_signals(
        self, mock_ensure_dir, mock_metadata, mock_get_scene_mgr, mock_analyze, mock_clear_cache
    ):
        """Should emit progress signals during analysis."""
        # Setup scene manager mock
        mock_scene = Mock()
        mock_scene_ref = Mock()
        mock_scene_ref.scene = mock_scene
        mock_scene_manager = Mock()
        mock_scene_manager.get_scene.return_value = mock_scene_ref
        mock_get_scene_mgr.return_value = mock_scene_manager

        mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}
        mock_analyze.return_value = {"total_distance": 10.0, "displacement": 5.0, "dominant_direction": "forward"}

        # Create worker
        worker = AnalysisWorker("test.fbx", ["root_motion"])

        # Capture progress signals
        progress_messages = []
        worker.progress.connect(lambda msg: progress_messages.append(msg))

        # Run worker
        worker.run()

        # Verify progress messages were emitted
        assert len(progress_messages) > 0
        assert any("Loading FBX scene" in msg for msg in progress_messages)
        assert any("root motion" in msg.lower() for msg in progress_messages)
