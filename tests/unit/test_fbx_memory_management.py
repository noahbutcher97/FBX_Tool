"""
Unit tests for FBX memory management and cleanup.

These tests ensure that FBX SDK resources are properly released
to prevent memory leaks during batch processing.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.mark.unit
class TestFBXLoaderMemoryManagement:
    """Test FBX loader properly manages memory."""

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    def test_load_fbx_returns_manager(self, mock_fbx_module):
        """Should return both scene AND manager for cleanup."""
        from fbx_tool.analysis.fbx_loader import load_fbx

        # Mock FBX SDK objects
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

        # Load FBX - patch os.path.exists inside the function
        with patch("os.path.exists", return_value=True):
            result = load_fbx("test.fbx")

        # CRITICAL: Must return tuple of (scene, manager)
        assert isinstance(result, tuple), "load_fbx must return tuple"
        assert len(result) == 2, "load_fbx must return (scene, manager)"

        scene, manager = result
        assert scene is not None, "Scene must not be None"
        assert manager is not None, "Manager must not be None for cleanup"
        assert scene == mock_scene
        assert manager == mock_manager

        # Verify importer was destroyed (but not manager yet)
        mock_importer.Destroy.assert_called_once()
        mock_manager.Destroy.assert_not_called()  # Caller's responsibility

    @patch("fbx_tool.analysis.fbx_loader.fbx")
    def test_load_fbx_cleans_up_on_init_error(self, mock_fbx_module):
        """Should destroy manager if initialization fails."""
        from fbx_tool.analysis.fbx_loader import load_fbx

        mock_manager = Mock()
        mock_ios = Mock()
        mock_importer = Mock()
        mock_status = Mock()

        mock_fbx_module.FbxManager.Create.return_value = mock_manager
        mock_fbx_module.FbxIOSettings.Create.return_value = mock_ios
        mock_fbx_module.FbxImporter.Create.return_value = mock_importer
        mock_fbx_module.IOSROOT = "IOSROOT"

        # Simulate initialization failure
        mock_importer.Initialize.return_value = False
        mock_importer.GetStatus.return_value = mock_status
        mock_status.GetErrorString.return_value = "Init failed"

        # Should raise and cleanup
        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="FBX SDK failed to initialize"):
                load_fbx("test.fbx")

        # Verify cleanup happened
        mock_importer.Destroy.assert_called_once()
        mock_manager.Destroy.assert_called_once()


@pytest.mark.unit
class TestCleanupFunction:
    """Test cleanup_fbx_scene function."""

    def test_cleanup_destroys_manager(self):
        """Should call Destroy() on manager."""
        from fbx_tool.analysis.fbx_loader import cleanup_fbx_scene

        mock_scene = Mock()
        mock_manager = Mock()

        cleanup_fbx_scene(mock_scene, mock_manager)

        # Manager must be destroyed
        mock_manager.Destroy.assert_called_once()

    def test_cleanup_handles_none_manager(self):
        """Should handle None manager gracefully."""
        from fbx_tool.analysis.fbx_loader import cleanup_fbx_scene

        mock_scene = Mock()

        # Should not crash
        cleanup_fbx_scene(mock_scene, None)
        cleanup_fbx_scene(None, None)

    def test_cleanup_handles_none_scene(self):
        """Should handle None scene gracefully."""
        from fbx_tool.analysis.fbx_loader import cleanup_fbx_scene

        mock_manager = Mock()

        # Should still destroy manager
        cleanup_fbx_scene(None, mock_manager)
        mock_manager.Destroy.assert_called_once()


@pytest.mark.unit
class TestBatchProcessingMemoryUsage:
    """Test that batch processing doesn't accumulate memory."""

    def test_batch_results_dont_store_scenes(self):
        """Batch results should NOT store FBX scenes."""
        # This is a design test - documents the requirement
        # that batch_results should only store models, not scenes

        batch_result_example = {
            "file": "test.fbx",
            "success": True,
            "model": {"gait_type": "walk"},
            # "scene": scene  # ❌ MUST NOT store scene - causes memory leak
        }

        assert "scene" not in batch_result_example, "Batch results must NOT store scenes - causes memory leaks"

    @patch("fbx_tool.gui.main_window.load_fbx")
    def test_worker_cleans_up_manager_on_success(self, mock_load):
        """Worker must clean up FBX manager even on success."""
        # Import here to avoid PyQt issues in headless environment
        try:
            from fbx_tool.gui.main_window import AnalysisWorker
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load.return_value = (mock_scene, mock_manager)

        with patch("fbx_tool.gui.main_window.get_scene_metadata") as mock_metadata:
            mock_metadata.return_value = {"duration": 1.0, "frame_rate": 30.0, "bone_count": 10}

            worker = AnalysisWorker("test.fbx", [])

            # Patch cleanup to verify it's called
            with patch("fbx_tool.analysis.fbx_loader.cleanup_fbx_scene") as mock_cleanup:
                worker.run()

                # Verify cleanup was called with manager
                mock_cleanup.assert_called_once()
                call_args = mock_cleanup.call_args[0]
                assert call_args[1] == mock_manager, "Worker must pass manager to cleanup_fbx_scene()"

    @patch("fbx_tool.gui.main_window.load_fbx")
    def test_worker_cleans_up_manager_on_error(self, mock_load):
        """Worker must clean up FBX manager even on error."""
        try:
            from fbx_tool.gui.main_window import AnalysisWorker
        except ImportError:
            pytest.skip("GUI not available in headless environment")

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load.return_value = (mock_scene, mock_manager)

        with patch("fbx_tool.gui.main_window.get_scene_metadata") as mock_metadata:
            # Simulate error during processing
            mock_metadata.side_effect = Exception("Processing error")

            worker = AnalysisWorker("test.fbx", ["dopesheet"])

            # Patch cleanup to verify it's called even on error
            with patch("fbx_tool.analysis.fbx_loader.cleanup_fbx_scene") as mock_cleanup:
                worker.run()

                # Verify cleanup was still called
                assert mock_cleanup.called, "Worker must cleanup even on error"
                if mock_cleanup.called:
                    call_args = mock_cleanup.call_args[0]
                    assert call_args[1] == mock_manager, "Worker must pass manager to cleanup even on error"
