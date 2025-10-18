"""
Unit tests for FBX Scene Manager with reference counting.

These tests ensure that:
- Reference counting works correctly
- Scenes are cached and reused
- Cleanup happens when ref count hits 0
- Thread safety is maintained
- Clear button properly releases references
"""

import threading
import time
from unittest.mock import MagicMock, Mock, call, patch

import pytest


@pytest.mark.unit
class TestFBXSceneReference:
    """Test FBXSceneReference smart pointer behavior."""

    def test_reference_creation(self):
        """Should create reference with scene and manager."""
        from fbx_tool.analysis.scene_manager import FBXSceneReference

        mock_scene = Mock()
        mock_manager = Mock()
        mock_scene_manager = Mock()

        ref = FBXSceneReference(mock_scene, mock_manager, "test.fbx", mock_scene_manager)

        assert ref.scene == mock_scene
        assert ref.manager == mock_manager
        assert ref.filepath == "test.fbx"
        assert not ref._released

    def test_reference_release(self):
        """Should call scene manager release when released."""
        from fbx_tool.analysis.scene_manager import FBXSceneReference

        mock_scene = Mock()
        mock_manager = Mock()
        mock_scene_manager = Mock()

        ref = FBXSceneReference(mock_scene, mock_manager, "test.fbx", mock_scene_manager)
        ref.release()

        assert ref._released
        mock_scene_manager._release_scene.assert_called_once_with("test.fbx")

    def test_reference_release_idempotent(self):
        """Should only release once even if called multiple times."""
        from fbx_tool.analysis.scene_manager import FBXSceneReference

        mock_scene = Mock()
        mock_manager = Mock()
        mock_scene_manager = Mock()

        ref = FBXSceneReference(mock_scene, mock_manager, "test.fbx", mock_scene_manager)
        ref.release()
        ref.release()  # Second call
        ref.release()  # Third call

        # Should only call once
        mock_scene_manager._release_scene.assert_called_once_with("test.fbx")

    def test_reference_context_manager(self):
        """Should auto-release when used as context manager."""
        from fbx_tool.analysis.scene_manager import FBXSceneReference

        mock_scene = Mock()
        mock_manager = Mock()
        mock_scene_manager = Mock()

        ref = FBXSceneReference(mock_scene, mock_manager, "test.fbx", mock_scene_manager)

        with ref:
            assert not ref._released

        # Should auto-release on exit
        assert ref._released
        mock_scene_manager._release_scene.assert_called_once_with("test.fbx")


@pytest.mark.unit
class TestSceneManager:
    """Test FBXSceneManager core functionality."""

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_get_scene_loads_new_scene(self, mock_load_fbx):
        """Should load scene from disk on first request."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()
        scene_ref = scene_mgr.get_scene("test.fbx")

        # Should load from disk
        mock_load_fbx.assert_called_once_with("test.fbx")

        # Should return reference with correct scene
        assert scene_ref.scene == mock_scene
        assert scene_ref.manager == mock_manager

        # Should cache it with ref_count = 1
        assert "test.fbx" in scene_mgr._scenes
        assert scene_mgr._scenes["test.fbx"].ref_count == 1

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_get_scene_reuses_cached_scene(self, mock_load_fbx):
        """Should reuse cached scene on second request (cache hit)."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()

        # First request - loads from disk
        ref1 = scene_mgr.get_scene("test.fbx")
        assert scene_mgr._scenes["test.fbx"].ref_count == 1

        # Second request - cache hit!
        ref2 = scene_mgr.get_scene("test.fbx")
        assert scene_mgr._scenes["test.fbx"].ref_count == 2

        # Should only load once
        mock_load_fbx.assert_called_once_with("test.fbx")

        # Both refs should point to same scene
        assert ref1.scene == ref2.scene

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_release_scene_decrements_ref_count(self, mock_cleanup, mock_load_fbx):
        """Should decrement ref count when reference released."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()

        ref1 = scene_mgr.get_scene("test.fbx")
        ref2 = scene_mgr.get_scene("test.fbx")
        assert scene_mgr._scenes["test.fbx"].ref_count == 2

        # Release one reference
        ref1.release()
        assert scene_mgr._scenes["test.fbx"].ref_count == 1

        # Should NOT cleanup yet (ref_count > 0)
        mock_cleanup.assert_not_called()
        assert "test.fbx" in scene_mgr._scenes

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_release_scene_cleans_up_at_zero(self, mock_cleanup, mock_load_fbx):
        """Should cleanup scene when ref count hits 0."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()

        ref1 = scene_mgr.get_scene("test.fbx")
        ref2 = scene_mgr.get_scene("test.fbx")
        assert scene_mgr._scenes["test.fbx"].ref_count == 2

        # Release both references
        ref1.release()
        ref2.release()

        # Should cleanup when ref_count = 0
        mock_cleanup.assert_called_once_with(mock_scene, mock_manager)
        assert "test.fbx" not in scene_mgr._scenes

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_force_reload(self, mock_load_fbx):
        """Should reload scene when force_reload=True."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene1 = Mock()
        mock_manager1 = Mock()
        mock_scene2 = Mock()
        mock_manager2 = Mock()
        mock_load_fbx.side_effect = [(mock_scene1, mock_manager1), (mock_scene2, mock_manager2)]

        scene_mgr = FBXSceneManager()

        # First load
        ref1 = scene_mgr.get_scene("test.fbx")
        assert ref1.scene == mock_scene1

        # Force reload
        ref2 = scene_mgr.get_scene("test.fbx", force_reload=True)
        assert ref2.scene == mock_scene2

        # Should load twice
        assert mock_load_fbx.call_count == 2

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_cleanup_all(self, mock_cleanup, mock_load_fbx):
        """Should cleanup all cached scenes."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene1 = Mock()
        mock_manager1 = Mock()
        mock_scene2 = Mock()
        mock_manager2 = Mock()
        mock_load_fbx.side_effect = [(mock_scene1, mock_manager1), (mock_scene2, mock_manager2)]

        scene_mgr = FBXSceneManager()

        # Load two scenes
        ref1 = scene_mgr.get_scene("file1.fbx")
        ref2 = scene_mgr.get_scene("file2.fbx")

        # Cleanup all
        scene_mgr.cleanup_all()

        # Should cleanup both
        assert mock_cleanup.call_count == 2
        assert len(scene_mgr._scenes) == 0

    def test_cache_stats(self):
        """Should return accurate cache statistics."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        with patch("fbx_tool.analysis.scene_manager.load_fbx") as mock_load_fbx:
            mock_load_fbx.side_effect = [(Mock(), Mock()), (Mock(), Mock())]

            scene_mgr = FBXSceneManager()

            ref1 = scene_mgr.get_scene("file1.fbx")
            ref2a = scene_mgr.get_scene("file2.fbx")
            ref2b = scene_mgr.get_scene("file2.fbx")  # Second ref to file2

            stats = scene_mgr.get_cache_stats()

            assert stats["cached_scenes"] == 2
            assert len(stats["scenes"]) == 2

            # Find file2 in stats
            file2_stats = next(s for s in stats["scenes"] if s["filepath"] == "file2.fbx")
            assert file2_stats["ref_count"] == 2


@pytest.mark.unit
class TestSceneManagerThreadSafety:
    """Test thread safety of scene manager."""

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_concurrent_access(self, mock_load_fbx):
        """Should handle concurrent access from multiple threads."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()
        refs = []
        errors = []

        def get_scene_worker():
            try:
                ref = scene_mgr.get_scene("test.fbx")
                refs.append(ref)
            except Exception as e:
                errors.append(e)

        # Launch 10 threads concurrently
        threads = [threading.Thread(target=get_scene_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have errors
        assert len(errors) == 0

        # Should have 10 references
        assert len(refs) == 10

        # Should only load once (cache hit for all)
        mock_load_fbx.assert_called_once()

        # Ref count should be 10
        assert scene_mgr._scenes["test.fbx"].ref_count == 10


@pytest.mark.unit
class TestSceneManagerIntegration:
    """Integration tests for common workflows."""

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_gui_holds_reference_workflow(self, mock_cleanup, mock_load_fbx):
        """Test: GUI loads file, opens visualizer, closes visualizer, runs analysis."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()

        # 1. GUI loads file
        gui_ref = scene_mgr.get_scene("walk.fbx")
        assert scene_mgr._scenes["walk.fbx"].ref_count == 1

        # 2. Visualizer opens (gets its own reference)
        viz_ref = scene_mgr.get_scene("walk.fbx")
        assert scene_mgr._scenes["walk.fbx"].ref_count == 2

        # 3. Visualizer closes (releases reference)
        viz_ref.release()
        assert scene_mgr._scenes["walk.fbx"].ref_count == 1
        # Scene still cached! (GUI holds ref)
        assert "walk.fbx" in scene_mgr._scenes
        mock_cleanup.assert_not_called()

        # 4. Analysis runs (cache hit!)
        with scene_mgr.get_scene("walk.fbx") as analysis_ref:
            assert scene_mgr._scenes["walk.fbx"].ref_count == 2
            # Same scene reused
            assert analysis_ref.scene == mock_scene

        # Analysis done
        assert scene_mgr._scenes["walk.fbx"].ref_count == 1

        # 5. GUI clears (releases last reference)
        gui_ref.release()
        # Now cleanup should happen
        mock_cleanup.assert_called_once_with(mock_scene, mock_manager)
        assert "walk.fbx" not in scene_mgr._scenes

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_clear_button_releases_memory(self, mock_cleanup, mock_load_fbx):
        """Test: Clear button releases all GUI references."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_load_fbx.side_effect = [(Mock(), Mock()), (Mock(), Mock()), (Mock(), Mock())]

        scene_mgr = FBXSceneManager()

        # GUI loads batch of 3 files
        gui_refs = {
            "file1.fbx": scene_mgr.get_scene("file1.fbx"),
            "file2.fbx": scene_mgr.get_scene("file2.fbx"),
            "file3.fbx": scene_mgr.get_scene("file3.fbx"),
        }

        assert len(scene_mgr._scenes) == 3

        # User clicks "Clear"
        for ref in gui_refs.values():
            ref.release()
        gui_refs.clear()

        # All scenes should be cleaned up
        assert mock_cleanup.call_count == 3
        assert len(scene_mgr._scenes) == 0

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_analysis_while_visualizer_open(self, mock_load_fbx):
        """Test: Run analysis while visualizer is open (scene sharing)."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene = Mock()
        mock_manager = Mock()
        mock_load_fbx.return_value = (mock_scene, mock_manager)

        scene_mgr = FBXSceneManager()

        # GUI and visualizer both hold references
        gui_ref = scene_mgr.get_scene("walk.fbx")
        viz_ref = scene_mgr.get_scene("walk.fbx")
        assert scene_mgr._scenes["walk.fbx"].ref_count == 2

        # User runs analysis while visualizer open
        with scene_mgr.get_scene("walk.fbx") as analysis_ref:
            # ref_count = 3 (GUI + viz + analysis)
            assert scene_mgr._scenes["walk.fbx"].ref_count == 3

            # CRITICAL: Analysis and visualizer share the same scene!
            assert analysis_ref.scene == viz_ref.scene

        # Analysis done, ref_count back to 2
        assert scene_mgr._scenes["walk.fbx"].ref_count == 2


@pytest.mark.unit
class TestSceneManagerEdgeCases:
    """Test edge cases and error handling."""

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_release_non_existent_scene(self, mock_load_fbx):
        """Should handle releasing scene that's not cached."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        scene_mgr = FBXSceneManager()

        # Try to release scene that was never loaded
        # Should not crash
        scene_mgr._release_scene("nonexistent.fbx")

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_get_singleton_instance(self, mock_load_fbx):
        """Should return same singleton instance."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager, get_scene_manager

        instance1 = FBXSceneManager.get_instance()
        instance2 = FBXSceneManager.get_instance()
        instance3 = get_scene_manager()

        assert instance1 is instance2
        assert instance2 is instance3

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_force_reload_with_active_refs(self, mock_cleanup, mock_load_fbx):
        """Should warn when force reloading scene with active references."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        mock_scene1 = Mock()
        mock_manager1 = Mock()
        mock_scene2 = Mock()
        mock_manager2 = Mock()
        mock_load_fbx.side_effect = [(mock_scene1, mock_manager1), (mock_scene2, mock_manager2)]

        scene_mgr = FBXSceneManager()

        # Load and hold reference
        ref1 = scene_mgr.get_scene("test.fbx")
        assert scene_mgr._scenes["test.fbx"].ref_count == 1

        # Force reload while ref active (prints warning)
        ref2 = scene_mgr.get_scene("test.fbx", force_reload=True)

        # Should cleanup old scene
        mock_cleanup.assert_called_once_with(mock_scene1, mock_manager1)

        # Should load new scene
        assert ref2.scene == mock_scene2


@pytest.mark.unit
class TestVisualizerSmartCaching:
    """Test visualizer smart caching behavior."""

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_smart_caching_keeps_current_plus_neighbors(self, mock_load_fbx):
        """Should keep current + previous + next files, release others."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        # Mock multiple scenes
        scenes = [(Mock(), Mock()) for _ in range(10)]
        mock_load_fbx.side_effect = scenes

        scene_mgr = FBXSceneManager()

        # Simulate visualizer loading files 0, 1, 2, 3
        refs = {}
        for i in range(4):
            refs[i] = scene_mgr.get_scene(f"file{i}.fbx")

        assert len(scene_mgr._scenes) == 4
        assert all(scene_mgr._scenes[f"file{i}.fbx"].ref_count == 1 for i in range(4))

        # Simulate smart caching: switch to file 2, should keep files 1, 2, 3
        # Release files 0
        files_to_keep = {1, 2, 3}
        for i in list(refs.keys()):
            if i not in files_to_keep:
                refs[i].release()
                del refs[i]

        # File 0 should be cleaned up
        assert "file0.fbx" not in scene_mgr._scenes
        # Files 1, 2, 3 should still be cached
        assert all(f"file{i}.fbx" in scene_mgr._scenes for i in range(1, 4))

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_smart_caching_with_gui_refs(self, mock_cleanup, mock_load_fbx):
        """Test: Visualizer releases ref but GUI keeps it cached."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        scenes = [(Mock(), Mock()) for _ in range(5)]
        mock_load_fbx.side_effect = scenes

        scene_mgr = FBXSceneManager()

        # GUI loads batch of 5 files
        gui_refs = {i: scene_mgr.get_scene(f"file{i}.fbx") for i in range(5)}

        # All files have ref_count = 1 (GUI only)
        assert all(scene_mgr._scenes[f"file{i}.fbx"].ref_count == 1 for i in range(5))

        # Visualizer gets ref to file 2
        viz_ref = scene_mgr.get_scene("file2.fbx")
        assert scene_mgr._scenes["file2.fbx"].ref_count == 2

        # Visualizer releases (smart caching moved away)
        viz_ref.release()

        # File 2 should STILL be cached (GUI has it)
        assert "file2.fbx" in scene_mgr._scenes
        assert scene_mgr._scenes["file2.fbx"].ref_count == 1
        mock_cleanup.assert_not_called()

        # Visualizer switches back to file 2 - cache hit!
        viz_ref2 = scene_mgr.get_scene("file2.fbx")
        assert scene_mgr._scenes["file2.fbx"].ref_count == 2

        # Should NOT load again (cache hit)
        assert mock_load_fbx.call_count == 5  # Only initial loads

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    @patch("fbx_tool.analysis.scene_manager.cleanup_fbx_scene")
    def test_smart_caching_boundary_cases(self, mock_cleanup, mock_load_fbx):
        """Test: Smart caching at boundaries (first and last file)."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        scenes = [(Mock(), Mock()) for _ in range(10)]
        mock_load_fbx.side_effect = scenes

        scene_mgr = FBXSceneManager()

        # At first file (index 0), should keep 0 and 1
        ref0 = scene_mgr.get_scene("file0.fbx")
        ref1 = scene_mgr.get_scene("file1.fbx")
        assert len(scene_mgr._scenes) == 2

        # At last file (index 9), should keep 8 and 9
        # Simulate switching to end
        for ref in [ref0, ref1]:
            ref.release()

        ref8 = scene_mgr.get_scene("file8.fbx")
        ref9 = scene_mgr.get_scene("file9.fbx")

        # Files 0 and 1 should be cleaned up (no other refs)
        assert "file0.fbx" not in scene_mgr._scenes
        assert "file1.fbx" not in scene_mgr._scenes

        # Files 8 and 9 should be cached
        assert "file8.fbx" in scene_mgr._scenes
        assert "file9.fbx" in scene_mgr._scenes

    @patch("fbx_tool.analysis.scene_manager.load_fbx")
    def test_smart_caching_memory_usage(self, mock_load_fbx):
        """Test: Smart caching limits memory usage to 3 files max."""
        from fbx_tool.analysis.scene_manager import FBXSceneManager

        # Simulate 100 files
        scenes = [(Mock(), Mock()) for _ in range(100)]
        mock_load_fbx.side_effect = scenes

        scene_mgr = FBXSceneManager()

        # Simulate visualizer switching through all 100 files
        # Using smart caching: keep current ± 1
        current_refs = {}

        for i in range(100):
            # Smart caching logic
            files_to_keep = {i}
            if i > 0:
                files_to_keep.add(i - 1)
            if i < 99:
                files_to_keep.add(i + 1)

            # Release old refs
            for old_i in list(current_refs.keys()):
                if old_i not in files_to_keep:
                    current_refs[old_i].release()
                    del current_refs[old_i]

            # Get new refs
            for keep_i in files_to_keep:
                if keep_i not in current_refs:
                    current_refs[keep_i] = scene_mgr.get_scene(f"file{keep_i}.fbx")

            # At any point, should have at most 3 refs
            assert len(current_refs) <= 3, f"Too many refs at index {i}: {len(current_refs)}"

        # Final state: should only have last 2 files (99 and 98)
        assert len(current_refs) == 2
        assert 98 in current_refs
        assert 99 in current_refs
