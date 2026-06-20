"""
Unit tests for utils module

Tests cover:
- File I/O utilities
- Data conversion utilities
- Chain detection utilities
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from fbx_tool.analysis.utils import (
    _infer_chain_name,
    convert_numpy_to_native,
    detect_chains_from_hierarchy,
    ensure_output_dir,
    prepare_output_file,
    safe_overwrite,
    write_dict_list_to_csv,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.unit
class TestFileIOUtilities:
    """Test file I/O utility functions."""

    def test_ensure_output_dir_creates_directory(self, temp_dir):
        """Test that ensure_output_dir creates nested directories."""
        test_path = os.path.join(temp_dir, "subdir", "nested", "file.csv")
        ensure_output_dir(test_path)

        # Check that parent directory was created
        assert os.path.exists(os.path.dirname(test_path))

    def test_ensure_output_dir_handles_existing_directory(self, temp_dir):
        """Test that ensure_output_dir handles existing directories gracefully."""
        test_path = os.path.join(temp_dir, "existing", "file.csv")
        os.makedirs(os.path.dirname(test_path))

        # Should not raise error
        ensure_output_dir(test_path)
        assert os.path.exists(os.path.dirname(test_path))

    def test_safe_overwrite_removes_existing_file(self, temp_dir):
        """Test that safe_overwrite removes existing files."""
        test_file = os.path.join(temp_dir, "test.txt")

        # Create a file
        with open(test_file, "w") as f:
            f.write("old content")

        assert os.path.exists(test_file)

        # Overwrite
        safe_overwrite(test_file)

        # File should be gone
        assert not os.path.exists(test_file)

    def test_safe_overwrite_handles_nonexistent_file(self, temp_dir):
        """Test that safe_overwrite handles non-existent files gracefully."""
        test_file = os.path.join(temp_dir, "nonexistent.txt")

        # Should not raise error
        safe_overwrite(test_file)

    def test_prepare_output_file_creates_and_clears(self, temp_dir):
        """Test that prepare_output_file creates directory and clears old file."""
        test_path = os.path.join(temp_dir, "subdir", "output.csv")

        # Create old file
        os.makedirs(os.path.dirname(test_path))
        with open(test_path, "w") as f:
            f.write("old data")

        # Prepare for new output
        prepare_output_file(test_path)

        # Directory should exist, file should be gone
        assert os.path.exists(os.path.dirname(test_path))
        assert not os.path.exists(test_path)

    def test_write_dict_list_to_csv_creates_valid_csv(self, temp_dir):
        """Test that write_dict_list_to_csv creates valid CSV files."""
        test_path = os.path.join(temp_dir, "test.csv")
        rows = [{"Name": "Alice", "Age": "25", "Score": "95.5"}, {"Name": "Bob", "Age": "30", "Score": "87.3"}]

        write_dict_list_to_csv(rows, test_path)

        # Verify file was created
        assert os.path.exists(test_path)

        # Verify content
        with open(test_path, "r") as f:
            import csv

            reader = csv.reader(f)
            lines = list(reader)

            assert lines[0] == ["Name", "Age", "Score"]
            assert lines[1] == ["Alice", "25", "95.5"]
            assert lines[2] == ["Bob", "30", "87.3"]


@pytest.mark.unit
class TestDataProcessingUtilities:
    """Test data processing utility functions."""

    def test_convert_numpy_to_native_handles_integers(self):
        """Test converting numpy integers to native Python int."""
        np_int = np.int64(42)
        result = convert_numpy_to_native(np_int)

        assert isinstance(result, int)
        assert result == 42

    def test_convert_numpy_to_native_handles_floats(self):
        """Test converting numpy floats to native Python float."""
        np_float = np.float64(3.14159)
        result = convert_numpy_to_native(np_float)

        assert isinstance(result, float)
        assert abs(result - 3.14159) < 1e-10

    def test_convert_numpy_to_native_handles_arrays(self):
        """Test converting numpy arrays to lists."""
        np_array = np.array([1, 2, 3, 4, 5])
        result = convert_numpy_to_native(np_array)

        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_convert_numpy_to_native_handles_nested_dicts(self):
        """Test converting nested dictionaries with numpy types."""
        data = {
            "count": np.int32(10),
            "score": np.float64(95.5),
            "values": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner": np.int64(42)},
        }

        result = convert_numpy_to_native(data)

        assert isinstance(result["count"], int)
        assert isinstance(result["score"], float)
        assert isinstance(result["values"], list)
        assert isinstance(result["nested"]["inner"], int)

    def test_convert_numpy_to_native_handles_lists(self):
        """Test converting lists containing numpy types."""
        data = [np.int32(1), np.float64(2.5), np.array([3, 4])]
        result = convert_numpy_to_native(data)

        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)


@pytest.mark.unit
class TestChainDefinitionUtilities:
    """Test chain definition utility functions."""

    def test_detect_chains_from_hierarchy_finds_simple_chain(self):
        """Test detecting a simple linear chain."""
        hierarchy = {"bone1": None, "bone2": "bone1", "bone3": "bone2", "bone4": "bone3"}

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

        assert len(chains) >= 1
        # Should find one chain containing these bones
        found_chain = None
        for chain in chains.values():
            if "bone1" in chain and "bone4" in chain:
                found_chain = chain
                break

        assert found_chain is not None
        assert len(found_chain) == 4

    def test_detect_chains_from_hierarchy_finds_multiple_chains(self):
        """Test detecting multiple independent chains."""
        hierarchy = {
            # Chain 1
            "a1": None,
            "a2": "a1",
            "a3": "a2",
            "a4": "a3",
            # Chain 2 (branching from a1)
            "b1": "a1",
            "b2": "b1",
            "b3": "b2",
        }

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

        # Should find at least 2 chains
        assert len(chains) >= 2

    def test_detect_chains_from_hierarchy_respects_min_length(self):
        """Test that chains shorter than min_chain_length are ignored."""
        hierarchy = {
            "bone1": None,
            "bone2": "bone1",  # Only 2 bones - too short
        }

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)

        # Should not detect this short chain
        assert len(chains) == 0

    def test_detect_chains_from_hierarchy_handles_branches(self):
        """Test handling of branching (non-linear) hierarchies."""
        hierarchy = {
            "root": None,
            "child1": "root",
            "child2": "root",  # Branch here
            "grandchild1": "child1",
            "grandchild2a": "child2",
            "grandchild2b": "child2",  # Another branch
        }

        chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=2)

        # Should detect chains that don't include branch points
        assert isinstance(chains, dict)

    def test_infer_chain_name_identifies_left_leg(self):
        """Test chain name generation for left leg."""
        chain = ["thigh_l", "calf_l", "foot_l"]
        name = _infer_chain_name(chain)

        assert name == "LeftLeg"

    def test_infer_chain_name_identifies_right_arm(self):
        """Test chain name generation for right arm."""
        chain = ["upperarm_r", "lowerarm_r", "hand_r"]
        name = _infer_chain_name(chain)

        assert name == "RightArm"

    def test_infer_chain_name_identifies_spine(self):
        """Test chain name generation for spine."""
        chain = ["pelvis", "spine_01", "spine_02"]
        name = _infer_chain_name(chain)

        assert name == "Spine"

    def test_infer_chain_name_falls_back_to_first_bone(self):
        """Test chain name falls back to first bone name for unknown chains."""
        chain = ["unknown_bone", "another_bone"]
        name = _infer_chain_name(chain)

        assert name == "unknown_bone"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for utility functions."""

    def test_convert_numpy_to_native_handles_none(self):
        """Test converting None values."""
        result = convert_numpy_to_native(None)
        assert result is None

    def test_convert_numpy_to_native_handles_strings(self):
        """Test that strings pass through unchanged."""
        result = convert_numpy_to_native("test string")
        assert result == "test string"
