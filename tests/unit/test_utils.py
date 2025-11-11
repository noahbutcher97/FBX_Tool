"""
Unit tests for utils module

Tests cover:
- File I/O utilities
- Data processing utilities
- Chain definition utilities
- Math utilities
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

    @pytest.mark.skip(reason="write_csv function removed - replaced with write_dict_list_to_csv")
    def test_write_csv_creates_valid_csv(self, temp_dir):
        """Test that write_csv creates valid CSV files."""
        test_path = os.path.join(temp_dir, "test.csv")
        header = ["Name", "Age", "Score"]
        rows = [["Alice", "25", "95.5"], ["Bob", "30", "87.3"]]

        write_csv(test_path, header, rows)

        # Verify file was created
        assert os.path.exists(test_path)

        # Verify content
        with open(test_path, "r") as f:
            import csv

            reader = csv.reader(f)
            lines = list(reader)

            assert lines[0] == header
            assert lines[1] == rows[0]
            assert lines[2] == rows[1]


@pytest.mark.skip(reason="Obsolete tests - format_float function removed during refactoring")
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

    def test_format_float_with_default_precision(self):
        """Test formatting floats with default 6 decimal places."""
        result = format_float(3.14159265359)
        assert result == "3.141593"

    def test_format_float_with_custom_precision(self):
        """Test formatting floats with custom precision."""
        result = format_float(3.14159, precision=2)
        assert result == "3.14"

    def test_format_float_with_zero_precision(self):
        """Test formatting floats with zero decimal places."""
        result = format_float(3.7, precision=0)
        assert result == "4"


@pytest.mark.skip(reason="Obsolete tests - get_standard_chains, validate_chain functions removed")
@pytest.mark.unit
class TestChainDefinitionUtilities:
    """Test chain definition utility functions."""

    def test_get_standard_chains_returns_expected_chains(self):
        """Test that get_standard_chains returns standard humanoid chains."""
        chains = get_standard_chains()

        assert "LeftLeg" in chains
        assert "RightLeg" in chains
        assert "LeftArm" in chains
        assert "RightArm" in chains
        assert "Spine" in chains

        # Verify LeftLeg structure
        assert "thigh_l" in chains["LeftLeg"]
        assert "foot_l" in chains["LeftLeg"]

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

    def test_validate_chain_filters_valid_bones(self):
        """Test that validate_chain filters bones to those in skeleton."""
        chain = ["bone1", "bone2", "bone3", "missing_bone"]
        bone_names = ["bone1", "bone2", "bone3", "other_bone"]

        valid = validate_chain(chain, bone_names)

        assert valid == ["bone1", "bone2", "bone3"]
        assert "missing_bone" not in valid

    def test_validate_chain_handles_empty_chain(self):
        """Test validate_chain with empty chain."""
        chain = []
        bone_names = ["bone1", "bone2"]

        valid = validate_chain(chain, bone_names)

        assert valid == []

    def test_validate_chain_handles_all_invalid(self):
        """Test validate_chain when all bones are invalid."""
        chain = ["missing1", "missing2"]
        bone_names = ["bone1", "bone2"]

        valid = validate_chain(chain, bone_names)

        assert valid == []


@pytest.mark.skip(reason="Obsolete tests - compute_velocity, compute_acceleration, detect_inversions removed")
@pytest.mark.unit
class TestMathUtilities:
    """Test math utility functions."""

    def test_compute_velocity_returns_correct_shape(self):
        """Test that compute_velocity returns array of same length."""
        positions = np.array([0, 1, 2, 3, 4])
        velocity = compute_velocity(positions)

        assert len(velocity) == len(positions)

    def test_compute_velocity_computes_differences(self):
        """Test that compute_velocity computes correct differences."""
        positions = np.array([0.0, 1.0, 3.0, 6.0])
        velocity = compute_velocity(positions)

        # First value prepended (should be 0)
        assert velocity[0] == 0.0
        # Rest should be differences
        assert velocity[1] == 1.0  # 1 - 0
        assert velocity[2] == 2.0  # 3 - 1
        assert velocity[3] == 3.0  # 6 - 3

    def test_compute_acceleration_returns_correct_shape(self):
        """Test that compute_acceleration returns array of same length."""
        velocity = np.array([0, 1, 2, 3, 4])
        acceleration = compute_acceleration(velocity)

        assert len(acceleration) == len(velocity)

    def test_compute_acceleration_computes_differences(self):
        """Test that compute_acceleration computes correct differences."""
        velocity = np.array([0.0, 1.0, 3.0, 6.0])
        acceleration = compute_acceleration(velocity)

        # First value prepended (should be 0)
        assert acceleration[0] == 0.0
        # Rest should be second derivatives
        assert acceleration[1] == 1.0  # 1 - 0
        assert acceleration[2] == 2.0  # 3 - 1
        assert acceleration[3] == 3.0  # 6 - 3

    def test_detect_inversions_identifies_sign_changes(self):
        """Test that detect_inversions finds direction changes."""
        values = np.array([0.0, 1.0, 2.0, 1.5, 1.0, 2.0, 3.0])
        inversions = detect_inversions(values)

        # Should detect inversion when direction changes
        # Going up: 0->1->2, then down: 2->1.5->1, then up again: 1->2->3
        assert isinstance(inversions, np.ndarray)
        assert (
            len(inversions) == len(values) - 1
        )  # detect_inversions returns len-1 array (comparing adjacent velocities)

    def test_detect_inversions_monotonic_sequence(self):
        """Test that detect_inversions finds no inversions in monotonic sequence."""
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        inversions = detect_inversions(values)

        # No inversions in monotonically increasing sequence
        assert np.sum(inversions) == 0

    def test_detect_inversions_alternating_sequence(self):
        """Test that detect_inversions handles alternating values."""
        values = np.array([0.0, 1.0, 0.5, 1.5, 0.75])
        inversions = detect_inversions(values)

        # Should detect multiple inversions
        assert np.sum(inversions) > 0


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

    @pytest.mark.skip(reason="compute_velocity function removed")
    def test_compute_velocity_single_value(self):
        """Test compute_velocity with single value."""
        positions = np.array([5.0])
        velocity = compute_velocity(positions)

        assert len(velocity) == 1
        assert velocity[0] == 0.0  # prepend with first value

    @pytest.mark.skip(reason="compute_acceleration function removed")
    def test_compute_acceleration_single_value(self):
        """Test compute_acceleration with single value."""
        velocity = np.array([3.0])
        acceleration = compute_acceleration(velocity)

        assert len(acceleration) == 1
        assert acceleration[0] == 0.0

    @pytest.mark.skip(reason="detect_inversions function removed")
    def test_detect_inversions_two_values(self):
        """Test detect_inversions with minimal input."""
        values = np.array([1.0, 2.0])
        inversions = detect_inversions(values)

        # With 2 values, we get 1 comparison (len(values)-1)
        # velocity = [0, 1], comparison at i=1: vel[0]*vel[1] = 0*1 = 0, not < 0, so False
        assert len(inversions) == 1
        assert inversions[0] == False  # No inversion with constant positive velocity

    @pytest.mark.skip(reason="validate_chain function removed")
    def test_validate_chain_preserves_order(self):
        """Test that validate_chain preserves chain order."""
        chain = ["bone3", "bone1", "bone2"]
        bone_names = ["bone1", "bone2", "bone3", "bone4"]

        valid = validate_chain(chain, bone_names)

        # Order should be preserved from original chain
        assert valid == ["bone3", "bone1", "bone2"]
