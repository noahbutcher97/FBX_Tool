"""
Unit tests for gait_summary module.

Tests the GaitSummaryAnalysis data container class.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fbx_tool.analysis.gait_summary import GaitSummaryAnalysis

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_fbx_path():
    """Sample FBX file path."""
    return "/path/to/animation.fbx"


@pytest.fixture
def sample_dopesheet_path():
    """Sample dopesheet file path."""
    return "/path/to/dopesheet.json"


@pytest.fixture
def sample_gait_summary():
    """Sample gait summary data."""
    return {
        "LeftLeg": {
            "gait_type": "walking",
            "avg_stride_length": 1.2,
            "avg_stride_duration": 0.6,
            "cadence": 100.0,
        },
        "RightLeg": {
            "gait_type": "walking",
            "avg_stride_length": 1.18,
            "avg_stride_duration": 0.62,
            "cadence": 97.0,
        },
    }


@pytest.fixture
def sample_chain_conf():
    """Sample chain confidence data."""
    return {
        "LeftLeg": {"confidence": 0.95, "detection_method": "heel_strike"},
        "RightLeg": {"confidence": 0.92, "detection_method": "heel_strike"},
    }


@pytest.fixture
def sample_joint_conf():
    """Sample joint confidence data."""
    return {
        "LeftFoot": {"confidence": 0.98, "position_accuracy": 0.01},
        "RightFoot": {"confidence": 0.97, "position_accuracy": 0.012},
    }


@pytest.fixture
def sample_stride_segments():
    """Sample stride segments."""
    return [
        {"chain": "LeftLeg", "start_frame": 0, "end_frame": 30, "stride_length": 1.2},
        {"chain": "RightLeg", "start_frame": 15, "end_frame": 45, "stride_length": 1.18},
        {"chain": "LeftLeg", "start_frame": 30, "end_frame": 60, "stride_length": 1.22},
    ]


# ==============================================================================
# TEST INITIALIZATION
# ==============================================================================


@pytest.mark.unit
class TestGaitSummaryInitialization:
    """Test GaitSummaryAnalysis initialization."""

    def test_init_with_all_parameters(
        self,
        sample_fbx_path,
        sample_dopesheet_path,
        sample_gait_summary,
        sample_chain_conf,
        sample_joint_conf,
        sample_stride_segments,
    ):
        """Test initialization with all parameters."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            gait_summary=sample_gait_summary,
            chain_conf=sample_chain_conf,
            joint_conf=sample_joint_conf,
            stride_segments=sample_stride_segments,
        )

        assert analysis.fbx_path == sample_fbx_path
        assert analysis.dopesheet_path == sample_dopesheet_path
        assert analysis.gait_summary == sample_gait_summary
        assert analysis.chain_conf == sample_chain_conf
        assert analysis.joint_conf == sample_joint_conf
        assert analysis.stride_segments == sample_stride_segments

    def test_init_with_minimal_parameters(self, sample_fbx_path, sample_dopesheet_path):
        """Test initialization with minimal parameters."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        assert analysis.fbx_path == sample_fbx_path
        assert analysis.dopesheet_path == sample_dopesheet_path
        assert analysis.gait_summary == {}
        assert analysis.chain_conf == {}
        assert analysis.joint_conf == {}
        assert analysis.stride_segments == []

    def test_init_defaults(self, sample_fbx_path, sample_dopesheet_path):
        """Test that default values are set correctly."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        assert isinstance(analysis.gait_summary, dict)
        assert isinstance(analysis.chain_conf, dict)
        assert isinstance(analysis.joint_conf, dict)
        assert isinstance(analysis.stride_segments, list)


# ==============================================================================
# TEST STRIDE COUNT
# ==============================================================================


@pytest.mark.unit
class TestGetStrideCount:
    """Test stride count retrieval."""

    def test_get_stride_count_with_strides(self, sample_fbx_path, sample_dopesheet_path, sample_stride_segments):
        """Test stride count with multiple strides."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, stride_segments=sample_stride_segments
        )

        assert analysis.get_stride_count() == 3

    def test_get_stride_count_empty(self, sample_fbx_path, sample_dopesheet_path):
        """Test stride count with no strides."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        assert analysis.get_stride_count() == 0

    def test_get_stride_count_single(self, sample_fbx_path, sample_dopesheet_path):
        """Test stride count with single stride."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            stride_segments=[{"chain": "LeftLeg", "start_frame": 0, "end_frame": 30}],
        )

        assert analysis.get_stride_count() == 1


# ==============================================================================
# TEST GAIT TYPE RETRIEVAL
# ==============================================================================


@pytest.mark.unit
class TestGetGaitType:
    """Test gait type retrieval."""

    def test_get_gait_type_left_leg(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary):
        """Test getting gait type for left leg."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=sample_gait_summary
        )

        assert analysis.get_gait_type("LeftLeg") == "walking"

    def test_get_gait_type_right_leg(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary):
        """Test getting gait type for right leg."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=sample_gait_summary
        )

        assert analysis.get_gait_type("RightLeg") == "walking"

    def test_get_gait_type_default_chain(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary):
        """Test getting gait type with default chain parameter."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=sample_gait_summary
        )

        # Default chain is "LeftLeg"
        assert analysis.get_gait_type() == "walking"

    def test_get_gait_type_missing_chain(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary):
        """Test getting gait type for non-existent chain."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=sample_gait_summary
        )

        assert analysis.get_gait_type("NonExistentChain") == "Unknown"

    def test_get_gait_type_empty_summary(self, sample_fbx_path, sample_dopesheet_path):
        """Test getting gait type with empty summary."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        assert analysis.get_gait_type("LeftLeg") == "Unknown"

    def test_get_gait_type_missing_field(self, sample_fbx_path, sample_dopesheet_path):
        """Test getting gait type when gait_type field is missing."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            gait_summary={"LeftLeg": {"avg_stride_length": 1.2}},  # No gait_type field
        )

        assert analysis.get_gait_type("LeftLeg") == "Unknown"


# ==============================================================================
# TEST CHAIN CONFIDENCE RETRIEVAL
# ==============================================================================


@pytest.mark.unit
class TestGetChainConfidence:
    """Test chain confidence retrieval."""

    def test_get_chain_confidence_left_leg(self, sample_fbx_path, sample_dopesheet_path, sample_chain_conf):
        """Test getting chain confidence for left leg."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, chain_conf=sample_chain_conf
        )

        assert analysis.get_chain_confidence("LeftLeg") == 0.95

    def test_get_chain_confidence_right_leg(self, sample_fbx_path, sample_dopesheet_path, sample_chain_conf):
        """Test getting chain confidence for right leg."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, chain_conf=sample_chain_conf
        )

        assert analysis.get_chain_confidence("RightLeg") == 0.92

    def test_get_chain_confidence_missing_chain(self, sample_fbx_path, sample_dopesheet_path, sample_chain_conf):
        """Test getting chain confidence for non-existent chain."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, chain_conf=sample_chain_conf
        )

        assert analysis.get_chain_confidence("NonExistentChain") == 0.0

    def test_get_chain_confidence_empty(self, sample_fbx_path, sample_dopesheet_path):
        """Test getting chain confidence with empty chain_conf."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        assert analysis.get_chain_confidence("LeftLeg") == 0.0

    def test_get_chain_confidence_missing_field(self, sample_fbx_path, sample_dopesheet_path):
        """Test getting chain confidence when confidence field is missing."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            chain_conf={"LeftLeg": {"detection_method": "heel_strike"}},  # No confidence field
        )

        assert analysis.get_chain_confidence("LeftLeg") == 0.0


# ==============================================================================
# TEST DICTIONARY CONVERSION
# ==============================================================================


@pytest.mark.unit
class TestToDict:
    """Test dictionary conversion."""

    def test_to_dict_complete(
        self,
        sample_fbx_path,
        sample_dopesheet_path,
        sample_gait_summary,
        sample_chain_conf,
        sample_joint_conf,
        sample_stride_segments,
    ):
        """Test to_dict with complete data."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            gait_summary=sample_gait_summary,
            chain_conf=sample_chain_conf,
            joint_conf=sample_joint_conf,
            stride_segments=sample_stride_segments,
        )

        result = analysis.to_dict()

        assert result["fbx_path"] == sample_fbx_path
        assert result["dopesheet_path"] == sample_dopesheet_path
        assert result["gait_summary"] == sample_gait_summary
        assert result["chain_conf"] == sample_chain_conf
        assert result["stride_segments"] == sample_stride_segments

    def test_to_dict_minimal(self, sample_fbx_path, sample_dopesheet_path):
        """Test to_dict with minimal data."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        result = analysis.to_dict()

        assert result["fbx_path"] == sample_fbx_path
        assert result["dopesheet_path"] == sample_dopesheet_path
        assert result["gait_summary"] == {}
        assert result["chain_conf"] == {}
        assert result["joint_conf"] == {}
        assert result["stride_segments"] == []

    def test_to_dict_converts_joint_conf_keys_to_str(self, sample_fbx_path, sample_dopesheet_path):
        """Test that joint_conf keys are converted to strings."""
        # Use integer keys to test conversion
        joint_conf = {123: {"confidence": 0.98}, 456: {"confidence": 0.97}}

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, joint_conf=joint_conf
        )

        result = analysis.to_dict()

        # Keys should be converted to strings
        assert "123" in result["joint_conf"]
        assert "456" in result["joint_conf"]
        assert 123 not in result["joint_conf"]

    def test_to_dict_handles_numpy_types(self, sample_fbx_path, sample_dopesheet_path):
        """Test that numpy types are converted to native Python types."""
        gait_summary = {
            "LeftLeg": {
                "avg_stride_length": np.float64(1.2),
                "avg_stride_duration": np.float32(0.6),
                "cadence": np.int32(100),
            }
        }

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=gait_summary
        )

        result = analysis.to_dict()

        # Check that values are native Python types
        left_leg = result["gait_summary"]["LeftLeg"]
        assert isinstance(left_leg["avg_stride_length"], (int, float))
        assert isinstance(left_leg["avg_stride_duration"], (int, float))
        assert isinstance(left_leg["cadence"], (int, float))

        # Verify they're not numpy types
        assert not isinstance(left_leg["avg_stride_length"], np.generic)
        assert not isinstance(left_leg["avg_stride_duration"], np.generic)
        assert not isinstance(left_leg["cadence"], np.generic)

    def test_to_dict_handles_numpy_arrays(self, sample_fbx_path, sample_dopesheet_path):
        """Test that numpy arrays are converted to lists."""
        gait_summary = {"LeftLeg": {"stride_lengths": np.array([1.2, 1.18, 1.22])}}

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=gait_summary
        )

        result = analysis.to_dict()

        # Check that array is converted to list
        assert isinstance(result["gait_summary"]["LeftLeg"]["stride_lengths"], list)
        assert result["gait_summary"]["LeftLeg"]["stride_lengths"] == [1.2, 1.18, 1.22]


# ==============================================================================
# TEST JSON EXPORT
# ==============================================================================


@pytest.mark.unit
class TestToJson:
    """Test JSON export."""

    def test_to_json_creates_file(
        self,
        sample_fbx_path,
        sample_dopesheet_path,
        sample_gait_summary,
        sample_chain_conf,
        sample_joint_conf,
        sample_stride_segments,
    ):
        """Test that to_json creates a file."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            gait_summary=sample_gait_summary,
            chain_conf=sample_chain_conf,
            joint_conf=sample_joint_conf,
            stride_segments=sample_stride_segments,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gait_summary.json"
            analysis.to_json(str(output_path))

            assert output_path.exists()

    def test_to_json_valid_content(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary):
        """Test that to_json writes valid JSON."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=sample_gait_summary
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gait_summary.json"
            analysis.to_json(str(output_path))

            # Read and parse JSON
            with open(output_path) as f:
                data = json.load(f)

            assert data["fbx_path"] == sample_fbx_path
            assert data["dopesheet_path"] == sample_dopesheet_path
            assert data["gait_summary"] == sample_gait_summary

    def test_to_json_creates_parent_directories(self, sample_fbx_path, sample_dopesheet_path):
        """Test that to_json creates parent directories if needed."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "gait_summary.json"
            analysis.to_json(str(output_path))

            assert output_path.exists()

    def test_to_json_handles_numpy_types(self, sample_fbx_path, sample_dopesheet_path):
        """Test that to_json handles numpy types correctly."""
        gait_summary = {
            "LeftLeg": {
                "avg_stride_length": np.float64(1.2),
                "stride_lengths": np.array([1.2, 1.18, 1.22]),
            }
        }

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=gait_summary
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gait_summary.json"
            # Should not raise an error
            analysis.to_json(str(output_path))

            # Verify JSON is valid and parseable
            with open(output_path) as f:
                data = json.load(f)

            assert data["gait_summary"]["LeftLeg"]["avg_stride_length"] == 1.2
            assert data["gait_summary"]["LeftLeg"]["stride_lengths"] == [1.2, 1.18, 1.22]


# ==============================================================================
# TEST STRING REPRESENTATION
# ==============================================================================


@pytest.mark.unit
class TestRepr:
    """Test string representation."""

    def test_repr_format(self, sample_fbx_path, sample_dopesheet_path, sample_gait_summary, sample_stride_segments):
        """Test that __repr__ returns expected format."""
        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path,
            dopesheet_path=sample_dopesheet_path,
            gait_summary=sample_gait_summary,
            stride_segments=sample_stride_segments,
        )

        repr_str = repr(analysis)

        assert "GaitSummaryAnalysis" in repr_str
        assert sample_fbx_path in repr_str
        assert "strides=3" in repr_str
        assert "gait_type=walking" in repr_str

    def test_repr_minimal(self, sample_fbx_path, sample_dopesheet_path):
        """Test __repr__ with minimal data."""
        analysis = GaitSummaryAnalysis(fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path)

        repr_str = repr(analysis)

        assert "GaitSummaryAnalysis" in repr_str
        assert sample_fbx_path in repr_str
        assert "strides=0" in repr_str
        assert "gait_type=Unknown" in repr_str


# ==============================================================================
# TEST EDGE CASES
# ==============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_strings_as_paths(self):
        """Test initialization with empty string paths."""
        analysis = GaitSummaryAnalysis(fbx_path="", dopesheet_path="")

        assert analysis.fbx_path == ""
        assert analysis.dopesheet_path == ""

    def test_none_values_in_gait_summary(self, sample_fbx_path, sample_dopesheet_path):
        """Test handling of None values in gait summary."""
        gait_summary = {"LeftLeg": None}

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=gait_summary
        )

        # Should handle gracefully
        assert analysis.get_gait_type("LeftLeg") == "Unknown"

    def test_nested_numpy_conversion(self, sample_fbx_path, sample_dopesheet_path):
        """Test deeply nested numpy type conversion."""
        gait_summary = {
            "LeftLeg": {
                "metrics": {"stride_data": {"lengths": np.array([1.2, 1.18]), "durations": np.array([0.6, 0.62])}}
            }
        }

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, gait_summary=gait_summary
        )

        result = analysis.to_dict()

        # Check nested arrays are converted
        stride_data = result["gait_summary"]["LeftLeg"]["metrics"]["stride_data"]
        assert isinstance(stride_data["lengths"], list)
        assert isinstance(stride_data["durations"], list)

    def test_special_characters_in_paths(self):
        """Test handling of special characters in file paths."""
        fbx_path = "/path/with spaces/and-special_chars!/animation.fbx"
        dopesheet_path = "/path/with spaces/dopesheet.json"

        analysis = GaitSummaryAnalysis(fbx_path=fbx_path, dopesheet_path=dopesheet_path)

        assert analysis.fbx_path == fbx_path
        assert analysis.dopesheet_path == dopesheet_path

        repr_str = repr(analysis)
        assert fbx_path in repr_str

    def test_none_values_in_chain_conf(self, sample_fbx_path, sample_dopesheet_path):
        """Test handling of None values in chain_conf."""
        chain_conf = {"LeftLeg": None}

        analysis = GaitSummaryAnalysis(
            fbx_path=sample_fbx_path, dopesheet_path=sample_dopesheet_path, chain_conf=chain_conf
        )

        # Should handle gracefully
        assert analysis.get_chain_confidence("LeftLeg") == 0.0
