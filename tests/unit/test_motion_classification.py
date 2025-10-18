"""
Unit tests for motion_classification module.

Tests natural language generation, animation classification, and metadata generation.
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from fbx_tool.analysis.motion_classification import (
    SEGMENT_TEMPLATES,
    TRANSITION_TEMPLATES,
    TURNING_TEMPLATES,
    classify_animation_type,
    describe_segment,
    describe_transition,
    generate_motion_summary,
    generate_narrative_summary,
)

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_segments():
    """Sample temporal segments for testing."""
    return [
        {
            "start_frame": 0,
            "end_frame": 29,
            "duration_seconds": 1.0,
            "motion_state": "idle",
            "direction": "stationary",
            "is_turning": False,
            "composite_label": "idle_stationary",
        },
        {
            "start_frame": 30,
            "end_frame": 89,
            "duration_seconds": 2.0,
            "motion_state": "walking",
            "direction": "forward",
            "is_turning": False,
            "composite_label": "walking_forward",
        },
        {
            "start_frame": 90,
            "end_frame": 149,
            "duration_seconds": 2.0,
            "motion_state": "running",
            "direction": "forward",
            "is_turning": True,
            "composite_label": "running_forward_turning",
        },
        {
            "start_frame": 150,
            "end_frame": 179,
            "duration_seconds": 1.0,
            "motion_state": "walking",
            "direction": "forward",
            "is_turning": False,
            "composite_label": "walking_forward",
        },
        {
            "start_frame": 180,
            "end_frame": 209,
            "duration_seconds": 1.0,
            "motion_state": "idle",
            "direction": "stationary",
            "is_turning": False,
            "composite_label": "idle_stationary",
        },
    ]


@pytest.fixture
def sample_transitions():
    """Sample segment transitions for testing."""
    return [
        {
            "from_frame": 29,
            "to_frame": 30,
            "gap_seconds": 0.0,
            "from_state": "idle_stationary",
            "to_state": "walking_forward",
            "transition_type": "start_moving",
        },
        {
            "from_frame": 89,
            "to_frame": 90,
            "gap_seconds": 0.0,
            "from_state": "walking_forward",
            "to_state": "running_forward",
            "transition_type": "accelerate",
        },
        {
            "from_frame": 149,
            "to_frame": 150,
            "gap_seconds": 0.0,
            "from_state": "running_forward",
            "to_state": "walking_forward",
            "transition_type": "decelerate",
        },
        {
            "from_frame": 179,
            "to_frame": 180,
            "gap_seconds": 0.0,
            "from_state": "walking_forward",
            "to_state": "idle_stationary",
            "transition_type": "stop_moving",
        },
    ]


@pytest.fixture
def sample_turning_events():
    """Sample turning events for testing."""
    return [
        {
            "start_frame": 100,
            "end_frame": 130,
            "total_rotation_degrees": 85.0,
            "severity": "sharp",
            "direction": "left",
        },
    ]


@pytest.fixture
def sample_root_motion_summary():
    """Sample root motion summary for testing."""
    return {
        "total_distance": 150.5,
        "displacement": 120.3,
        "dominant_direction": "forward",
    }


@pytest.fixture
def sample_gait_summary():
    """Sample gait summary for testing."""
    return {
        "avg_stride_length": 1.2,
        "avg_stride_duration": 0.6,
        "cadence": 100.0,
    }


# ==============================================================================
# TEST SEGMENT DESCRIPTION
# ==============================================================================


@pytest.mark.unit
class TestDescribeSegment:
    """Test segment description generation."""

    def test_describe_segment_with_template_match(self):
        """Test segment description with exact template match."""
        segment = {
            "composite_label": "walking_forward",
            "motion_state": "walking",
            "direction": "forward",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert description == "walking forward"

    def test_describe_segment_with_turning(self):
        """Test segment description with turning qualifier."""
        segment = {
            "composite_label": "running_forward",
            "motion_state": "running",
            "direction": "forward",
            "is_turning": True,
        }
        description = describe_segment(segment)
        assert "while turning" in description

    def test_describe_segment_fallback_stationary(self):
        """Test segment description fallback for stationary motion."""
        segment = {
            "composite_label": "custom_stationary",
            "motion_state": "crouching",
            "direction": "stationary",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert "in place" in description

    def test_describe_segment_fallback_forward(self):
        """Test segment description fallback for forward motion."""
        segment = {
            "composite_label": "custom_forward",
            "motion_state": "jogging",
            "direction": "forward",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert "jogging forward" in description

    def test_describe_segment_fallback_backward(self):
        """Test segment description fallback for backward motion."""
        segment = {
            "composite_label": "custom_backward",
            "motion_state": "walking",
            "direction": "backward",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert "walking backward" in description

    def test_describe_segment_fallback_strafe(self):
        """Test segment description fallback for strafe motion."""
        segment = {
            "composite_label": "custom_strafe",
            "motion_state": "walking",
            "direction": "strafe_left",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert "while moving" in description
        assert "strafe left" in description

    def test_describe_segment_missing_fields(self):
        """Test segment description with missing fields."""
        segment = {}
        description = describe_segment(segment)
        # Should not crash, should handle gracefully
        assert isinstance(description, str)
        assert len(description) > 0


# ==============================================================================
# TEST TRANSITION DESCRIPTION
# ==============================================================================


@pytest.mark.unit
class TestDescribeTransition:
    """Test transition description generation."""

    def test_describe_transition_start_moving(self):
        """Test description of start_moving transition."""
        transition = {"transition_type": "start_moving"}
        description = describe_transition(transition)
        assert description == "begins moving"

    def test_describe_transition_stop_moving(self):
        """Test description of stop_moving transition."""
        transition = {"transition_type": "stop_moving"}
        description = describe_transition(transition)
        assert description == "comes to a stop"

    def test_describe_transition_accelerate(self):
        """Test description of accelerate transition."""
        transition = {"transition_type": "accelerate"}
        description = describe_transition(transition)
        assert description == "accelerates"

    def test_describe_transition_decelerate(self):
        """Test description of decelerate transition."""
        transition = {"transition_type": "decelerate"}
        description = describe_transition(transition)
        assert description == "decelerates"

    def test_describe_transition_unknown(self):
        """Test description of unknown transition type."""
        transition = {"transition_type": "custom_transition"}
        description = describe_transition(transition)
        assert description == "transitions"

    def test_describe_transition_missing_type(self):
        """Test description with missing transition_type."""
        transition = {}
        description = describe_transition(transition)
        assert description == "transitions"


# ==============================================================================
# TEST NARRATIVE GENERATION
# ==============================================================================


@pytest.mark.unit
class TestGenerateNarrativeSummary:
    """Test narrative summary generation."""

    def test_generate_narrative_basic(self, sample_segments, sample_transitions, sample_turning_events):
        """Test basic narrative generation."""
        narrative = generate_narrative_summary(sample_segments, sample_transitions, sample_turning_events)
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        assert narrative.endswith(".")
        assert "character" in narrative.lower()

    def test_generate_narrative_includes_transitions(self, sample_segments, sample_transitions, sample_turning_events):
        """Test narrative includes transition descriptions."""
        narrative = generate_narrative_summary(sample_segments, sample_transitions, sample_turning_events)
        assert "begins moving" in narrative or "start" in narrative.lower()

    def test_generate_narrative_includes_turning(self, sample_segments, sample_transitions, sample_turning_events):
        """Test narrative includes turning events."""
        narrative = generate_narrative_summary(sample_segments, sample_transitions, sample_turning_events)
        # Turning event is in frame 100-130, which overlaps with segment 90-149
        assert "turn" in narrative.lower() or "left" in narrative.lower()

    def test_generate_narrative_ends_with_idle(self):
        """Test narrative ending when animation ends in idle."""
        segments = [
            {"motion_state": "walking", "direction": "forward", "is_turning": False, "start_frame": 0, "end_frame": 29},
            {
                "motion_state": "idle",
                "direction": "stationary",
                "is_turning": False,
                "start_frame": 30,
                "end_frame": 59,
            },
        ]
        transitions = []
        turning_events = []
        narrative = generate_narrative_summary(segments, transitions, turning_events)
        assert "rest" in narrative.lower()

    def test_generate_narrative_empty_segments(self):
        """Test narrative generation with empty segments."""
        narrative = generate_narrative_summary([], [], [])
        # Should handle gracefully, possibly return empty or minimal narrative
        assert isinstance(narrative, str)

    def test_generate_narrative_single_segment(self):
        """Test narrative with single segment."""
        segments = [
            {"motion_state": "idle", "direction": "stationary", "is_turning": False, "start_frame": 0, "end_frame": 59}
        ]
        narrative = generate_narrative_summary(segments, [], [])
        assert isinstance(narrative, str)
        assert "idle" in narrative.lower() or "still" in narrative.lower()


# ==============================================================================
# TEST ANIMATION CLASSIFICATION
# ==============================================================================


@pytest.mark.unit
class TestClassifyAnimationType:
    """Test animation type classification."""

    def test_classify_static_pose(self):
        """Test classification of static pose (mostly idle)."""
        segments = [{"motion_state": "idle"} for _ in range(10)]
        classification = classify_animation_type(segments)
        assert classification["type"] == "static_pose"
        assert classification["confidence"] > 0.8
        assert classification["dominant_state"] == "idle"

    def test_classify_walk_cycle(self):
        """Test classification of walk cycle (mostly walking)."""
        segments = [{"motion_state": "walking"} for _ in range(9)]
        segments.append({"motion_state": "idle"})  # 90% walking
        classification = classify_animation_type(segments)
        assert classification["type"] == "walk_cycle"
        assert classification["confidence"] > 0.8
        assert classification["dominant_state"] == "walking"

    def test_classify_run_cycle(self):
        """Test classification of run cycle (mostly running)."""
        segments = [{"motion_state": "running"} for _ in range(9)]
        segments.append({"motion_state": "idle"})
        classification = classify_animation_type(segments)
        assert classification["type"] == "run_cycle"
        assert classification["confidence"] > 0.8

    def test_classify_mixed_locomotion(self):
        """Test classification of mixed locomotion (walking and running)."""
        segments = [{"motion_state": "walking"} for _ in range(5)]
        segments.extend([{"motion_state": "running"} for _ in range(5)])
        classification = classify_animation_type(segments)
        assert classification["type"] == "mixed_locomotion"
        assert classification["confidence"] == 0.7

    def test_classify_acrobatic(self):
        """Test classification of acrobatic animation (includes jumping)."""
        segments = [
            {"motion_state": "walking"},
            {"motion_state": "jumping"},
            {"motion_state": "walking"},
            {"motion_state": "jumping"},
        ]
        classification = classify_animation_type(segments)
        assert classification["type"] == "acrobatic"

    def test_classify_varied_movement(self):
        """Test classification of varied movement (no clear dominant type)."""
        segments = [
            {"motion_state": "walking"},
            {"motion_state": "idle"},
            {"motion_state": "crouching"},
            {"motion_state": "standing"},
        ]
        classification = classify_animation_type(segments)
        assert classification["type"] == "varied_movement"

    def test_classify_empty_segments(self):
        """Test classification with no segments."""
        classification = classify_animation_type([])
        assert classification["type"] == "unknown"
        assert classification["confidence"] == 0.0

    def test_classify_custom_cycle(self):
        """Test classification of custom motion cycle."""
        segments = [{"motion_state": "crouching"} for _ in range(10)]
        classification = classify_animation_type(segments)
        assert classification["type"] == "crouching_cycle"
        assert classification["confidence"] > 0.8

    def test_classify_state_distribution(self):
        """Test that classification includes state distribution."""
        segments = [{"motion_state": "walking"} for _ in range(7)]
        segments.extend([{"motion_state": "running"} for _ in range(3)])
        classification = classify_animation_type(segments)
        assert "state_distribution" in classification
        assert classification["state_distribution"]["walking"] == 7
        assert classification["state_distribution"]["running"] == 3


# ==============================================================================
# TEST INTEGRATED MOTION SUMMARY
# ==============================================================================


@pytest.mark.unit
class TestGenerateMotionSummary:
    """Test complete motion summary generation."""

    def test_generate_motion_summary_complete(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test complete motion summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            assert "narrative" in result
            assert "classification" in result
            assert "metadata" in result
            assert "segment_descriptions" in result

            # Check narrative
            assert isinstance(result["narrative"], str)
            assert len(result["narrative"]) > 0

            # Check classification
            assert "type" in result["classification"]
            assert "confidence" in result["classification"]

            # Check metadata
            metadata = result["metadata"]
            assert "narrative_description" in metadata
            assert "classification" in metadata
            assert "statistics" in metadata
            assert "root_motion" in metadata
            assert "segments" in metadata

            # Check segment descriptions
            assert len(result["segment_descriptions"]) == len(sample_segments)

    def test_generate_motion_summary_with_gait(
        self,
        sample_segments,
        sample_transitions,
        sample_turning_events,
        sample_root_motion_summary,
        sample_gait_summary,
    ):
        """Test motion summary generation with gait data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                gait_summary=sample_gait_summary,
                output_dir=tmpdir,
            )

            assert "gait_metrics" in result["metadata"]
            assert result["metadata"]["gait_metrics"] == sample_gait_summary

    def test_generate_motion_summary_creates_files(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test that motion summary creates all expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            # Check that all files were created
            expected_files = [
                "motion_summary.txt",
                "animation_metadata.json",
                "segment_descriptions.csv",
                "motion_classification.json",
            ]

            for filename in expected_files:
                filepath = Path(tmpdir) / filename
                assert filepath.exists(), f"Expected file {filename} was not created"

    def test_generate_motion_summary_txt_content(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test content of motion_summary.txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            txt_path = Path(tmpdir) / "motion_summary.txt"
            content = txt_path.read_text()

            assert "ANIMATION MOTION SUMMARY" in content
            assert "Classification:" in content
            assert "Description:" in content
            assert "Statistics:" in content

    def test_generate_motion_summary_json_valid(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test that JSON files are valid and parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            # Check animation_metadata.json
            metadata_path = Path(tmpdir) / "animation_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
                assert "narrative_description" in metadata
                assert "classification" in metadata

            # Check motion_classification.json
            classification_path = Path(tmpdir) / "motion_classification.json"
            with open(classification_path) as f:
                classification = json.load(f)
                assert "type" in classification
                assert "confidence" in classification

    def test_generate_motion_summary_csv_valid(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test that CSV file is valid and readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            csv_path = Path(tmpdir) / "segment_descriptions.csv"
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == len(sample_segments)
                assert "segment_id" in rows[0]
                assert "description" in rows[0]

    def test_generate_motion_summary_statistics(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test that statistics are calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            stats = result["metadata"]["statistics"]
            assert stats["segment_count"] == len(sample_segments)
            assert stats["transition_count"] == len(sample_transitions)
            assert stats["turning_event_count"] == len(sample_turning_events)
            assert stats["total_duration"] > 0

    def test_generate_motion_summary_root_motion(
        self, sample_segments, sample_transitions, sample_turning_events, sample_root_motion_summary
    ):
        """Test that root motion data is included correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=sample_segments,
                transitions=sample_transitions,
                turning_events=sample_turning_events,
                root_motion_summary=sample_root_motion_summary,
                output_dir=tmpdir,
            )

            root_motion = result["metadata"]["root_motion"]
            assert root_motion["total_distance"] == 150.5
            assert root_motion["displacement"] == 120.3
            assert root_motion["dominant_direction"] == "forward"

    def test_generate_motion_summary_empty_inputs(self):
        """Test motion summary with empty inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=[],
                transitions=[],
                turning_events=[],
                root_motion_summary={},
                output_dir=tmpdir,
            )

            # Should handle gracefully without crashing
            assert "narrative" in result
            assert "classification" in result
            assert result["classification"]["type"] == "unknown"


# ==============================================================================
# TEST EDGE CASES
# ==============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_segment_description_with_underscores(self):
        """Test that underscores in motion states are handled."""
        segment = {
            "composite_label": "custom_label",
            "motion_state": "crouch_walking",
            "direction": "forward",
            "is_turning": False,
        }
        description = describe_segment(segment)
        assert "crouch walking" in description

    def test_narrative_with_no_transitions(self):
        """Test narrative generation with segments but no transitions."""
        segments = [
            {"motion_state": "walking", "direction": "forward", "is_turning": False, "start_frame": 0, "end_frame": 59}
        ]
        narrative = generate_narrative_summary(segments, [], [])
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_classification_with_ties(self):
        """Test classification when multiple states have equal counts."""
        segments = [{"motion_state": "walking"}, {"motion_state": "running"}]
        classification = classify_animation_type(segments)
        # Should pick one as dominant (implementation-dependent)
        assert classification["dominant_state"] in ["walking", "running"]

    def test_missing_root_motion_fields(self):
        """Test handling of incomplete root motion summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_motion_summary(
                segments=[
                    {
                        "start_frame": 0,
                        "end_frame": 29,
                        "duration_seconds": 1.0,
                        "motion_state": "idle",
                        "direction": "stationary",
                    }
                ],
                transitions=[],
                turning_events=[],
                root_motion_summary={},  # Empty root motion summary
                output_dir=tmpdir,
            )

            root_motion = result["metadata"]["root_motion"]
            assert root_motion["total_distance"] == 0
            assert root_motion["displacement"] == 0
            assert root_motion["dominant_direction"] == "unknown"
