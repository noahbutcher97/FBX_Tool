"""
Unit tests for temporal_segmentation module

Tests cover:
- Segment by continuity (label-based segmentation)
- Merge similar segments (gap bridging)
- Composite segment creation (combining motion state + direction + turning)
- Segment hierarchy building (coarse to fine)
- Segment transition classification
- Integrated temporal segmentation
- Edge cases (empty data, single segment, no transitions)
"""

import os
import tempfile

import numpy as np
import pytest

from fbx_tool.analysis.temporal_segmentation import (
    analyze_temporal_segmentation,
    build_segment_hierarchy,
    classify_segment_transition,
    create_composite_segments,
    merge_similar_segments,
    segment_by_continuity,
)


@pytest.mark.unit
class TestSegmentByContinuity:
    """Test continuity-based segmentation."""

    def test_segment_basic_state_sequence(self):
        """Should segment by continuous state labels."""
        # 30fps: 0-29 idle (1s), 30-59 walking (1s), 60-89 idle (1s)
        frame_data = []
        for i in range(90):
            if i < 30:
                state = "idle"
            elif i < 60:
                state = "walking"
            else:
                state = "idle"

            frame_data.append({"frame": i, "state": state})

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        # Should create 3 segments
        assert len(segments) == 3
        assert segments[0]["label"] == "idle"
        assert segments[0]["start_frame"] == 0
        assert segments[0]["end_frame"] == 29
        assert segments[1]["label"] == "walking"
        assert segments[2]["label"] == "idle"

    def test_segment_filters_short_durations(self):
        """Should filter out segments shorter than MIN_SEGMENT_DURATION_SECONDS."""
        # 30fps: 0-29 idle (1s), 30-31 walking (0.067s - too short), 32-61 idle (1s)
        frame_data = []
        for i in range(62):
            if i < 30:
                state = "idle"
            elif i < 32:
                state = "walking"  # Only 2 frames (0.067s < 0.5s threshold)
            else:
                state = "idle"

            frame_data.append({"frame": i, "state": state})

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        # Walking segment should be filtered out
        assert len(segments) == 2
        assert all(seg["label"] == "idle" for seg in segments)

    def test_segment_timing_accuracy(self):
        """Should accurately compute start/end frames and duration."""
        frame_data = [{"frame": i, "state": "walking"} for i in range(60)]

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        assert len(segments) == 1
        assert segments[0]["start_frame"] == 0
        assert segments[0]["end_frame"] == 59
        assert segments[0]["duration_seconds"] == 60 / 30.0  # 2.0 seconds

    def test_segment_empty_sequence(self):
        """Should handle empty frame data."""
        segments = segment_by_continuity([], frame_rate=30.0, label_key="state")

        assert segments == []

    def test_segment_single_frame(self):
        """Should handle single frame (too short to create segment)."""
        frame_data = [{"frame": 0, "state": "idle"}]

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        # Single frame = 0.033s < 0.5s threshold
        assert segments == []

    def test_segment_minimum_duration_boundary(self):
        """Should include segments at exactly MIN_SEGMENT_DURATION_SECONDS."""
        # 15 frames at 30fps = 0.5s exactly
        frame_data = [{"frame": i, "state": "walking"} for i in range(15)]

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        assert len(segments) == 1

    def test_segment_with_different_label_keys(self):
        """Should segment by any specified label key."""
        frame_data = []
        for i in range(60):
            direction = "forward" if i < 30 else "backward"
            frame_data.append({"frame": i, "direction": direction})

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="direction")

        assert len(segments) == 2
        assert segments[0]["label"] == "forward"
        assert segments[1]["label"] == "backward"


@pytest.mark.unit
class TestMergeSimilarSegments:
    """Test segment merging logic."""

    def test_merge_adjacent_identical_segments(self):
        """Should merge adjacent segments with same label."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 30, "end_frame": 59, "duration_seconds": 1.0, "label": "walking"},
        ]

        merged = merge_similar_segments(segments)

        # Should merge into single segment
        assert len(merged) == 1
        assert merged[0]["start_frame"] == 0
        assert merged[0]["end_frame"] == 59

    def test_keep_different_labels_separate(self):
        """Should not merge segments with different labels."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 30, "end_frame": 59, "duration_seconds": 1.0, "label": "running"},
        ]

        merged = merge_similar_segments(segments)

        # Should keep separate
        assert len(merged) == 2

    def test_merge_respects_max_gap_threshold(self):
        """Should only merge if gap is within MAX_MERGE_GAP_SECONDS."""
        # Gap of 10 frames at 30fps = 0.33s > MAX_MERGE_GAP_SECONDS (0.2s)
        segments = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 40, "end_frame": 69, "duration_seconds": 1.0, "label": "walking"},  # Gap too large
        ]

        merged = merge_similar_segments(segments)

        # Gap too large - should not merge
        assert len(merged) == 2

    def test_merge_small_gaps(self):
        """Should merge segments with small gaps."""
        # Gap of 1 frame at 30fps = 0.033s < MAX_MERGE_GAP_SECONDS (0.2s)
        segments = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 31, "end_frame": 60, "duration_seconds": 1.0, "label": "walking"},  # 1 frame gap
        ]

        merged = merge_similar_segments(segments)

        # Small gap - should merge
        assert len(merged) == 1
        assert merged[0]["end_frame"] == 60

    def test_merge_single_segment_returns_unchanged(self):
        """Should return single segment unchanged."""
        segments = [{"start_frame": 0, "end_frame": 59, "duration_seconds": 2.0, "label": "walking"}]

        merged = merge_similar_segments(segments)

        assert len(merged) == 1
        assert merged[0] == segments[0]

    def test_merge_empty_segments(self):
        """Should handle empty segment list."""
        merged = merge_similar_segments([])

        assert merged == []


@pytest.mark.unit
class TestCreateCompositeSegments:
    """Test composite segment creation."""

    def test_create_basic_composite_segments(self):
        """Should combine motion state and direction into composite segments."""
        motion_states = [{"start_frame": 0, "end_frame": 59, "duration_seconds": 2.0, "label": "walking"}]

        directions = [{"start_frame": 0, "end_frame": 59, "label": "forward"}]

        turning_events = []

        composites = create_composite_segments(motion_states, directions, turning_events, frame_rate=30.0)

        assert len(composites) == 1
        assert composites[0]["motion_state"] == "walking"
        assert composites[0]["direction"] == "forward"
        assert composites[0]["is_turning"] == False
        assert composites[0]["composite_label"] == "walking_forward"

    def test_composite_with_turning(self):
        """Should detect turning events within segments."""
        motion_states = [{"start_frame": 0, "end_frame": 59, "duration_seconds": 2.0, "label": "walking"}]

        directions = [{"start_frame": 0, "end_frame": 59, "label": "forward"}]

        turning_events = [{"start_frame": 20, "end_frame": 40, "direction": "left"}]

        composites = create_composite_segments(motion_states, directions, turning_events, frame_rate=30.0)

        assert composites[0]["is_turning"] == True
        assert composites[0]["composite_label"] == "walking_forward_turning"

    def test_composite_multiple_segments(self):
        """Should create composite for each motion state segment."""
        motion_states = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 30, "end_frame": 59, "duration_seconds": 1.0, "label": "running"},
        ]

        directions = [{"start_frame": 0, "end_frame": 59, "label": "forward"}]

        turning_events = []

        composites = create_composite_segments(motion_states, directions, turning_events, frame_rate=30.0)

        assert len(composites) == 2
        assert composites[0]["composite_label"] == "walking_forward"
        assert composites[1]["composite_label"] == "running_forward"

    def test_composite_direction_change_mid_motion(self):
        """Should detect direction changes within motion segments."""
        motion_states = [{"start_frame": 0, "end_frame": 59, "duration_seconds": 2.0, "label": "walking"}]

        directions = [
            {"start_frame": 0, "end_frame": 29, "label": "forward"},
            {"start_frame": 30, "end_frame": 59, "label": "backward"},
        ]

        turning_events = []

        composites = create_composite_segments(motion_states, directions, turning_events, frame_rate=30.0)

        # Mid-point is frame 29, which is in "forward" segment
        assert composites[0]["direction"] == "forward"

    def test_composite_handles_empty_motion_states(self):
        """Should handle empty motion state list."""
        composites = create_composite_segments([], [], [], frame_rate=30.0)

        assert composites == []

    def test_composite_unknown_direction_when_no_overlap(self):
        """Should use 'unknown' direction when no direction segment overlaps."""
        motion_states = [{"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"}]

        directions = [{"start_frame": 100, "end_frame": 129, "label": "forward"}]  # No overlap

        turning_events = []

        composites = create_composite_segments(motion_states, directions, turning_events, frame_rate=30.0)

        assert composites[0]["direction"] == "unknown"


@pytest.mark.unit
class TestBuildSegmentHierarchy:
    """Test hierarchical segment grouping."""

    def test_build_hierarchy_groups_same_motion_state(self):
        """Should group consecutive segments with same motion state."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "motion_state": "walking", "composite_label": "walking_forward"},
            {"start_frame": 30, "end_frame": 59, "motion_state": "walking", "composite_label": "walking_backward"},
            {"start_frame": 60, "end_frame": 89, "motion_state": "running", "composite_label": "running_forward"},
        ]

        hierarchy = build_segment_hierarchy(segments)

        # Walking segments grouped, running separate
        assert len(hierarchy) == 2
        assert hierarchy[0]["motion_state"] == "walking"
        assert len(hierarchy[0]["sub_segments"]) == 2
        assert hierarchy[1]["motion_state"] == "running"
        assert len(hierarchy[1]["sub_segments"]) == 1

    def test_hierarchy_timing_spans_all_sub_segments(self):
        """Should span from first to last sub-segment."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "motion_state": "walking", "composite_label": "walking_forward"},
            {"start_frame": 30, "end_frame": 59, "motion_state": "walking", "composite_label": "walking_forward"},
        ]

        hierarchy = build_segment_hierarchy(segments)

        assert hierarchy[0]["start_frame"] == 0
        assert hierarchy[0]["end_frame"] == 59

    def test_hierarchy_single_segment(self):
        """Should create single group for single segment."""
        segments = [
            {"start_frame": 0, "end_frame": 59, "motion_state": "walking", "composite_label": "walking_forward"}
        ]

        hierarchy = build_segment_hierarchy(segments)

        assert len(hierarchy) == 1
        assert len(hierarchy[0]["sub_segments"]) == 1

    def test_hierarchy_empty_segments(self):
        """Should handle empty segment list."""
        hierarchy = build_segment_hierarchy([])

        assert hierarchy == []

    def test_hierarchy_alternating_states(self):
        """Should create new group for each state change."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "motion_state": "walking", "composite_label": "walking_forward"},
            {"start_frame": 30, "end_frame": 59, "motion_state": "running", "composite_label": "running_forward"},
            {"start_frame": 60, "end_frame": 89, "motion_state": "walking", "composite_label": "walking_forward"},
        ]

        hierarchy = build_segment_hierarchy(segments)

        # 3 groups (walk, run, walk)
        assert len(hierarchy) == 3


@pytest.mark.unit
class TestClassifySegmentTransition:
    """Test segment transition classification."""

    def test_classify_start_moving(self):
        """Should classify idle → walking/running as start_moving."""
        from_seg = {"motion_state": "idle", "direction": "forward"}
        to_seg_walk = {"motion_state": "walking", "direction": "forward"}
        to_seg_run = {"motion_state": "running", "direction": "forward"}

        assert classify_segment_transition(from_seg, to_seg_walk) == "start_moving"
        assert classify_segment_transition(from_seg, to_seg_run) == "start_moving"

    def test_classify_stop_moving(self):
        """Should classify walking/running → idle as stop_moving."""
        from_seg_walk = {"motion_state": "walking", "direction": "forward"}
        from_seg_run = {"motion_state": "running", "direction": "forward"}
        to_seg = {"motion_state": "idle", "direction": "forward"}

        assert classify_segment_transition(from_seg_walk, to_seg) == "stop_moving"
        assert classify_segment_transition(from_seg_run, to_seg) == "stop_moving"

    def test_classify_accelerate(self):
        """Should classify walking → running as accelerate."""
        from_seg = {"motion_state": "walking", "direction": "forward"}
        to_seg = {"motion_state": "running", "direction": "forward"}

        assert classify_segment_transition(from_seg, to_seg) == "accelerate"

    def test_classify_decelerate(self):
        """Should classify running → walking as decelerate."""
        from_seg = {"motion_state": "running", "direction": "forward"}
        to_seg = {"motion_state": "walking", "direction": "forward"}

        assert classify_segment_transition(from_seg, to_seg) == "decelerate"

    def test_classify_direction_change(self):
        """Should classify same state with different direction as direction_change."""
        from_seg = {"motion_state": "walking", "direction": "forward"}
        to_seg = {"motion_state": "walking", "direction": "backward"}

        assert classify_segment_transition(from_seg, to_seg) == "direction_change"

    def test_classify_continue(self):
        """Should classify same state and direction as continue."""
        from_seg = {"motion_state": "walking", "direction": "forward"}
        to_seg = {"motion_state": "walking", "direction": "forward"}

        assert classify_segment_transition(from_seg, to_seg) == "continue"

    def test_classify_other_for_unknown(self):
        """Should classify unknown combinations as other."""
        from_seg = {"motion_state": "jumping", "direction": "forward"}
        to_seg = {"motion_state": "landing", "direction": "forward"}

        assert classify_segment_transition(from_seg, to_seg) == "other"


@pytest.mark.unit
class TestIntegratedAnalysis:
    """Test integrated temporal segmentation."""

    def test_analyze_complete_segmentation(self):
        """Should create unified temporal segmentation from multiple inputs."""
        motion_states = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 30, "end_frame": 59, "duration_seconds": 1.0, "label": "running"},
            {"start_frame": 60, "end_frame": 89, "duration_seconds": 1.0, "label": "walking"},
        ]

        movement_segments = [{"start_frame": 0, "end_frame": 89, "label": "forward"}]

        turning_events = [{"start_frame": 40, "end_frame": 50, "direction": "left"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_temporal_segmentation(
                motion_states, movement_segments, turning_events, frame_rate=30.0, output_dir=tmpdir
            )

        # Should create segments, hierarchy, and transitions
        assert "segments_count" in result
        assert "hierarchy_groups" in result
        assert "transitions_count" in result
        assert result["segments_count"] > 0
        assert isinstance(result["segments"], list)
        assert isinstance(result["hierarchy"], list)

    def test_analyze_handles_empty_inputs(self):
        """Should handle empty input gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_temporal_segmentation([], [], [], frame_rate=30.0, output_dir=tmpdir)

        assert result["segments_count"] == 0
        assert result["hierarchy_groups"] == 0
        assert result["transitions_count"] == 0

    def test_analyze_creates_csv_files(self):
        """Should create CSV output files."""
        motion_states = [{"start_frame": 0, "end_frame": 59, "duration_seconds": 2.0, "label": "walking"}]

        movement_segments = [{"start_frame": 0, "end_frame": 59, "label": "forward"}]

        turning_events = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_temporal_segmentation(
                motion_states, movement_segments, turning_events, frame_rate=30.0, output_dir=tmpdir
            )

            # Check CSV files were created
            assert os.path.exists(os.path.join(tmpdir, "temporal_segments.csv"))
            assert os.path.exists(os.path.join(tmpdir, "segment_hierarchy.csv"))
            # segment_transitions.csv only created if there are transitions


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_segment_all_same_label(self):
        """Should create single segment when all frames have same label."""
        frame_data = [{"frame": i, "state": "walking"} for i in range(60)]

        segments = segment_by_continuity(frame_data, frame_rate=30.0, label_key="state")

        assert len(segments) == 1
        assert segments[0]["label"] == "walking"

    def test_merge_maintains_segment_order(self):
        """Should maintain chronological order after merging."""
        segments = [
            {"start_frame": 0, "end_frame": 29, "duration_seconds": 1.0, "label": "walking"},
            {"start_frame": 30, "end_frame": 59, "duration_seconds": 1.0, "label": "running"},
            {"start_frame": 60, "end_frame": 89, "duration_seconds": 1.0, "label": "walking"},
        ]

        merged = merge_similar_segments(segments)

        # Should be in chronological order
        for i in range(len(merged) - 1):
            assert merged[i]["end_frame"] < merged[i + 1]["start_frame"]
