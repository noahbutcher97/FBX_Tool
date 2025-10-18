"""
FBX Tool - GUI Entry Point
PyQt6-based graphical interface for FBX animation analysis.
Supports drag-and-drop, file selection, and modular analysis operations.
WITH ROBUST ERROR HANDLING - Continues execution even if individual steps fail.
"""

import os
import sys
import traceback

from PyQt6.QtCore import QRect, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fbx_tool.analysis.chain_analysis import analyze_chains
from fbx_tool.analysis.constraint_violation_detection import (
    analyze_constraint_violations,
)
from fbx_tool.analysis.directional_change_detection import analyze_directional_changes
from fbx_tool.analysis.dopesheet_export import export_dopesheet
from fbx_tool.analysis.fbx_loader import get_scene_metadata, load_fbx
from fbx_tool.analysis.foot_contact_analysis import analyze_foot_contacts
from fbx_tool.analysis.gait_analysis import analyze_gait
from fbx_tool.analysis.gait_summary import GaitSummaryAnalysis
from fbx_tool.analysis.joint_analysis import analyze_joints
from fbx_tool.analysis.motion_classification import generate_motion_summary
from fbx_tool.analysis.motion_transition_detection import analyze_motion_transitions
from fbx_tool.analysis.pose_validity_analysis import analyze_pose_validity
from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
from fbx_tool.analysis.temporal_segmentation import analyze_temporal_segmentation
from fbx_tool.analysis.utils import clear_trajectory_cache, ensure_output_dir
from fbx_tool.analysis.velocity_analysis import analyze_velocity
from fbx_tool.visualization.opengl_viewer import launch_skeleton_viewer


class AnalysisOperationsDialog(QDialog):
    """Dialog for selecting analysis operations to run."""

    OPERATIONS = [
        ("dopesheet", "Export Dopesheet", "Export frame-by-frame bone rotation data to CSV"),
        ("joints", "Analyze Joints", "Calculate joint metrics, stability, and IK suitability"),
        ("chains", "Analyze Chains", "Analyze bone chains for IK confidence and coherence"),
        ("gait", "Analyze Gait", "Detect stride segments and gait patterns"),
        ("velocity", "Analyze Velocity/Motion Quality", "Compute velocity, acceleration, jerk, and smoothness metrics"),
        ("foot_contact", "Analyze Foot Contacts", "Detect ground contacts, foot sliding, and penetration issues"),
        (
            "pose_validity",
            "Analyze Pose Validity",
            "Validate bone lengths, joint angles, and detect self-intersections",
        ),
        (
            "constraint_violations",
            "Analyze Constraint Violations",
            "Detect IK chain issues, hierarchy problems, and curve discontinuities",
        ),
        (
            "root_motion",
            "Analyze Root Motion",
            "Analyze root bone movement, direction, turning, and spatial displacement",
        ),
        (
            "directional_changes",
            "Detect Directional Changes",
            "Detect forward/backward/strafe movements and turning events",
        ),
        (
            "motion_transitions",
            "Detect Motion Transitions",
            "Classify locomotion states (idle/walk/run) and transitions",
        ),
        ("temporal_segmentation", "Temporal Segmentation", "Segment animation into coherent movement phrases"),
        ("motion_summary", "Generate Motion Summary", "Create natural language description of animation content"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Analysis Operations")
        self.setMinimumWidth(500)
        self.setMinimumHeight(350)

        self.checkboxes = {}
        self.selected_operations = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Select one or more analysis operations to run:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Checkboxes for each operation
        for op_id, op_name, op_desc in self.OPERATIONS:
            checkbox = QCheckBox(op_name)
            checkbox.setToolTip(op_desc)
            checkbox.setProperty("operation_id", op_id)
            self.checkboxes[op_id] = checkbox
            layout.addWidget(checkbox)

        layout.addSpacing(10)

        # Select All / Deselect All buttons
        select_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        select_buttons_layout.addWidget(deselect_all_btn)

        layout.addLayout(select_buttons_layout)

        layout.addSpacing(10)

        # Run / Cancel buttons
        button_layout = QHBoxLayout()
        run_btn = QPushButton("Run Selected")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._accept)
        button_layout.addWidget(run_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _select_all(self):
        """Select all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def _accept(self):
        """Collect selected operations and accept dialog."""
        self.selected_operations = [op_id for op_id, checkbox in self.checkboxes.items() if checkbox.isChecked()]

        if not self.selected_operations:
            # No operations selected, show warning
            self.parent().result.append("⚠ No analysis operations selected.")
            self.reject()
        else:
            self.accept()

    def get_selected_operations(self):
        """Return list of selected operation IDs."""
        return self.selected_operations


class RecentFilesDialog(QDialog):
    """Dialog for selecting multiple recent files with checkboxes."""

    def __init__(self, recent_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Recent Files")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.recent_files = recent_files
        self.checkboxes = []
        self.selected_files = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Select one or more files for batch processing:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        if not self.recent_files:
            no_files_label = QLabel("No recent files available")
            no_files_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(no_files_label)
        else:
            # Add checkbox for each file
            for filepath in self.recent_files:
                filename = os.path.basename(filepath)
                checkbox = QCheckBox(filename)
                checkbox.setToolTip(filepath)
                checkbox.setProperty("filepath", filepath)
                self.checkboxes.append(checkbox)
                scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Select All / Deselect All buttons
        select_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        select_buttons_layout.addWidget(deselect_all_btn)

        layout.addLayout(select_buttons_layout)

        # OK / Cancel buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Open Selected")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _select_all(self):
        """Select all checkboxes."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def _accept(self):
        """Collect selected files and accept dialog."""
        self.selected_files = [cb.property("filepath") for cb in self.checkboxes if cb.isChecked()]

        if not self.selected_files:
            # No files selected, show warning
            self.parent().result.setText("⚠ No files selected from recent files.")
            self.reject()
        else:
            self.accept()

    def get_selected_files(self):
        """Return list of selected file paths."""
        return self.selected_files


class AnalysisWorker(QThread):
    """
    Background worker thread for running analysis operations.

    Features robust error handling:
    - Individual try/except for each analysis step
    - Continues execution even if individual steps fail
    - Tracks errors and displays summary at completion
    - File-specific output directories
    """

    finished = pyqtSignal(object, object)  # (model, None) - scene is cleaned up in worker
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, fbx_file, operations):
        super().__init__()
        self.fbx_file = fbx_file
        self.operations = operations
        self._is_running = True
        self.errors = []

    def run(self):
        """Run analysis with robust error handling for each step."""
        # Clear trajectory cache to ensure fresh analysis for this file
        clear_trajectory_cache()

        # Create file-specific output directory
        base_name = os.path.splitext(os.path.basename(self.fbx_file))[0]
        output_dir = f"output/{base_name}/"
        ensure_output_dir(output_dir)

        # Get scene manager
        from fbx_tool.analysis.scene_manager import get_scene_manager

        scene_manager = get_scene_manager()
        scene_ref = None
        scene = None
        dopesheet_path = None
        joint_conf = {}
        chain_conf = {}
        stride_segments = []
        gait_summary = {}
        root_motion_summary = {}
        movement_segments = []
        turning_events = []
        motion_states = []
        temporal_segments = []
        segment_transitions = []
        motion_narrative = ""

        try:
            # STEP 1: Load FBX Scene (CRITICAL - must succeed)
            if not self._is_running:
                return

            self.progress.emit(f"\n{'='*50}")
            self.progress.emit(f"Analyzing: {os.path.basename(self.fbx_file)}")
            self.progress.emit(f"{'='*50}")
            self.progress.emit(f"Loading FBX scene...")

            try:
                # Get scene from scene manager (may be cached if GUI/visualizer has it)
                scene_ref = scene_manager.get_scene(self.fbx_file)
                scene = scene_ref.scene
                metadata = get_scene_metadata(scene)
                self.progress.emit(f"  Duration: {metadata.get('duration', 0):.2f}s")
                self.progress.emit(f"  Frame Rate: {metadata.get('frame_rate', 0):.2f} FPS")
                self.progress.emit(f"  Bone Count: {metadata.get('bone_count', 0)}")
            except Exception as e:
                error_msg = f"✗ CRITICAL: Failed to load FBX scene: {str(e)}"
                self.progress.emit(error_msg)
                self.error.emit(error_msg)
                return

            # STEP 2: Export Dopesheet (NON-CRITICAL)
            if "dopesheet" in self.operations and self._is_running:
                self.progress.emit("\nExporting dopesheet...")
                try:
                    dopesheet_path = os.path.join(output_dir, "dopesheet.csv")
                    export_dopesheet(scene, dopesheet_path)
                    self.progress.emit(f"  ✓ Dopesheet exported")
                except Exception as e:
                    error_msg = f"  ✗ Dopesheet export failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Dopesheet Export", str(e)))

            # STEP 3: Joint Analysis (NON-CRITICAL)
            if "joints" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing joints...")
                try:
                    joint_conf = analyze_joints(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ {len(joint_conf)} joints analyzed")
                except Exception as e:
                    error_msg = f"  ✗ Joint analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Joint Analysis", str(e)))

            # STEP 4: Chain Analysis (NON-CRITICAL)
            if "chains" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing chains...")
                try:
                    chain_conf = analyze_chains(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ {len(chain_conf)} chains analyzed")
                except Exception as e:
                    error_msg = f"  ✗ Chain analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Chain Analysis", str(e)))

            # STEP 5: Gait Analysis (NON-CRITICAL)
            if "gait" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing gait patterns...")
                try:
                    stride_segments, gait_summary = analyze_gait(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ {len(stride_segments)} stride segments detected")
                except Exception as e:
                    error_msg = f"  ✗ Gait analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Gait Analysis", str(e)))

            # STEP 6: Velocity/Motion Quality Analysis (NON-CRITICAL)
            if "velocity" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing velocity and motion quality (translational + rotational)...")
                try:
                    velocity_results = analyze_velocity(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ Velocity analysis complete")
                    self.progress.emit(f"    - {velocity_results['total_bones']} bones analyzed (position + rotation)")
                    self.progress.emit(f"    - {velocity_results['velocity_spikes_count']} velocity spikes detected")
                    self.progress.emit(
                        f"    - {velocity_results['acceleration_peaks_count']} acceleration peaks detected"
                    )
                    self.progress.emit(f"    - {velocity_results['jerk_spikes_count']} jerk spikes detected")
                    self.progress.emit(f"    - {velocity_results['chains_analyzed']} chains analyzed")
                    self.progress.emit(
                        f"    - {velocity_results['temporal_data_points']} temporal data points recorded"
                    )
                    if velocity_results["high_jitter_bones"] > 0:
                        self.progress.emit(f"    - ⚠ {velocity_results['high_jitter_bones']} bones with high jitter")
                except Exception as e:
                    error_msg = f"  ✗ Velocity analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Velocity Analysis", str(e)))

            # STEP 7: Foot Contact Analysis (NON-CRITICAL)
            if "foot_contact" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing foot contacts and ground interaction...")
                try:
                    foot_contact_results = analyze_foot_contacts(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ Foot contact analysis complete")
                    self.progress.emit(f"    - {foot_contact_results['feet_detected']} feet detected")
                    self.progress.emit(f"    - Ground height: {foot_contact_results['ground_height']:.2f} units")
                    self.progress.emit(f"    - {foot_contact_results['total_contacts']} contact events analyzed")
                    if foot_contact_results["contacts_with_sliding"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {foot_contact_results['contacts_with_sliding']} contacts with foot sliding"
                        )
                        self.progress.emit(
                            f"    - Total sliding distance: {foot_contact_results['total_sliding_distance']:.2f} units"
                        )
                except Exception as e:
                    error_msg = f"  ✗ Foot contact analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Foot Contact Analysis", str(e)))

            # STEP 8: Pose Validity Analysis (NON-CRITICAL)
            if "pose_validity" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing pose validity (bone lengths, joint angles, intersections)...")
                try:
                    pose_results = analyze_pose_validity(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ Pose validity analysis complete")
                    self.progress.emit(f"    - {pose_results['total_bones']} bones validated")
                    self.progress.emit(f"    - Overall validity score: {pose_results['overall_validity_score']:.2f}")
                    if pose_results["bones_with_length_violations"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {pose_results['bones_with_length_violations']} bones with length violations"
                        )
                    if pose_results["bones_with_angle_violations"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {pose_results['bones_with_angle_violations']} bones with angle violations"
                        )
                    if pose_results["self_intersections_detected"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {pose_results['self_intersections_detected']} self-intersections detected"
                        )
                except Exception as e:
                    error_msg = f"  ✗ Pose validity analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Pose Validity Analysis", str(e)))

            # STEP 9: Constraint Violation Detection (NON-CRITICAL)
            if "constraint_violations" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing constraint violations (IK chains, hierarchy, curves)...")
                try:
                    constraint_results = analyze_constraint_violations(scene, output_dir=output_dir)
                    self.progress.emit(f"  ✓ Constraint violation analysis complete")
                    self.progress.emit(f"    - {constraint_results['total_chains']} chains analyzed")
                    self.progress.emit(
                        f"    - Overall constraint score: {constraint_results['overall_constraint_score']:.2f}"
                    )
                    if constraint_results["ik_violations"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {constraint_results['ik_violations']} IK chain violations detected"
                        )
                    if constraint_results["hierarchy_violations"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {constraint_results['hierarchy_violations']} hierarchy violations detected"
                        )
                    if constraint_results["curve_discontinuities"] > 0:
                        self.progress.emit(
                            f"    - ⚠ {constraint_results['curve_discontinuities']} curve discontinuities detected"
                        )
                except Exception as e:
                    error_msg = f"  ✗ Constraint violation analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Constraint Violation Analysis", str(e)))

            # STEP 10: Root Motion Analysis (NON-CRITICAL)
            if "root_motion" in self.operations and self._is_running:
                self.progress.emit("\nAnalyzing root motion (movement, direction, turning)...")
                try:
                    root_motion_result = analyze_root_motion(scene, output_dir=output_dir)
                    root_motion_summary = root_motion_result
                    self.progress.emit(f"  ✓ Root motion analysis complete")
                    self.progress.emit(
                        f"    - Total distance traveled: {root_motion_summary.get('total_distance', 0):.2f} units"
                    )
                    self.progress.emit(
                        f"    - Net displacement: {root_motion_summary.get('displacement', 0):.2f} units"
                    )
                    self.progress.emit(
                        f"    - Dominant direction: {root_motion_summary.get('dominant_direction', 'unknown')}"
                    )
                except Exception as e:
                    error_msg = f"  ✗ Root motion analysis failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Root Motion Analysis", str(e)))

            # STEP 11: Directional Change Detection (NON-CRITICAL)
            if "directional_changes" in self.operations and self._is_running:
                self.progress.emit("\nDetecting directional changes and turning events...")
                try:
                    directional_result = analyze_directional_changes(scene, output_dir=output_dir)
                    movement_segments = directional_result.get("movement_segments", [])
                    turning_events = directional_result.get("turning_events", [])
                    self.progress.emit(f"  ✓ Directional change detection complete")
                    self.progress.emit(f"    - {len(movement_segments)} movement segments detected")
                    self.progress.emit(f"    - {len(turning_events)} turning events detected")
                except Exception as e:
                    error_msg = f"  ✗ Directional change detection failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Directional Change Detection", str(e)))

            # STEP 12: Motion Transition Detection (NON-CRITICAL)
            if "motion_transitions" in self.operations and self._is_running:
                self.progress.emit("\nDetecting motion transitions (idle/walk/run states)...")
                try:
                    transition_result = analyze_motion_transitions(scene, output_dir=output_dir)
                    motion_states = transition_result.get("motion_states", [])
                    self.progress.emit(f"  ✓ Motion transition detection complete")
                    self.progress.emit(f"    - {len(motion_states)} motion state segments detected")
                    state_counts = {}
                    for state in motion_states:
                        label = state.get("label", "unknown")
                        state_counts[label] = state_counts.get(label, 0) + 1
                    for state_type, count in state_counts.items():
                        self.progress.emit(f"    - {state_type}: {count} segments")
                except Exception as e:
                    error_msg = f"  ✗ Motion transition detection failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Motion Transition Detection", str(e)))

            # STEP 13: Temporal Segmentation (NON-CRITICAL, depends on steps 11 & 12)
            if "temporal_segmentation" in self.operations and self._is_running:
                self.progress.emit("\nCreating temporal segmentation...")
                try:
                    metadata = get_scene_metadata(scene)
                    frame_rate = metadata.get("frame_rate", 30.0)
                    duration = metadata.get("duration", 0)
                    total_frames = int(duration * frame_rate) + 1

                    # Handle missing motion states: create default continuous segment
                    if not motion_states:
                        self.progress.emit("  ℹ No motion transitions detected - creating continuous segment")
                        motion_states = [
                            {
                                "start_frame": 0,
                                "end_frame": total_frames - 1,
                                "duration_frames": total_frames,
                                "duration_seconds": duration,
                                "motion_state": "continuous",
                            }
                        ]

                    # Handle missing movement segments: create default full-range segment
                    if not movement_segments:
                        self.progress.emit("  ℹ No directional changes detected - creating continuous segment")
                        movement_segments = [
                            {
                                "start_frame": 0,
                                "end_frame": total_frames - 1,
                                "direction": "unknown",
                            }
                        ]

                    segmentation_result = analyze_temporal_segmentation(
                        motion_states, movement_segments, turning_events, frame_rate, output_dir=output_dir
                    )
                    temporal_segments = segmentation_result.get("segments", [])
                    segment_transitions = segmentation_result.get("transitions", [])
                    self.progress.emit(f"  ✓ Temporal segmentation complete")
                    self.progress.emit(f"    - {len(temporal_segments)} unified segments created")
                    self.progress.emit(f"    - {len(segment_transitions)} transitions detected")
                except Exception as e:
                    error_msg = f"  ✗ Temporal segmentation failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Temporal Segmentation", str(e)))

            # STEP 14: Motion Summary Generation (NON-CRITICAL, depends on step 13)
            if "motion_summary" in self.operations and self._is_running:
                self.progress.emit("\nGenerating motion classification and natural language summary...")
                try:
                    # Need temporal_segments and segment_transitions from previous step
                    if not temporal_segments:
                        self.progress.emit("  ⚠ Skipping: requires temporal segmentation to run first")
                    else:
                        summary_result = generate_motion_summary(
                            segments=temporal_segments,
                            transitions=segment_transitions,
                            turning_events=turning_events,
                            root_motion_summary=root_motion_summary,
                            gait_summary=gait_summary,
                            output_dir=output_dir,
                        )
                        motion_narrative = summary_result.get("narrative", "")
                        classification = summary_result.get("classification", {})
                        self.progress.emit(f"  ✓ Motion summary generated")
                        self.progress.emit(f"    - Animation type: {classification.get('type', 'unknown')}")
                        self.progress.emit(f"    - Confidence: {classification.get('confidence', 0):.1%}")
                        self.progress.emit(f"\n  Narrative: {motion_narrative}")
                except Exception as e:
                    error_msg = f"  ✗ Motion summary generation failed: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Motion Summary Generation", str(e)))

            # STEP 15: Create Summary Model (even with partial data)
            if self._is_running:
                self.progress.emit("\nGenerating analysis summary...")
                try:
                    model = GaitSummaryAnalysis(
                        fbx_path=self.fbx_file,
                        dopesheet_path=dopesheet_path,
                        gait_summary=gait_summary,
                        chain_conf=chain_conf,
                        joint_conf=joint_conf,
                        stride_segments=stride_segments,
                    )

                    json_path = os.path.join(output_dir, "analysis_summary.json")
                    model.to_json(json_path)
                    self.progress.emit(f"  ✓ Summary saved")

                    # Report any errors that occurred
                    if self.errors:
                        self.progress.emit(f"\n⚠ {len(self.errors)} step(s) had errors:")
                        for step, msg in self.errors:
                            self.progress.emit(f"    - {step}: {msg}")
                        self.progress.emit(f"\nPartial results saved to: {output_dir}")

                    self.finished.emit(model, None)  # Don't pass scene to prevent memory accumulation

                except Exception as e:
                    error_msg = f"  ✗ Failed to create analysis model: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Model Creation", str(e)))
                    self.finished.emit(None, None)  # Don't pass scene to prevent memory accumulation

        except Exception as e:
            error_msg = f"✗ Unexpected error: {str(e)}"
            self.progress.emit(error_msg)
            self.progress.emit(f"\nTraceback:\n{traceback.format_exc()}")
            self.error.emit(error_msg)
        finally:
            # CRITICAL: Release scene reference (scene manager handles cleanup)
            if scene_ref is not None:
                scene_ref.release()
                self.progress.emit("\n🧹 Scene reference released")

    def stop(self):
        """Request thread to stop."""
        self._is_running = False


class DropZoneOverlay(QWidget):
    """Overlay widget that shows split drop zones during drag operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAcceptDrops(False)  # Parent handles drops
        self.hover_zone = None  # 'left' or 'right' or None

    def paintEvent(self, event):
        """Draw the split zones with highlighting."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get widget dimensions
        width = self.width()
        height = self.height()
        mid = width // 2

        # Define zone rectangles
        left_zone = QRect(0, 0, mid, height)
        right_zone = QRect(mid, 0, width - mid, height)

        # Colors
        replace_color = QColor(255, 100, 100, 100)  # Red-ish for replace
        add_color = QColor(100, 200, 100, 100)  # Green-ish for add
        replace_hover = QColor(255, 100, 100, 150)
        add_hover = QColor(100, 200, 100, 150)

        # Draw left zone (Replace)
        if self.hover_zone == "left":
            painter.fillRect(left_zone, replace_hover)
        else:
            painter.fillRect(left_zone, replace_color)

        # Draw right zone (Add)
        if self.hover_zone == "right":
            painter.fillRect(right_zone, add_hover)
        else:
            painter.fillRect(right_zone, add_color)

        # Draw divider line
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawLine(mid, 0, mid, height)

        # Draw text labels
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))

        # Left label
        painter.drawText(left_zone, Qt.AlignmentFlag.AlignCenter, "Drop to\nREPLACE BATCH")

        # Right label
        painter.drawText(right_zone, Qt.AlignmentFlag.AlignCenter, "Drop to\nADD TO BATCH")


class FBXAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FBX Tool - Animation Analyzer")
        self.setGeometry(300, 300, 750, 500)
        self.setAcceptDrops(True)
        self.fbx_files = []
        self.worker = None
        self.viewer_window = None  # Store viewer reference

        # Scene manager for reference-counted scene lifecycle
        from fbx_tool.analysis.scene_manager import get_scene_manager

        self.scene_manager = get_scene_manager()
        self.active_scene_refs = {}  # {filepath: FBXSceneReference} - GUI holds refs to keep scenes cached

        # Drag-and-drop overlay
        self.drop_overlay = None

        # Batch processing state
        self.file_queue = []
        self.current_operations = []
        self.batch_results = []
        self.total_files = 0
        self.completed_files = 0

        # Settings for recent files
        self.settings = QSettings("FBXTool", "FBXAnalyzer")
        self.max_recent_files = 10

        self.initUI()
        self._update_recent_files_menu()

    def initUI(self):
        layout = QVBoxLayout()

        # Instructions
        self.label = QLabel("Drag & drop FBX file(s) here, or use 'Load Files' button below.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAcceptDrops(False)  # Ensure label doesn't consume drops
        layout.addWidget(self.label)

        # Results display
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setAcceptDrops(False)  # Disable drops on text edit to allow parent to handle
        layout.addWidget(self.result)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setAcceptDrops(False)  # Ensure progress bar doesn't consume drops
        layout.addWidget(self.progress_bar)

        # File Import Group
        import_group = QGroupBox("File Import")
        import_group.setAcceptDrops(False)
        import_layout = QVBoxLayout()

        file_row1 = QHBoxLayout()
        # Load Files button (replaces batch)
        self.load_files_btn = QPushButton("🆕 Load Files")
        self.load_files_btn.clicked.connect(self.openFileDialog)
        self.load_files_btn.setAcceptDrops(False)
        file_row1.addWidget(self.load_files_btn)

        # Add Files button (adds to batch) - hidden when no files loaded
        self.add_files_btn = QPushButton("➕ Add Files")
        self.add_files_btn.clicked.connect(lambda: self.openFileDialog(add_to_batch=True))
        self.add_files_btn.setAcceptDrops(False)
        self.add_files_btn.setVisible(False)  # Hidden initially
        file_row1.addWidget(self.add_files_btn)
        import_layout.addLayout(file_row1)

        file_row2 = QHBoxLayout()
        # Recent Files button (replaces batch)
        self.recent_files_btn = QPushButton("🕐 Recent Files")
        self.recent_files_btn.setAcceptDrops(False)
        self.recent_files_btn.clicked.connect(self._open_recent_files_dialog)
        file_row2.addWidget(self.recent_files_btn)

        # Add Recent button (adds to batch) - hidden when no files loaded
        self.add_recent_btn = QPushButton("➕ Add Recent")
        self.add_recent_btn.clicked.connect(lambda: self._open_recent_files_dialog(add_to_batch=True))
        self.add_recent_btn.setAcceptDrops(False)
        self.add_recent_btn.setVisible(False)  # Hidden initially
        file_row2.addWidget(self.add_recent_btn)
        import_layout.addLayout(file_row2)

        file_row3 = QHBoxLayout()
        # Clear button
        self.clear_btn = QPushButton("🗑️ Clear Selection")
        self.clear_btn.clicked.connect(self.clearSelectedFiles)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setAcceptDrops(False)
        file_row3.addWidget(self.clear_btn)
        import_layout.addLayout(file_row3)

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        # Operations Group
        ops_group = QGroupBox("Operations")
        ops_group.setAcceptDrops(False)
        ops_layout = QVBoxLayout()

        action_row = QHBoxLayout()
        # Run Analysis button (opens dialog)
        self.run_analysis_btn = QPushButton("📊 Run Analysis")
        self.run_analysis_btn.clicked.connect(self._open_analysis_dialog)
        self.run_analysis_btn.setEnabled(False)
        self.run_analysis_btn.setAcceptDrops(False)
        action_row.addWidget(self.run_analysis_btn)

        # 3D Visualization button
        self.viz_btn = QPushButton("🎬 Visualize 3D")
        self.viz_btn.clicked.connect(self.launch_visualizer)
        self.viz_btn.setEnabled(False)
        self.viz_btn.setAcceptDrops(False)
        action_row.addWidget(self.viz_btn)
        ops_layout.addLayout(action_row)

        # Cancel button (separate row)
        cancel_row = QHBoxLayout()
        self.cancel_btn = QPushButton("⏹️ Cancel")
        self.cancel_btn.clicked.connect(self.cancelAnalysis)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setAcceptDrops(False)
        cancel_row.addWidget(self.cancel_btn)
        ops_layout.addLayout(cancel_row)

        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)

        self.setLayout(layout)

    def openFileDialog(self, add_to_batch=False):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open FBX Files", "", "FBX Files (*.fbx)")
        if fnames:
            # Track which files are new for marking
            new_files_set = set()

            # Add to batch or replace
            if add_to_batch and self.fbx_files:
                # Add files to existing batch, avoiding duplicates
                existing_set = set(self.fbx_files)
                new_files = [f for f in fnames if f not in existing_set]

                if not new_files:
                    self.result.append("\n⚠ All selected files are already in the batch.")
                    return

                self.fbx_files.extend(new_files)
                new_files_set = set(new_files)
                action_text = "Added to Batch"
            else:
                # Replace existing batch
                self.fbx_files = fnames
                new_files_set = set(fnames)
                action_text = "Files Loaded"

            # Acquire scene references for loaded files
            self._acquire_scene_references(new_files_set)

            # Update label
            total_count = len(self.fbx_files)
            if total_count == 1:
                self.label.setText(f"Selected: {os.path.basename(self.fbx_files[0])}")
            else:
                self.label.setText(f"Selected {total_count} file(s) in batch")

            self.enableOperationButtons(True)
            self.viz_btn.setEnabled(True)  # Enable viz immediately
            self.clear_btn.setEnabled(True)  # Enable clear button

            # Update Add buttons visibility
            self._update_add_buttons_visibility()

            # Clear result and show ALL animations in batch
            self.result.clear()
            self.result.append(f"📂 Current Batch ({total_count} animation{'s' if total_count != 1 else ''}):")
            for fname in self.fbx_files:
                animation_name = os.path.splitext(os.path.basename(fname))[0]
                if fname in new_files_set:
                    self.result.append(f"  ✓ {animation_name} ← NEW")
                else:
                    self.result.append(f"  ✓ {animation_name}")

            if add_to_batch:
                self.result.append(f"\n✓ Added {len(new_files_set)} animation(s) to batch")
            else:
                self.result.append(f"\n✓ Import(s) Successful")

            # Add to recent files
            for fname in fnames:
                self._add_recent_file(fname)

    def clearSelectedFiles(self):
        """Clear the selected FBX files and reset the UI."""
        # Release all scene references held by GUI
        for filepath, scene_ref in self.active_scene_refs.items():
            scene_ref.release()
            print(f"Released scene reference: {filepath}")
        self.active_scene_refs.clear()

        self.fbx_files = []
        self.viewer_window = None
        self.label.setText("Drag & drop FBX file(s) here, or use 'Load Files' button below.")
        self.result.clear()
        self.result.setText("File selection cleared. Ready to select new file(s).")
        self.enableOperationButtons(False)
        self.viz_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Hide Add buttons when no files loaded
        self._update_add_buttons_visibility()

    def _acquire_scene_references(self, filepaths):
        """Acquire scene references for loaded files to keep them cached.

        This ensures scenes remain in memory even after analysis completes,
        preventing the need to reload files for subsequent operations.
        """
        for filepath in filepaths:
            if filepath not in self.active_scene_refs:
                self.active_scene_refs[filepath] = self.scene_manager.get_scene(filepath)
                print(f"📦 Acquired scene reference: {os.path.basename(filepath)}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

            # Show overlay
            if self.drop_overlay is None:
                self.drop_overlay = DropZoneOverlay(self)
                self.drop_overlay.setGeometry(self.rect())
                self.drop_overlay.show()

    def dragMoveEvent(self, event):
        """Track mouse position to highlight the correct zone."""
        if self.drop_overlay:
            mouse_x = event.position().x()
            mid = self.width() // 2

            if mouse_x < mid:
                self.drop_overlay.hover_zone = "left"
            else:
                self.drop_overlay.hover_zone = "right"

            self.drop_overlay.update()  # Trigger repaint

    def dragLeaveEvent(self, event):
        """Hide overlay when drag leaves the window."""
        if self.drop_overlay:
            self.drop_overlay.hide()
            self.drop_overlay.deleteLater()
            self.drop_overlay = None

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith(".fbx")]

        # Determine which zone was dropped in
        mouse_x = event.position().x()
        mid = self.width() // 2
        add_to_batch = mouse_x >= mid  # Right side = add to batch

        # Hide overlay
        if self.drop_overlay:
            self.drop_overlay.hide()
            self.drop_overlay.deleteLater()
            self.drop_overlay = None

        if files:
            # Track which files are new for marking
            new_files_set = set()

            # Add to batch or replace
            if add_to_batch and self.fbx_files:
                # Add files to existing batch, avoiding duplicates
                existing_set = set(self.fbx_files)
                new_files = [f for f in files if f not in existing_set]

                if not new_files:
                    self.result.append("\n⚠ All dropped files are already in the batch.")
                    return

                self.fbx_files.extend(new_files)
                new_files_set = set(new_files)
            else:
                # Replace existing batch
                self.fbx_files = files
                new_files_set = set(files)

            # Acquire scene references for loaded files
            self._acquire_scene_references(new_files_set)

            # Update label
            total_count = len(self.fbx_files)
            if total_count == 1:
                self.label.setText(f"Selected: {os.path.basename(self.fbx_files[0])}")
            else:
                self.label.setText(f"Selected {total_count} file(s) in batch")

            self.enableOperationButtons(True)
            self.viz_btn.setEnabled(True)  # Enable viz immediately
            self.clear_btn.setEnabled(True)  # Enable clear button

            # Update Add buttons visibility
            self._update_add_buttons_visibility()

            # Clear result and show ALL animations in batch
            self.result.clear()
            self.result.append(f"📂 Current Batch ({total_count} animation{'s' if total_count != 1 else ''}):")
            for filepath in self.fbx_files:
                animation_name = os.path.splitext(os.path.basename(filepath))[0]
                if filepath in new_files_set:
                    self.result.append(f"  ✓ {animation_name} ← NEW")
                else:
                    self.result.append(f"  ✓ {animation_name}")

            if add_to_batch:
                self.result.append(f"\n✓ Added {len(new_files_set)} animation(s) to batch via drag & drop")
            else:
                self.result.append(f"\n✓ Import(s) Successful via drag & drop")

            # Add to recent files
            for filepath in files:
                self._add_recent_file(filepath)

    def runOperation(self, operations):
        if not self.fbx_files:
            self.result.setText("No FBX files selected!")
            return

        # Initialize batch processing
        self.file_queue = self.fbx_files.copy()
        self.current_operations = operations
        self.batch_results = []
        self.total_files = len(self.file_queue)
        self.completed_files = 0

        # Update UI
        self.enableOperationButtons(False)
        self.load_files_btn.setEnabled(False)
        self.recent_files_btn.setEnabled(False)
        self.add_files_btn.setEnabled(False)
        self.add_recent_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.viz_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result.clear()

        # Show batch summary
        if self.total_files > 1:
            self.result.append(f"📦 Batch Processing: {self.total_files} files")
            self.result.append("=" * 50)

        # Start processing first file
        self._process_next_file()

    def _process_next_file(self):
        """Process the next file in the queue."""
        if not self.file_queue:
            # All files processed - show summary
            self._on_batch_complete()
            return

        # Get next file
        fbx_file = self.file_queue.pop(0)
        self.completed_files += 1

        # Update progress bar with determinate progress
        self.progress_bar.setRange(0, self.total_files)
        self.progress_bar.setValue(self.completed_files)

        # Show progress if batch
        if self.total_files > 1:
            self.result.append(
                f"\n[{self.completed_files}/{self.total_files}] Processing: {os.path.basename(fbx_file)}"
            )

        # Start worker for this file
        self.worker = AnalysisWorker(fbx_file, self.current_operations)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self._on_file_complete)
        self.worker.error.connect(self._on_file_error)
        self.worker.start()

    def _on_file_complete(self, model, scene):
        """Handle completion of a single file."""
        # Store result (scene is now None - already cleaned up in worker)
        current_file = self.fbx_files[self.completed_files - 1]
        self.batch_results.append({"file": current_file, "success": True, "model": model})

        # Note: scene is None (cleaned up in worker to prevent memory leaks)
        # Visualization disabled for batch processing
        self.current_scene = None

        # Process next file or finish
        if self.file_queue:
            self._process_next_file()
        else:
            self._on_batch_complete()

    def _on_file_error(self, error_msg):
        """Handle error for a single file."""
        current_file = self.fbx_files[self.completed_files - 1]
        self.batch_results.append({"file": current_file, "success": False, "error": error_msg})

        # Continue with next file in batch mode
        if self.file_queue:
            self._process_next_file()
        else:
            self._on_batch_complete()

    def _on_batch_complete(self):
        """Handle completion of entire batch."""
        self.resetUI()

        # Show batch summary
        self.result.append("\n" + "=" * 50)
        if self.total_files > 1:
            self.result.append("📦 Batch Processing Complete!")
            self.result.append("=" * 50)

            successful = sum(1 for r in self.batch_results if r["success"])
            failed = self.total_files - successful

            self.result.append(f"  Total Files: {self.total_files}")
            self.result.append(f"  ✓ Successful: {successful}")
            if failed > 0:
                self.result.append(f"  ✗ Failed: {failed}")

            # Show individual results
            self.result.append("\n📋 Results:")
            for i, result in enumerate(self.batch_results, 1):
                filename = os.path.basename(result["file"])
                if result["success"]:
                    self.result.append(f"  {i}. ✓ {filename}")
                else:
                    self.result.append(f"  {i}. ✗ {filename} - {result.get('error', 'Unknown error')}")
        else:
            self.result.append("✓ Analysis Complete!")
            self.result.append("=" * 50)
            if self.batch_results and self.batch_results[0]["success"]:
                model = self.batch_results[0]["model"]
                if model:
                    self.result.append(f"  Gait Type: {model.get_gait_type()}")
                    self.result.append(f"  Stride Count: {model.get_stride_count()}")
                    base_name = os.path.splitext(os.path.basename(self.fbx_files[0]))[0]
                    self.result.append(f"  Results saved to: output/{base_name}/")

        self.result.append("=" * 50)

        # Enable visualization if we have files loaded (scene references held by GUI)
        if self.fbx_files and len(self.fbx_files) > 0:
            self.viz_btn.setEnabled(True)

    def cancelAnalysis(self):
        """Cancel current analysis and clear queue."""
        # Clear the queue to stop batch processing
        self.file_queue = []

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.result.append("\n⚠ Analysis cancelled by user.")
            self.resetUI()

    def updateProgress(self, message):
        self.result.append(message)

    def onAnalysisComplete(self, model, scene):
        """Legacy handler - redirects to batch handler."""
        self._on_file_complete(model, scene)

    def onAnalysisError(self, error_msg):
        """Legacy handler - redirects to batch handler."""
        self._on_file_error(error_msg)

    def launch_visualizer(self):
        """Launch 3D viewer using scene manager."""
        try:
            if not self.fbx_files:
                self.result.append("\n⚠ No FBX file selected. Please select a file first.")
                return

            filepath = self.fbx_files[0]

            # Get or create scene reference from scene manager
            self.result.append("\n⏳ Loading FBX for visualization...")
            if filepath not in self.active_scene_refs:
                # GUI doesn't have this scene yet - get reference and hold it
                self.active_scene_refs[filepath] = self.scene_manager.get_scene(filepath)
                self.result.append("✓ FBX loaded successfully")
            else:
                self.result.append("✓ Using cached scene")

            self.result.append("🎬 Launching 3D viewer...")
            # Launch visualizer - it will get its own reference from scene manager
            # Both GUI and visualizer hold references, so scene stays cached
            self.viewer_window = launch_skeleton_viewer(
                self.active_scene_refs[filepath].scene, fbx_files=self.fbx_files, scene_manager=self.scene_manager
            )

            if len(self.fbx_files) > 1:
                self.result.append(f"✓ Viewer window opened with {len(self.fbx_files)} animations")
                self.result.append("  Use Previous/Next buttons to switch between animations")
            else:
                self.result.append("✓ Viewer window opened")
        except Exception as e:
            import traceback

            self.result.append(f"\n✗ Visualization error: {str(e)}")
            self.result.append(f"Traceback:\n{traceback.format_exc()}")

    def _open_analysis_dialog(self):
        """Open dialog for selecting analysis operations."""
        dialog = AnalysisOperationsDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_operations = dialog.get_selected_operations()

            if selected_operations:
                self.runOperation(selected_operations)

    def enableOperationButtons(self, enabled):
        self.run_analysis_btn.setEnabled(enabled)

    def _update_add_buttons_visibility(self):
        """Show/hide Add buttons based on whether files are loaded."""
        has_files = len(self.fbx_files) > 0
        self.add_files_btn.setVisible(has_files)
        self.add_recent_btn.setVisible(has_files)

        # Update tooltips with current batch count
        if has_files:
            count = len(self.fbx_files)
            self.add_files_btn.setToolTip(
                f"Add more files to current batch ({count} file{'s' if count != 1 else ''} loaded)"
            )
            self.add_recent_btn.setToolTip(
                f"Add recent files to current batch ({count} file{'s' if count != 1 else ''} loaded)"
            )

    def resetUI(self):
        self.progress_bar.setVisible(False)
        self.enableOperationButtons(True)
        self.load_files_btn.setEnabled(True)
        self.recent_files_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        # Update Add button visibility based on current batch state
        self._update_add_buttons_visibility()

    def _add_recent_file(self, filepath):
        """Add a file to recent files list."""
        # Get current recent files
        recent_files = self.settings.value("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []

        # Remove if already exists (we'll add it to the front)
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to front
        recent_files.insert(0, filepath)

        # Limit to max
        recent_files = recent_files[: self.max_recent_files]

        # Save
        self.settings.setValue("recent_files", recent_files)

    def _update_recent_files_menu(self):
        """Dummy method for compatibility - no longer needed with dialog."""
        pass

    def _open_recent_files_dialog(self, add_to_batch=False):
        """Open dialog for selecting multiple recent files."""
        # Get recent files
        recent_files = self.settings.value("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []

        # Filter to only existing files
        recent_files = [f for f in recent_files if os.path.exists(f)]

        # Update settings with filtered list
        self.settings.setValue("recent_files", recent_files)

        if not recent_files:
            self.result.setText("No recent files available.")
            return

        # Show dialog
        dialog = RecentFilesDialog(recent_files, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_files = dialog.get_selected_files()

            if selected_files:
                # Track which files are new for marking
                new_files_set = set()

                # Add to batch or replace
                if add_to_batch and self.fbx_files:
                    # Add files to existing batch, avoiding duplicates
                    existing_set = set(self.fbx_files)
                    new_files = [f for f in selected_files if f not in existing_set]

                    if not new_files:
                        self.result.append("\n⚠ All selected files are already in the batch.")
                        return

                    self.fbx_files.extend(new_files)
                    new_files_set = set(new_files)
                else:
                    # Replace existing batch
                    self.fbx_files = selected_files
                    new_files_set = set(selected_files)

                # Acquire scene references for loaded files
                self._acquire_scene_references(new_files_set)

                # Update label
                total_count = len(self.fbx_files)
                if total_count == 1:
                    self.label.setText(f"Selected: {os.path.basename(self.fbx_files[0])}")
                else:
                    self.label.setText(f"Selected {total_count} file(s) in batch")

                self.enableOperationButtons(True)
                self.viz_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)

                # Update Add buttons visibility
                self._update_add_buttons_visibility()

                # Clear result and show ALL animations in batch
                self.result.clear()
                self.result.append(f"📂 Current Batch ({total_count} animation{'s' if total_count != 1 else ''}):")
                for filepath in self.fbx_files:
                    animation_name = os.path.splitext(os.path.basename(filepath))[0]
                    if filepath in new_files_set:
                        self.result.append(f"  ✓ {animation_name} ← NEW")
                    else:
                        self.result.append(f"  ✓ {animation_name}")

                if add_to_batch:
                    self.result.append(f"\n✓ Added {len(new_files_set)} animation(s) to batch")
                else:
                    self.result.append(f"\n✓ Import(s) Successful")

                # Move selected files to top of recent list
                for filepath in selected_files:
                    self._add_recent_file(filepath)

    def closeEvent(self, event):
        """Handle window close - cleanup FBX resources."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # Release all scene references held by GUI
        for filepath, scene_ref in self.active_scene_refs.items():
            scene_ref.release()
        self.active_scene_refs.clear()

        # Force cleanup any remaining scenes (safety net)
        self.scene_manager.cleanup_all()

        event.accept()


def main():
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    window = FBXAnalyzerApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
