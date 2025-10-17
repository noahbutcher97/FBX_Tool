"""
FBX Tool - GUI Entry Point
PyQt6-based graphical interface for FBX animation analysis.
Supports drag-and-drop, file selection, and modular analysis operations.
WITH ROBUST ERROR HANDLING - Continues execution even if individual steps fail.
"""

import sys
import os
import traceback
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QHBoxLayout, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from fbx_tool.analysis.fbx_loader import load_fbx, get_scene_metadata
from fbx_tool.analysis.dopesheet_export import export_dopesheet
from fbx_tool.analysis.gait_analysis import analyze_gait
from fbx_tool.analysis.chain_analysis import analyze_chains
from fbx_tool.analysis.joint_analysis import analyze_joints
from fbx_tool.analysis.gait_summary import GaitSummaryAnalysis
from fbx_tool.analysis.utils import ensure_output_dir
from fbx_tool.visualization.opengl_viewer import launch_skeleton_viewer


class AnalysisWorker(QThread):
    """
    Background worker thread for running analysis operations.

    Features robust error handling:
    - Individual try/except for each analysis step
    - Continues execution even if individual steps fail
    - Tracks errors and displays summary at completion
    - File-specific output directories
    """
    finished = pyqtSignal(object, object)  # (model, scene)
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
        # Create file-specific output directory
        base_name = os.path.splitext(os.path.basename(self.fbx_file))[0]
        output_dir = f"output/{base_name}/"
        ensure_output_dir(output_dir)

        scene = None
        dopesheet_path = None
        joint_conf = {}
        chain_conf = {}
        stride_segments = []
        gait_summary = {}

        try:
            # STEP 1: Load FBX Scene (CRITICAL - must succeed)
            if not self._is_running:
                return

            self.progress.emit(f"\n{'='*50}")
            self.progress.emit(f"Analyzing: {os.path.basename(self.fbx_file)}")
            self.progress.emit(f"{'='*50}")
            self.progress.emit(f"Loading FBX scene...")

            try:
                scene = load_fbx(self.fbx_file)
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

            # STEP 6: Create Summary Model (even with partial data)
            if self._is_running:
                self.progress.emit("\nGenerating analysis summary...")
                try:
                    model = GaitSummaryAnalysis(
                        fbx_path=self.fbx_file,
                        dopesheet_path=dopesheet_path,
                        gait_summary=gait_summary,
                        chain_conf=chain_conf,
                        joint_conf=joint_conf,
                        stride_segments=stride_segments
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

                    self.finished.emit(model, scene)

                except Exception as e:
                    error_msg = f"  ✗ Failed to create analysis model: {str(e)}"
                    self.progress.emit(error_msg)
                    self.errors.append(("Model Creation", str(e)))
                    self.finished.emit(None, scene)

        except Exception as e:
            error_msg = f"✗ Unexpected error: {str(e)}"
            self.progress.emit(error_msg)
            self.progress.emit(f"\nTraceback:\n{traceback.format_exc()}")
            self.error.emit(error_msg)

    def stop(self):
        """Request thread to stop."""
        self._is_running = False


class FBXAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FBX Tool - Animation Analyzer")
        self.setGeometry(300, 300, 750, 500)
        self.setAcceptDrops(True)
        self.fbx_files = []
        self.worker = None
        self.current_scene = None
        self.viewer_window = None  # Store viewer reference
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Instructions
        self.label = QLabel("Drag & drop FBX file(s) here, or use 'Choose File(s)' button below.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Results display
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        layout.addWidget(self.result)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # File selection buttons
        file_hbox = QHBoxLayout()
        self.choose_btn = QPushButton("Choose File(s)...")
        self.choose_btn.clicked.connect(self.openFileDialog)
        file_hbox.addWidget(self.choose_btn)
        layout.addLayout(file_hbox)

        # Analysis operation buttons
        ops_group = QGroupBox("Analysis Operations")
        ops_layout = QVBoxLayout()

        ops_row1 = QHBoxLayout()
        self.dopesheet_btn = QPushButton("Export Dopesheet")
        self.dopesheet_btn.clicked.connect(lambda: self.runOperation(["dopesheet"]))
        self.dopesheet_btn.setEnabled(False)
        ops_row1.addWidget(self.dopesheet_btn)

        self.joints_btn = QPushButton("Analyze Joints")
        self.joints_btn.clicked.connect(lambda: self.runOperation(["joints"]))
        self.joints_btn.setEnabled(False)
        ops_row1.addWidget(self.joints_btn)
        ops_layout.addLayout(ops_row1)

        ops_row2 = QHBoxLayout()
        self.chains_btn = QPushButton("Analyze Chains")
        self.chains_btn.clicked.connect(lambda: self.runOperation(["chains"]))
        self.chains_btn.setEnabled(False)
        ops_row2.addWidget(self.chains_btn)

        self.gait_btn = QPushButton("Analyze Gait")
        self.gait_btn.clicked.connect(lambda: self.runOperation(["gait"]))
        self.gait_btn.setEnabled(False)
        ops_row2.addWidget(self.gait_btn)
        ops_layout.addLayout(ops_row2)

        ops_row3 = QHBoxLayout()
        self.run_all_btn = QPushButton("Run All Analyses")
        self.run_all_btn.clicked.connect(lambda: self.runOperation(["dopesheet", "joints", "chains", "gait"]))
        self.run_all_btn.setEnabled(False)
        ops_row3.addWidget(self.run_all_btn)

        self.viz_btn = QPushButton("🎬 Visualize 3D")
        self.viz_btn.clicked.connect(self.launch_visualizer)
        self.viz_btn.setEnabled(False)
        ops_row3.addWidget(self.viz_btn)
        ops_layout.addLayout(ops_row3)

        ops_row4 = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancelAnalysis)
        self.cancel_btn.setEnabled(False)
        ops_row4.addWidget(self.cancel_btn)
        ops_layout.addLayout(ops_row4)

        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)

        self.setLayout(layout)

    def openFileDialog(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open FBX Files", "", "FBX Files (*.fbx)")
        if fnames:
            self.fbx_files = fnames
            self.label.setText(f"Selected {len(fnames)} file(s): {', '.join(os.path.basename(f) for f in fnames)}")
            self.enableOperationButtons(True)
            self.viz_btn.setEnabled(True)  # Enable viz immediately
            self.result.setText("Ready to analyze or visualize.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith('.fbx')]
        if files:
            self.fbx_files = files
            self.label.setText(f"Dropped {len(files)} file(s): {', '.join(os.path.basename(f) for f in files)}")
            self.enableOperationButtons(True)
            self.viz_btn.setEnabled(True)  # Enable viz immediately
            self.result.setText("Ready to analyze or visualize.")

    def runOperation(self, operations):
        if not self.fbx_files:
            self.result.setText("No FBX files selected!")
            return

        self.enableOperationButtons(False)
        self.choose_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.viz_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.result.clear()

        fbx_file = self.fbx_files[0]
        self.worker = AnalysisWorker(fbx_file, operations)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.onAnalysisComplete)
        self.worker.error.connect(self.onAnalysisError)
        self.worker.start()

    def cancelAnalysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.result.append("\n⚠ Analysis cancelled by user.")
            self.resetUI()

    def updateProgress(self, message):
        self.result.append(message)

    def onAnalysisComplete(self, model, scene):
        self.current_scene = scene
        self.resetUI()
        self.result.append("\n" + "="*50)
        self.result.append("✓ Analysis Complete!")
        self.result.append("="*50)
        if model:
            self.result.append(f"  Gait Type: {model.get_gait_type()}")
            self.result.append(f"  Stride Count: {model.get_stride_count()}")
            base_name = os.path.splitext(os.path.basename(self.fbx_files[0]))[0]
            self.result.append(f"  Results saved to: output/{base_name}/")
        self.result.append("="*50)
        
        if self.current_scene:
            self.viz_btn.setEnabled(True)

    def onAnalysisError(self, error_msg):
        self.resetUI()
        self.result.append(f"\n✗ Critical Error: {error_msg}")

    def launch_visualizer(self):
        """Launch 3D viewer, loading scene if necessary."""
        try:
            # If we don't have a scene yet, load it
            if not self.current_scene and self.fbx_files:
                self.result.append("\n⏳ Loading FBX for visualization...")
                from fbx_tool.analysis.fbx_loader import load_fbx
                self.current_scene = load_fbx(self.fbx_files[0])
                self.result.append("✓ FBX loaded successfully")
            
            if self.current_scene:
                self.result.append("🎬 Launching 3D viewer...")
                # Store viewer reference to prevent garbage collection
                self.viewer_window = launch_skeleton_viewer(self.current_scene)
                self.result.append("✓ Viewer window opened")
            else:
                self.result.append("\n⚠ No FBX file selected. Please select a file first.")
        except Exception as e:
            import traceback
            self.result.append(f"\n✗ Visualization error: {str(e)}")
            self.result.append(f"Traceback:\n{traceback.format_exc()}")

    def enableOperationButtons(self, enabled):
        self.dopesheet_btn.setEnabled(enabled)
        self.joints_btn.setEnabled(enabled)
        self.chains_btn.setEnabled(enabled)
        self.gait_btn.setEnabled(enabled)
        self.run_all_btn.setEnabled(enabled)

    def resetUI(self):
        self.progress_bar.setVisible(False)
        self.enableOperationButtons(True)
        self.choose_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main():
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    window = FBXAnalyzerApp()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())