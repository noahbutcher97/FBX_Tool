"""
FBX Tool - Main CLI Entry Point
Runs comprehensive biomechanical analysis on FBX animation files.
Supports single and batch file processing with robust error handling.

Usage:
    python examples/run_analysis.py <fbx_file1> [fbx_file2 ...]

Example:
    python examples/run_analysis.py "assets/Test/FBX/Female Walk.fbx"
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fbx_tool.analysis.chain_analysis import analyze_chains
from fbx_tool.analysis.constraint_violation_detection import analyze_constraint_violations
from fbx_tool.analysis.dopesheet_export import export_dopesheet
from fbx_tool.analysis.fbx_loader import get_scene_metadata
from fbx_tool.analysis.foot_contact_analysis import analyze_foot_contacts
from fbx_tool.analysis.gait_analysis import analyze_gait
from fbx_tool.analysis.gait_summary import GaitSummaryAnalysis
from fbx_tool.analysis.joint_analysis import analyze_joints
from fbx_tool.analysis.pose_validity_analysis import analyze_pose_validity
from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
from fbx_tool.analysis.scene_manager import get_scene_manager
from fbx_tool.analysis.utils import ensure_output_dir
from fbx_tool.analysis.velocity_analysis import analyze_velocity


def run_analysis(fbx_file):
    """
    Run full animation analysis pipeline on a single FBX file.

    Uses modern scene manager for proper resource handling and memory management.
    Continues execution even if individual analysis steps fail, logging errors.

    Args:
        fbx_file (str): Path to FBX file to analyze

    Returns:
        dict: Analysis results from all modules (or None if critical failure)
    """
    # Create file-specific output directory
    base_name = Path(fbx_file).stem
    output_dir = f"output/{base_name}/"
    ensure_output_dir(output_dir)

    # Error tracking
    errors = []
    results = {}

    print(f"\n{'='*70}")
    print(f"FBX Tool - Comprehensive Analysis")
    print(f"File: {os.path.basename(fbx_file)}")
    print(f"{'='*70}")

    start_time = time.time()

    # Use scene manager for proper resource handling (mandatory as of 2025-10-17)
    scene_manager = get_scene_manager()

    try:
        with scene_manager.get_scene(fbx_file) as scene_ref:
            scene = scene_ref.scene

            # STEP 1: Extract Scene Metadata
            print("\n[1/10] Extracting scene metadata...")
            try:
                metadata = get_scene_metadata(scene)
                print(f"  Duration: {metadata.get('duration', 0):.2f}s")
                print(f"  Frame Rate: {metadata.get('frame_rate', 0):.2f} FPS")
                print(f"  Frame Count: {metadata.get('frame_count', 0)}")
                print(f"  Bone Count: {metadata.get('bone_count', 0)}")
                results["metadata"] = metadata
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Metadata Extraction", str(e), traceback.format_exc()))

            # STEP 2: Export Dopesheet
            print("\n[2/10] Exporting dopesheet...")
            try:
                dopesheet_path = os.path.join(output_dir, "dopesheet.csv")
                export_dopesheet(scene, dopesheet_path)
                print(f"  SUCCESS")
                results["dopesheet_path"] = dopesheet_path
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Dopesheet Export", str(e), traceback.format_exc()))

            # STEP 3: Joint Analysis
            print("\n[3/10] Analyzing joints...")
            try:
                joint_conf = analyze_joints(scene, output_dir=output_dir)
                print(f"  SUCCESS: {len(joint_conf)} joints analyzed")
                results["joints"] = joint_conf
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Joint Analysis", str(e), traceback.format_exc()))

            # STEP 4: Chain Analysis
            print("\n[4/10] Analyzing IK chains...")
            try:
                chain_conf = analyze_chains(scene, output_dir=output_dir)
                print(f"  SUCCESS: {len(chain_conf)} chains detected")
                results["chains"] = chain_conf
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Chain Analysis", str(e), traceback.format_exc()))

            # STEP 5: Velocity Analysis
            print("\n[5/10] Analyzing velocity patterns...")
            try:
                velocity_data = analyze_velocity(scene, output_dir=output_dir)
                print(f"  SUCCESS: {velocity_data.get('total_bones', 0)} bones analyzed")
                results["velocity"] = velocity_data
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Velocity Analysis", str(e), traceback.format_exc()))

            # STEP 6: Foot Contact Analysis
            print("\n[6/10] Detecting foot contacts...")
            try:
                foot_contacts = analyze_foot_contacts(scene, output_dir=output_dir)
                print(f"  SUCCESS: {foot_contacts.get('total_contacts', 0)} contacts detected")
                results["foot_contacts"] = foot_contacts
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Foot Contact Analysis", str(e), traceback.format_exc()))

            # STEP 7: Root Motion Analysis
            print("\n[7/10] Analyzing root motion...")
            try:
                root_motion = analyze_root_motion(scene, output_dir=output_dir)
                print(f"  SUCCESS: Total distance = {root_motion.get('total_distance', 0):.2f} units")
                results["root_motion"] = root_motion
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Root Motion Analysis", str(e), traceback.format_exc()))

            # STEP 8: Gait Analysis
            print("\n[8/10] Analyzing gait patterns...")
            try:
                stride_segments, gait_summary = analyze_gait(scene, output_dir=output_dir)
                print(f"  SUCCESS: {len(stride_segments)} stride segments detected")
                results["gait"] = {"segments": stride_segments, "summary": gait_summary}
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Gait Analysis", str(e), traceback.format_exc()))

            # STEP 9: Pose Validity Analysis
            print("\n[9/10] Validating pose integrity...")
            try:
                pose_validity = analyze_pose_validity(scene, output_dir=output_dir)
                score = pose_validity.get("overall_validity_score", 0)
                print(f"  SUCCESS: Validity score = {score:.2f}")
                results["pose_validity"] = pose_validity
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Pose Validity Analysis", str(e), traceback.format_exc()))

            # STEP 10: Constraint Violation Detection
            print("\n[10/10] Detecting constraint violations...")
            try:
                constraints = analyze_constraint_violations(scene, output_dir=output_dir)
                score = constraints.get("overall_constraint_score", 0)
                chains = constraints.get("total_chains", 0)
                violations = (
                    constraints.get("ik_violations", 0)
                    + constraints.get("hierarchy_violations", 0)
                    + constraints.get("curve_discontinuities", 0)
                )
                print(f"  SUCCESS: Constraint score = {score:.2f}")
                print(f"  Chains analyzed: {chains}, Total violations: {violations}")
                results["constraints"] = constraints
            except Exception as e:
                print(f"  FAILED: {str(e)}")
                errors.append(("Constraint Violation Detection", str(e), traceback.format_exc()))

        # Scene automatically cleaned up when exiting context manager

        # Print Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Analysis Complete ({elapsed:.2f}s)")
        print(f"{'='*70}")
        print(f"Modules completed: {len(results)}/10")
        print(f"Output directory: {output_dir}")

        # Print error summary if any failures occurred
        if errors:
            print(f"\nWARNING: {len(errors)} module(s) failed:")
            for step, msg, _ in errors:
                print(f"  - {step}: {msg}")

            # Write detailed error log
            try:
                error_log_path = os.path.join(output_dir, "errors.log")
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"FBX Tool Error Log - {fbx_file}\n")
                    f.write(f"{'='*70}\n\n")
                    for step, msg, trace in errors:
                        f.write(f"[{step}]\n")
                        f.write(f"Error: {msg}\n")
                        f.write(f"Traceback:\n{trace}\n")
                        f.write(f"{'-'*70}\n\n")
                print(f"\nDetailed error log saved to: {error_log_path}")
            except Exception:
                pass  # Don't crash if we can't write error log

        return results

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"\nCRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())
        return None


def main():
    """Main entry point for CLI."""
    if len(sys.argv) < 2:
        print("FBX Tool - Comprehensive Animation Analysis\n")
        print("Usage: python examples/run_analysis.py <fbx_file1> [fbx_file2 ...]\n")
        print("Example:")
        print('  python examples/run_analysis.py "assets/Test/FBX/Female Walk.fbx"')
        sys.exit(1)

    files = sys.argv[1:]

    # Validate files
    valid_files = []
    for fbx_file in files:
        if not os.path.exists(fbx_file):
            print(f"ERROR: File not found: {fbx_file}")
            continue
        if not fbx_file.lower().endswith(".fbx"):
            print(f"ERROR: Not an FBX file: {fbx_file}")
            continue
        valid_files.append(fbx_file)

    if not valid_files:
        print("ERROR: No valid FBX files provided.")
        sys.exit(1)

    print(f"\nFBX Tool - Batch Analysis")
    print(f"{'='*70}")
    print(f"Processing {len(valid_files)} file(s)...\n")

    results = []
    for fbx_file in valid_files:
        analysis_results = run_analysis(fbx_file)
        results.append((fbx_file, analysis_results))

    # Print batch summary
    print(f"\n\n{'='*70}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"{'='*70}")

    success_count = sum(1 for _, res in results if res is not None)
    print(f"Files processed: {len(valid_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(valid_files) - success_count}")

    if success_count < len(valid_files):
        print("\nFailed files:")
        for fbx_file, res in results:
            if res is None:
                print(f"  - {os.path.basename(fbx_file)}")


if __name__ == "__main__":
    main()
