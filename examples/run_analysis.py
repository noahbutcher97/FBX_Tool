"""
FBX Tool - Main CLI Entry Point
Runs comprehensive biomechanical analysis on FBX animation files.
Supports single and batch file processing with robust error handling.
"""

import sys
import os
import time
import traceback
from analysis_modules.fbx_loader import load_fbx, get_scene_metadata
from analysis_modules.dopesheet_export import export_dopesheet
from analysis_modules.gait_analysis import analyze_gait
from analysis_modules.chain_analysis import analyze_chains
from analysis_modules.joint_analysis import analyze_joints
from analysis_modules.gait_summary import GaitSummaryAnalysis
from analysis_modules.utils import ensure_output_dir


def run_analysis(fbx_file):
    """
    Run full animation analysis pipeline on a single FBX file.

    Continues execution even if individual analysis steps fail,
    logging errors for debugging.

    Args:
        fbx_file (str): Path to FBX file to analyze

    Returns:
        GaitSummaryAnalysis: Unified analysis model (or None if critical failure)
    """
    # Create file-specific output directory
    base_name = os.path.splitext(os.path.basename(fbx_file))[0]
    output_dir = f"output/{base_name}/"
    ensure_output_dir(output_dir)

    # Error tracking
    errors = []

    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(fbx_file)}")
    print(f"{'='*60}")

    start_time = time.time()

    # Initialize result variables
    scene = None
    metadata = {}
    dopesheet_path = None
    joint_conf = {}
    chain_conf = {}
    stride_segments = []
    gait_summary = {}

    try:
        # STEP 1: Load FBX Scene (CRITICAL - must succeed)
        print("Loading FBX scene...")
        try:
            scene = load_fbx(fbx_file)
            metadata = get_scene_metadata(scene)
            print(f"  Duration: {metadata.get('duration', 0):.2f}s")
            print(f"  Frame Rate: {metadata.get('frame_rate', 0):.2f} FPS")
            print(f"  Bone Count: {metadata.get('bone_count', 0)}")
        except Exception as e:
            error_msg = f"✗ CRITICAL: Failed to load FBX scene: {str(e)}"
            print(error_msg)
            errors.append(("FBX Loading", str(e), traceback.format_exc()))
            return None  # Cannot continue without scene

        # STEP 2: Export Dopesheet (NON-CRITICAL)
        print("\nExporting dopesheet...")
        try:
            dopesheet_path = os.path.join(output_dir, "dopesheet.csv")
            export_dopesheet(scene, dopesheet_path)
            print(f"  ✓ Dopesheet exported: {dopesheet_path}")
        except Exception as e:
            error_msg = f"✗ Dopesheet export failed: {str(e)}"
            print(error_msg)
            errors.append(("Dopesheet Export", str(e), traceback.format_exc()))

        # STEP 3: Joint Analysis (NON-CRITICAL)
        print("\nAnalyzing joints...")
        try:
            joint_conf = analyze_joints(scene, output_dir=output_dir)
            print(f"  ✓ {len(joint_conf)} joints analyzed.")
        except Exception as e:
            error_msg = f"✗ Joint analysis failed: {str(e)}"
            print(error_msg)
            errors.append(("Joint Analysis", str(e), traceback.format_exc()))

        # STEP 4: Chain Analysis (NON-CRITICAL)
        print("\nAnalyzing chains...")
        try:
            chain_conf = analyze_chains(scene, output_dir=output_dir)
            print(f"  ✓ {len(chain_conf)} chains analyzed.")
        except Exception as e:
            error_msg = f"✗ Chain analysis failed: {str(e)}"
            print(error_msg)
            errors.append(("Chain Analysis", str(e), traceback.format_exc()))

        # STEP 5: Gait Analysis (NON-CRITICAL)
        print("\nAnalyzing gait patterns...")
        try:
            stride_segments, gait_summary = analyze_gait(scene, output_dir=output_dir)
            print(f"  ✓ {len(stride_segments)} stride segments detected.")
        except Exception as e:
            error_msg = f"✗ Gait analysis failed: {str(e)}"
            print(error_msg)
            errors.append(("Gait Analysis", str(e), traceback.format_exc()))

        # STEP 6: Create Unified Analysis Model
        print("\nGenerating analysis summary...")
        try:
            model = GaitSummaryAnalysis(
                fbx_path=fbx_file,
                dopesheet_path=dopesheet_path,
                gait_summary=gait_summary,
                chain_conf=chain_conf,
                joint_conf=joint_conf,
                stride_segments=stride_segments
            )

            json_path = os.path.join(output_dir, "analysis_summary.json")
            model.to_json(json_path)
            print(f"  ✓ Summary saved: {json_path}")

        except Exception as e:
            error_msg = f"✗ Failed to create analysis model: {str(e)}"
            print(error_msg)
            errors.append(("Model Creation", str(e), traceback.format_exc()))
            model = None

        # STEP 7: Print Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Analysis Complete ({elapsed:.2f}s)")
        print(f"{'='*60}")

        if model:
            print(f"  Gait Type: {model.get_gait_type()}")
            print(f"  Stride Count: {model.get_stride_count()}")
        print(f"  Results saved to: {output_dir}")

        # Print error summary if any failures occurred
        if errors:
            print(f"\n⚠ {len(errors)} analysis step(s) failed:")
            for step, msg, _ in errors:
                print(f"    - {step}: {msg}")
            print(f"\nDetailed error log saved to: {output_dir}errors.log")

            # Write detailed error log
            try:
                error_log_path = os.path.join(output_dir, "errors.log")
                with open(error_log_path, 'w') as f:
                    f.write(f"FBX Tool Error Log - {fbx_file}\n")
                    f.write(f"{'='*60}\n\n")
                    for step, msg, trace in errors:
                        f.write(f"[{step}]\n")
                        f.write(f"Error: {msg}\n")
                        f.write(f"Traceback:\n{trace}\n")
                        f.write(f"{'-'*60}\n\n")
            except:
                pass  # Don't crash if we can't write error log

        return model

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"\n✗ Unexpected error during analysis: {str(e)}")
        print(traceback.format_exc())
        return None


def main():
    """Main entry point for CLI."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <fbx_file1> [fbx_file2 ...]")
        sys.exit(1)

    files = sys.argv[1:]

    # Validate files
    valid_files = []
    for fbx_file in files:
        if not os.path.exists(fbx_file):
            print(f"✗ File not found: {fbx_file}")
            continue
        if not fbx_file.lower().endswith('.fbx'):
            print(f"✗ Not an FBX file: {fbx_file}")
            continue
        valid_files.append(fbx_file)

    if not valid_files:
        print("✗ No valid FBX files provided.")
        sys.exit(1)

    print(f"\nFBX Tool - Batch Analysis")
    print(f"{'='*60}")
    print(f"Processing {len(valid_files)} file(s)...")

    results = []
    for fbx_file in valid_files:
        model = run_analysis(fbx_file)
        results.append((fbx_file, model))

    # Print batch summary
    print(f"\n\n{'='*60}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")

    success_count = sum(1 for _, model in results if model is not None)
    print(f"Files processed: {len(valid_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(valid_files) - success_count}")

    if success_count < len(valid_files):
        print("\nFailed files:")
        for fbx_file, model in results:
            if model is None:
                print(f"  - {os.path.basename(fbx_file)}")


if __name__ == "__main__":
    main()