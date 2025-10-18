# Debug Scripts

This folder contains ad-hoc debugging and inspection scripts for manual testing and exploration.

## Scripts

### Inspection Scripts

**`inspect_animation_layers.py`**
- Purpose: Inspect animation layers and curves in FBX file
- Usage: Hardcoded file path
- Output: Animation stack information

**`inspect_bones.py`**
- Purpose: Inspect bone names and hierarchy in FBX file
- Usage: `python inspect_bones.py <fbx_file>`
- Output: Bone hierarchy structure

### Test Scripts

**`test_motion_states.py`**
- Purpose: Quick test for motion state detection with debug logging
- Created: Session 2025-10-18
- Usage: Hardcoded file path
- Output: Motion state detection results

## Notes

- These are **not pytest tests** - they are manual debugging scripts
- They may have hardcoded file paths
- They are for development/debugging only
- Run them directly with `python tests/debug/script_name.py`

## When to Use

Use these scripts when:
- Debugging FBX SDK issues
- Inspecting animation file structure
- Testing new features in isolation
- Quick prototyping without full test infrastructure

## When to Create Proper Tests

If a debug script becomes important for regression testing:
1. Convert it to a proper pytest test
2. Move it to `tests/unit/` or `tests/integration/`
3. Use fixtures and mocks instead of hardcoded paths
4. Add to CI/CD pipeline

---

**Last Updated:** 2025-10-18
