# Exploratory Tests

This folder contains exploratory test scripts written during development to understand FBX SDK behavior and test specific features.

## Scripts

**`test_animation_extraction.py`**
- Purpose: Test animation data extraction from FBX
- Usage: `python test_animation_extraction.py <fbx_file>`
- Explores: Time span, frame rate, animation data

**`test_animation_variance.py`**
- Purpose: Analyze animation variance and data quality
- Explores: Motion patterns, variance detection

**`test_chain_detection.py`**
- Purpose: Test dynamic chain detection algorithm
- Usage: Hardcoded file path
- Explores: Bone hierarchy, chain detection logic

**`test_fbx_coordinate_system.py`**
- Purpose: Test coordinate system detection
- Explores: Forward axis, coordinate transformations

**`test_stack_1.py`**
- Purpose: Test animation stack selection
- Explores: Multiple animation stacks, stack switching

## Difference from Debug Scripts

**Debug scripts (`tests/debug/`):** Simple inspection/debugging tools
**Exploratory tests:** More complex experiments with specific features

## Migration Path

These scripts may eventually be:
1. **Converted to unit tests** - If the feature is stable and needs regression testing
2. **Archived** - If the exploration is complete and documented elsewhere
3. **Kept** - If they're useful for understanding edge cases

## Notes

- These are **not pytest tests** - they are exploratory experiments
- They may have hardcoded file paths
- They were written to understand specific FBX SDK behaviors
- They may be outdated as the codebase evolved

---

**Last Updated:** 2025-10-18
