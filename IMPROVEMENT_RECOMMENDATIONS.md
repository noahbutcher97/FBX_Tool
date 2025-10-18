# Code Improvement Recommendations
## Edge Case Handling & Data Collection Enhancements

Based on TDD review and analysis of existing modules, here are recommended improvements:

---

## 1. **gait_analysis.py** - Critical Issues

### Issue A: Silent Failures
**Location**: Lines 87-88, 64-65
```python
if len(contacts_idx) < 2:
    continue  # ❌ Silently skips - user doesn't know why no gait detected
```

**Improvement**:
```python
if len(contacts_idx) < 2:
    print(f"⚠ Warning: {cname} has insufficient foot contacts ({len(contacts_idx)} found, need 2+). Skipping gait analysis.")
    continue

if cname not in chains:
    print(f"⚠ Warning: Chain '{cname}' not found in skeleton. Available chains: {list(chains.keys())}")
    continue
```

### Issue B: Array Bounds Not Validated
**Location**: Lines 92-100
```python
segment_vel = vel[last:c]  # ❌ No validation that c > last
conf = np.mean(np.pi / (1 + np.abs(segment_vel)))  # ❌ Could divide by empty array
```

**Improvement**:
```python
if c <= last:
    print(f"⚠ Warning: Invalid stride segment {cname} [{last}:{c}] - skipping")
    continue

segment_vel = vel[last:c]
if len(segment_vel) == 0:
    print(f"⚠ Warning: Empty velocity segment for {cname} - skipping")
    continue

conf = np.mean(np.pi / (1 + np.abs(segment_vel)))
```

### Issue C: Unsafe Dictionary Creation
**Location**: Lines 152-156
```python
gait_summary = {
    "cycle_rate": float(summary_info[0][1] / summary_info[0][2]),  # ❌ Could be empty or division by zero
    "confidence": float(np.mean([s[5] for s in stride_segments])) if stride_segments else 0.0,
    "gait_type": summary_info[0][6] if summary_info else "Unknown"
}
```

**Improvement**:
```python
gait_summary = {
    "cycle_rate": float(summary_info[0][1] / summary_info[0][2]) if (summary_info and len(summary_info[0]) > 2 and summary_info[0][2] > 0) else 0.0,
    "confidence": float(np.mean([s[5] for s in stride_segments])) if stride_segments else 0.0,
    "gait_type": summary_info[0][6] if (summary_info and len(summary_info[0]) > 6) else "Unknown",
    "total_strides": len(stride_segments),
    "chains_analyzed": len([c for c in ["LeftLeg", "RightLeg"] if c in chains])
}
```

---

## 2. **chain_analysis.py** - Robustness Issues

### Issue A: Complex List Comprehension Can Fail Silently
**Location**: Line 76
```python
segs = [(b, hierarchy[b]) for b in chain if hierarchy.get(b) and (hierarchy[b], b) in joint_data]
# ❌ Silently filters out bones - user doesn't know why chain is incomplete
```

**Improvement**:
```python
segs = []
missing_bones = []
for b in chain:
    parent = hierarchy.get(b)
    if not parent:
        missing_bones.append(f"{b} (no parent)")
        continue
    if (parent, b) not in joint_data:
        missing_bones.append(f"{b} (no joint data)")
        continue
    segs.append((b, parent))

if missing_bones:
    print(f"⚠ Chain {cname}: Excluded bones: {', '.join(missing_bones)}")

if len(segs) < 2:
    print(f"⚠ Chain {cname}: Insufficient valid segments ({len(segs)}/min 2) - skipping")
    continue
```

### Issue B: NaN Handling in Correlations
**Location**: Lines 109-111
```python
corrs.append(np.corrcoef(w1, w2)[0, 1])
cross_t = np.nanmean(corrs) if len(corrs) > 0 else 0
# ❌ What if ALL correlations are NaN? nanmean returns NaN!
```

**Improvement**:
```python
# Filter out NaN/inf correlations before computing mean
valid_corrs = [c for c in corrs if np.isfinite(c)]
if len(valid_corrs) == 0:
    cross_t = 0.0
    print(f"⚠ Chain {cname}: No valid temporal correlations - using 0.0")
else:
    cross_t = np.mean(valid_corrs)
```

---

## 3. **velocity_analysis.py** - Statistical Robustness

### Issue A: Division by Zero in detect_spikes()
**Location**: Lines 138-143
```python
def detect_spikes(values, threshold_multiplier=3.0):
    mean = np.mean(values)
    std = np.std(values)  # ❌ Could be 0 for constant values
    threshold = mean + (threshold_multiplier * std)
    spike_indices = np.where(values > threshold)[0]
    return spike_indices
```

**Improvement**:
```python
def detect_spikes(values, threshold_multiplier=3.0, min_std=1e-6):
    """
    Detect spikes using statistical outlier detection.

    Args:
        values: 1D array of scalar values
        threshold_multiplier: Number of standard deviations for outlier
        min_std: Minimum std to avoid false positives on constant data

    Returns:
        Array of frame indices where spikes occur
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    mean = np.mean(values)
    std = np.std(values)

    # Handle constant or near-constant values
    if std < min_std:
        return np.array([], dtype=int)  # No spikes in constant data

    threshold = mean + (threshold_multiplier * std)
    spike_indices = np.where(values > threshold)[0]
    return spike_indices
```

### Issue B: Missing Import Validation
**Location**: Line 650
```python
from fbx_tool.analysis.chain_analysis import detect_chains  # ❌ This function doesn't exist!
```

**Fix**: Use the correct function from utils:
```python
from fbx_tool.analysis.utils import detect_chains_from_hierarchy, build_bone_hierarchy
```

### Issue C: Division by Zero in Severity Calculations
**Location**: Lines 470, 482, 493
```python
'severity': acceleration_mag[frame_idx] / mean_acceleration if mean_acceleration > 0 else 0
'severity': jerk_mag[frame_idx] / mean_jerk if mean_jerk > 0 else 0  # ✅ Already has guard
```

**Status**: ✅ Already properly handled

---

## 4. **utils.py** - Enhanced Data Collection

### Issue A: Zero-Duration Animation Stacks
**Location**: Lines 103, 113-115
```python
duration = time_span.GetStop().GetSecondDouble() - time_span.GetStart().GetSecondDouble()

if duration > max_duration:  # ❌ What if ALL durations are 0?
    max_duration = duration
    selected_stack = stack
    selected_index = i
```

**Improvement**:
```python
duration = time_span.GetStop().GetSecondDouble() - time_span.GetStart().GetSecondDouble()

# Prefer mixamo stack
if "mixamo" in stack_name.lower() and duration > 0:
    selected_stack = stack
    selected_index = i
    break

# Otherwise use the longest NON-ZERO duration stack
if duration > max_duration and duration > 0:
    max_duration = duration
    selected_stack = stack
    selected_index = i
```

### Issue B: Empty Chain Detection Results
**Location**: Lines 304-312
```python
if len(chain) >= min_chain_length:
    visited.update(chain)
    chain_name = _generate_chain_name(chain)
    chains[chain_name] = chain
# ❌ No feedback if NO chains meet minimum length
```

**Improvement**:
```python
if len(chain) >= min_chain_length:
    visited.update(chain)
    chain_name = _generate_chain_name(chain)
    chains[chain_name] = chain
else:
    # Log short chains for debugging
    if len(chain) > 1:
        print(f"ℹ Skipping short chain: {' -> '.join(chain)} (length {len(chain)} < min {min_chain_length})")

# After loop completes:
if not chains:
    print(f"⚠ Warning: No chains detected with minimum length {min_chain_length}. Consider lowering min_chain_length.")
```

---

## 5. **General Improvements Across All Modules**

### A. Add Input Validation Decorators
```python
def validate_scene(func):
    """Decorator to validate FBX scene before analysis."""
    def wrapper(scene, *args, **kwargs):
        if scene is None:
            raise ValueError("Scene cannot be None")

        root = scene.GetRootNode()
        if not root or root.GetChildCount() == 0:
            raise ValueError("Scene has no bones/nodes")

        return func(scene, *args, **kwargs)
    return wrapper

@validate_scene
def analyze_gait(scene, output_dir="output/"):
    # ... existing code
```

### B. Enhanced Progress Reporting
```python
def analyze_chains(scene, output_dir="output/"):
    """... existing docstring ..."""

    print(f"Starting chain analysis...")
    hierarchy = build_bone_hierarchy(scene)
    print(f"  Found {len(hierarchy)} bones in hierarchy")

    chains = detect_chains_from_hierarchy(hierarchy, min_chain_length=3)
    print(f"  Detected {len(chains)} chains: {list(chains.keys())}")

    # ... rest of analysis

    print(f"✓ Chain analysis complete: {len(chain_results)} chains analyzed")
```

### C. Comprehensive Return Values
Instead of returning minimal data, return detailed diagnostics:
```python
return {
    'chain_conf': chain_conf,
    'total_chains': len(chains),
    'analyzed_chains': len(chain_results),
    'skipped_chains': len(chains) - len(chain_results),
    'total_bones': len(hierarchy),
    'warnings': warnings_list  # Collect warnings during analysis
}
```

---

## Priority Recommendations

### HIGH PRIORITY (Crash/Data Loss Prevention):
1. ✅ Add division-by-zero guards in `detect_spikes()`
2. ✅ Validate array bounds in `gait_analysis.py` stride loops
3. ✅ Fix missing import in `velocity_analysis.py`
4. ✅ Handle NaN correlations in `chain_analysis.py`

### MEDIUM PRIORITY (User Experience):
5. ✅ Add warning messages for silently skipped chains/bones
6. ✅ Validate dictionary creation with empty data
7. ✅ Add progress reporting to long-running analysis

### LOW PRIORITY (Data Quality):
8. ✅ Enhanced return values with diagnostics
9. ✅ Input validation decorators
10. ✅ Better chain detection feedback

---

## Testing Recommendations

For modules requiring FBX scenes, create **fixture-based integration tests**:

```python
@pytest.fixture
def minimal_fbx_scene():
    """Create minimal valid FBX scene for testing."""
    # Use FBX SDK to programmatically create simple test scene
    # OR load a known-good minimal FBX file
    pass

def test_gait_analysis_with_no_leg_chains(minimal_fbx_scene):
    """Test gait analysis gracefully handles missing leg chains."""
    # Remove leg chains from scene
    # Verify no crash and appropriate warning message
    pass
```

---

## Implementation Strategy

1. Start with **HIGH PRIORITY** fixes (safety/correctness)
2. Add **validation** at module entry points
3. Enhance **error messages** for better debugging
4. Add **integration tests** with minimal FBX fixtures
5. Gradually add **data collection enhancements**

Would you like me to implement these improvements?
