# TDD Examples and Best Practices

## TDD Workflow

**MANDATORY:** All new features and incomplete modules MUST follow strict TDD.

### TDD Process
1. **Write tests FIRST** - Define expected behavior before implementation
2. **Run tests** - Watch them fail (red)
3. **Implement code to pass tests** - Not just minimal code, but robust implementation
4. **Refactor** - Improve code while keeping tests green
5. **Add edge cases** - Expand test coverage iteratively

## Writing Robust Tests

**CRITICAL:** Tests must be comprehensive enough to demand robust implementations. Don't write trivial tests that pass with placeholder code.

### ❌ BAD Test (Too Minimal)

```python
def test_detect_contacts():
    """Test detects contacts."""
    contacts = detect_foot_contacts([], [])
    assert contacts is not None  # Passes with "return []"
```

**Problem:** This test passes with trivial placeholder code. It doesn't verify actual behavior.

### ✅ GOOD Tests (Demand Robust Implementation)

#### Example 1: Normal Case with Multiple Assertions

```python
def test_detect_contacts_with_clear_ground_strike():
    """Should detect contact when foot velocity drops to zero near ground."""
    # Arrange - Foot descending, hits ground, stays stationary
    positions = np.array([
        [0, 20, 0],  # Frame 0: High
        [0, 10, 0],  # Frame 1: Descending
        [0, 2, 0],   # Frame 2: Near ground
        [0, 0, 0],   # Frame 3: On ground (CONTACT START)
        [0, 0, 0],   # Frame 4: Stationary
        [0, 0, 0],   # Frame 5: Stationary (CONTACT END)
        [0, 5, 0],   # Frame 6: Lifting off
    ])
    velocities = np.diff(positions[:, 1])  # Vertical velocity

    # Act
    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    # Assert - Multiple conditions to verify correctness
    assert len(contacts) == 1, "Should detect exactly one contact period"
    assert contacts[0]['start_frame'] == 3, "Contact should start at frame 3"
    assert contacts[0]['end_frame'] == 5, "Contact should end at frame 5"
    assert contacts[0]['duration'] > 0, "Contact must have positive duration"
    assert 0.0 <= contacts[0]['confidence'] <= 1.0, "Confidence must be in [0,1]"
```

**Why this is good:**
- Uses realistic test data (descending → contact → lift off)
- Tests multiple aspects of the output
- Includes clear comments explaining expected behavior
- Verifies frame indices, duration, and confidence bounds

#### Example 2: Testing Adaptive Behavior

```python
def test_detect_contacts_with_root_motion():
    """Should work with root motion (non-zero ground height)."""
    # Character walking with Y offset
    positions = create_walking_pattern(ground_height=50.0, num_strides=3)
    velocities = np.gradient(positions[:, 1])

    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    assert len(contacts) >= 3, "Should detect at least 3 contacts for 3 strides"
    # Verify adaptive thresholding worked (not hardcoded ground=0)
    for contact in contacts:
        assert contact['ground_height'] > 40, "Should detect elevated ground"
```

**Why this is good:**
- Tests that code adapts to different scenarios (root motion)
- Verifies no hardcoded assumptions (ground height = 0)
- Uses helper functions for complex test data

#### Example 3: Graceful Degradation

```python
def test_detect_contacts_empty_data_graceful_degradation():
    """Should handle empty data gracefully without crashing."""
    contacts = detect_foot_contacts(np.array([]), np.array([]), frame_rate=30.0)

    assert contacts == [], "Empty input should return empty list"
    # Should not raise exception
```

**Why this is good:**
- Tests edge case (empty input)
- Verifies graceful handling without crashes
- Documents expected behavior for invalid input

#### Example 4: Single-Frame Edge Case

```python
def test_detect_contacts_single_frame_edge_case():
    """Should handle single-frame animation without crashing."""
    positions = np.array([[0, 0, 0]])
    velocities = np.array([])

    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    assert contacts == [], "Single frame should return no contacts"
```

**Why this is good:**
- Tests boundary condition (single frame)
- Prevents crashes on minimal input
- Simple but important edge case

## Test Organization Patterns

### Test File Structure

```python
import pytest
import numpy as np
from fbx_tool.analysis.module_name import function_to_test

# Use fixtures from conftest.py
# test_normal_cases
def test_basic_functionality():
    """Tests the happy path with normal inputs."""
    pass

def test_with_variation_1():
    """Tests normal case with specific variation."""
    pass

# test_edge_cases
def test_empty_input():
    """Should handle empty data gracefully."""
    pass

def test_single_element():
    """Should handle minimal valid input."""
    pass

def test_extreme_values():
    """Should handle very large or very small values."""
    pass

# test_adaptive_behavior
def test_adapts_to_different_scales():
    """Should work across different character scales."""
    pass

def test_adapts_to_different_framerates():
    """Should work with varying frame rates."""
    pass

# test_error_conditions
def test_invalid_input_types():
    """Should raise appropriate error for invalid types."""
    pass

def test_nan_values():
    """Should handle NaN values appropriately."""
    pass
```

### Using Fixtures from conftest.py

```python
def test_with_mock_scene(mock_scene):
    """Use shared mock scene fixture."""
    # mock_scene is automatically available
    result = analyze_scene(mock_scene)
    assert result is not None

def test_with_sample_data(sample_positions, sample_velocities):
    """Use shared sample data fixtures."""
    result = analyze_motion(sample_positions, sample_velocities)
    assert len(result) > 0

def test_with_temp_dir(temp_output_dir):
    """Use temporary directory for file output."""
    output_path = temp_output_dir / "result.csv"
    save_results(output_path)
    assert output_path.exists()
```

## Coverage Requirements

From `pytest.ini`:
- **Minimum:** 20% overall (enforced)
- **Target:** 80% for new modules
- **Current:** 24.31% overall
- **Reference standard:** gait_analysis.py (88% coverage, 22/22 tests passing)

**Study `tests/unit/test_gait_analysis.py` for comprehensive test patterns.**

## Common Test Patterns

### Pattern 1: Arrange-Act-Assert

```python
def test_something():
    """Test description."""
    # Arrange - Set up test data
    input_data = create_test_data()

    # Act - Call the function
    result = function_under_test(input_data)

    # Assert - Verify results
    assert result is not None
    assert result['key'] == expected_value
```

### Pattern 2: Parametrized Tests

```python
@pytest.mark.parametrize("frame_rate,expected_duration", [
    (30.0, 1.0),
    (60.0, 0.5),
    (24.0, 1.25),
])
def test_duration_calculation(frame_rate, expected_duration):
    """Should calculate duration correctly for different frame rates."""
    frames = 30
    duration = calculate_duration(frames, frame_rate)
    assert abs(duration - expected_duration) < 0.01
```

### Pattern 3: Testing Confidence Scores

```python
def test_returns_confidence_score():
    """All analysis functions should return confidence scores."""
    result = analyze_something(test_data)

    assert 'confidence' in result, "Must include confidence score"
    assert 0.0 <= result['confidence'] <= 1.0, "Confidence must be in [0,1]"
    assert isinstance(result['confidence'], float), "Confidence must be float"
```

### Pattern 4: Testing Adaptive Thresholds

```python
def test_adaptive_threshold_not_hardcoded():
    """Should derive threshold from data, not use hardcoded values."""
    # Test with small-scale data
    small_data = np.array([1, 2, 3, 4, 5])
    small_threshold = calculate_threshold(small_data)

    # Test with large-scale data
    large_data = np.array([100, 200, 300, 400, 500])
    large_threshold = calculate_threshold(large_data)

    # Thresholds should scale with data
    assert large_threshold > small_threshold * 50, "Should adapt to data scale"
```

## Common Pitfalls to Avoid

### ❌ Don't: Write Tests That Pass with Placeholder Code

```python
def test_bad():
    result = function()
    assert result is not None  # Passes with "return {}"
```

### ✅ Do: Write Tests That Demand Real Implementation

```python
def test_good():
    result = function(input_data)
    assert result['detected_contacts'] == 3
    assert result['total_duration'] > 0
    assert all(c['confidence'] > 0 for c in result['contacts'])
```

### ❌ Don't: Forget Edge Cases

```python
def test_only_normal_case():
    # Only tests happy path
    result = function([1, 2, 3, 4, 5])
    assert len(result) > 0
```

### ✅ Do: Test Edge Cases Explicitly

```python
def test_edge_cases():
    # Empty input
    assert function([]) == []

    # Single element
    assert function([1]) == []

    # All same values
    assert function([5, 5, 5, 5]) == []
```

### ❌ Don't: Test Implementation Details

```python
def test_bad():
    # Tests internal variable names
    obj = MyClass()
    assert obj._internal_cache is not None  # Brittle!
```

### ✅ Do: Test Public Behavior

```python
def test_good():
    # Tests observable behavior
    obj = MyClass()
    result = obj.analyze(data)
    assert result['status'] == 'success'
```

## Reference Examples

**Best reference:** `tests/unit/test_gait_analysis.py`
- 88% coverage
- 22/22 tests passing
- Comprehensive edge case coverage
- Good use of fixtures
- Clear test names and documentation
