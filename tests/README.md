# FBX Tool Tests

Comprehensive test suite for FBX animation analysis tool.

## Test Organization

```
tests/
├── README.md                  ← This file
├── conftest.py               ← Shared pytest fixtures
├── pytest.ini                ← Pytest configuration (in root)
│
├── unit/                     ← Unit tests (fast, isolated)
│   ├── test_chain_analysis.py
│   ├── test_constraint_violation_detection.py
│   ├── test_directional_change_detection.py
│   ├── test_fbx_memory_management.py
│   ├── test_foot_contact_analysis.py
│   ├── test_gait_analysis.py
│   ├── test_gait_summary.py
│   ├── test_gui_analysis_worker.py
│   ├── test_joint_analysis.py
│   ├── test_motion_classification.py
│   ├── test_motion_transition_detection.py
│   ├── test_pose_validity_analysis.py
│   ├── test_root_motion_analysis.py
│   ├── test_scene_manager.py
│   ├── test_temporal_segmentation.py
│   ├── test_utils.py
│   ├── test_utils_trajectory.py
│   └── test_velocity_analysis.py
│
├── integration/              ← Integration tests (slower, multi-component)
│   └── test_analysis_pipeline.py
│
├── debug/                    ← Ad-hoc debugging scripts
│   ├── README.md
│   ├── inspect_animation_layers.py
│   ├── inspect_bones.py
│   └── test_motion_states.py
│
└── exploratory/              ← Exploratory test scripts
    ├── README.md
    ├── test_animation_extraction.py
    ├── test_animation_variance.py
    ├── test_chain_detection.py
    ├── test_fbx_coordinate_system.py
    └── test_stack_1.py
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose:** Test individual functions and classes in isolation

**Characteristics:**
- Fast execution (< 1 second each)
- Mocked dependencies (no real FBX files)
- High coverage of edge cases
- Run on every commit (CI/CD)

**Example:**
```python
def test_detect_stride_segments_normal_gait():
    """Test stride detection with typical walking pattern."""
    # ... test implementation
```

**Run:**
```bash
pytest tests/unit/ -v
pytest tests/unit/test_gait_analysis.py -v
pytest tests/unit/test_gait_analysis.py::test_detect_stride_segments_normal_gait -v
```

### Integration Tests (`tests/integration/`)

**Purpose:** Test multiple components working together

**Characteristics:**
- Slower execution (1-10 seconds each)
- May use real FBX files or complex mocks
- Test complete workflows
- Run before releases

**Example:**
```python
def test_complete_analysis_pipeline():
    """Test full analysis from FBX load to final output."""
    # ... test implementation
```

**Run:**
```bash
pytest tests/integration/ -v
pytest tests/integration/test_analysis_pipeline.py -v
```

### Debug Scripts (`tests/debug/`)

**Purpose:** Manual debugging and inspection tools

**Characteristics:**
- Not pytest tests (run directly with `python`)
- May have hardcoded file paths
- For development/debugging only
- Not part of CI/CD

**Example:**
```python
# inspect_bones.py
python tests/debug/inspect_bones.py path/to/file.fbx
```

### Exploratory Tests (`tests/exploratory/`)

**Purpose:** Experiments and feature exploration

**Characteristics:**
- Not pytest tests (run directly with `python`)
- Written to understand FBX SDK behavior
- May be outdated or incomplete
- Can be converted to unit tests if useful

**Example:**
```python
# test_coordinate_system.py
python tests/exploratory/test_fbx_coordinate_system.py
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/ -v
```

### Integration Tests Only
```bash
pytest tests/integration/ -v
```

### Specific Test File
```bash
pytest tests/unit/test_gait_analysis.py -v
```

### Specific Test Function
```bash
pytest tests/unit/test_gait_analysis.py::test_detect_stride_segments_normal_gait -v
```

### With Coverage
```bash
pytest --cov=fbx_tool --cov-report=html
# Opens: htmlcov/index.html
```

### Parallel Execution
```bash
pytest -n auto  # Uses all CPU cores
pytest -n 4     # Uses 4 workers
pytest -n 0     # Disables parallelism (useful for debugging)
```

### Fast Tests Only
```bash
pytest -m "unit and not slow"
```

### Show Local Variables on Failure
```bash
pytest -l
```

### Re-run Failed Tests
```bash
pytest --lf  # Last failed
pytest --ff  # Failed first, then others
```

## Writing Tests

### Unit Test Template

```python
"""Test module description."""
import pytest
from fbx_tool.analysis.module import function_to_test


def test_function_normal_case():
    """Test function with typical input."""
    result = function_to_test(normal_input)
    assert result == expected_output


def test_function_edge_case():
    """Test function with edge case input."""
    result = function_to_test(edge_case_input)
    assert result == expected_output


def test_function_error_handling():
    """Test function handles errors correctly."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

### Using Fixtures

Fixtures are defined in `conftest.py` and available to all tests:

```python
def test_with_mock_scene(mock_fbx_scene):
    """Test using shared mock scene fixture."""
    result = analyze_scene(mock_fbx_scene)
    assert result is not None
```

### Mocking FBX SDK

See `docs/testing/MOCK_SETUP_PATTERNS.md` for FBX SDK mocking patterns.

## Test Coverage

**Minimum required:** 20%
**Current:** 22.36%

**High coverage modules:**
- Root motion analysis: 100%
- Motion transition detection: 72.13%
- Directional change detection: 65.62%
- Scene manager: 83.33%

**Coverage report:**
```bash
pytest --cov=fbx_tool --cov-report=term-missing
```

## Test Markers

Tests can be marked for selective execution:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.fbx
def test_requires_fbx_sdk():
    pass
```

**Run tests by marker:**
```bash
pytest -m unit        # Only unit tests
pytest -m "not slow"  # Skip slow tests
pytest -m fbx         # Only FBX SDK tests
```

## Debugging Tests

### Print Output
```bash
pytest -s  # Show print statements
```

### Drop into Debugger
```python
def test_something():
    import pdb; pdb.set_trace()
    # ... test code
```

### Verbose Output
```bash
pytest -vv  # Extra verbose
```

### Show Full Traceback
```bash
pytest --tb=long   # Long traceback
pytest --tb=short  # Short traceback (default)
pytest --tb=line   # One line per failure
```

## Continuous Integration

Tests run automatically on:
- Every commit (unit tests)
- Pull requests (all tests)
- Before releases (all tests with coverage)

**GitHub Actions** (if configured):
```yaml
- name: Run tests
  run: |
    pytest tests/unit/ --cov=fbx_tool --cov-report=xml
```

## Best Practices

### DO:
- ✅ Write tests for all new features
- ✅ Use descriptive test names
- ✅ Test edge cases and error conditions
- ✅ Use fixtures for shared setup
- ✅ Keep tests fast (< 1 second for unit tests)
- ✅ Mock external dependencies

### DON'T:
- ❌ Use hardcoded file paths
- ❌ Make tests depend on each other
- ❌ Test implementation details
- ❌ Ignore failing tests
- ❌ Skip coverage checks

## Related Documentation

- **[docs/testing/MOCK_SETUP_PATTERNS.md](../docs/testing/MOCK_SETUP_PATTERNS.md)** - FBX SDK mocking patterns
- **[docs/onboarding/README.md](../docs/onboarding/README.md)** - TDD workflow
- **pytest.ini** - Pytest configuration

---

**Last Updated:** 2025-10-18
