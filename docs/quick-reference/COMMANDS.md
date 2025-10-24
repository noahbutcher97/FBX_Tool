# Development Commands Quick Reference

## Environment Setup

```bash
# Create virtual environment (Python 3.10 required!)
python -m venv .fbxenv --system-site-packages
.fbxenv\Scripts\activate  # Windows
source .fbxenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # For development
```

## Testing

```bash
# Run all tests with coverage (parallel execution via pytest-xdist)
pytest

# Run specific test file
pytest tests/unit/test_gait_analysis.py -v

# Run single test function
pytest tests/unit/test_gait_analysis.py::test_detect_stride_segments_normal_gait -v

# Run tests without parallelization (useful for debugging)
pytest -n 0

# Run only fast unit tests (skip integration/slow)
pytest -m "unit and not slow"

# Run tests requiring FBX SDK
pytest -m fbx

# Skip coverage for faster runs
pytest --no-cov

# Show local variables on failure
pytest -l

# Re-run only last failed tests
pytest --lf

# Generate HTML coverage report
pytest --cov-report=html
# Opens: htmlcov/index.html
```

## Code Quality

```bash
# Format code (Black, 120 char line length)
black fbx_tool/ tests/

# Sort imports
isort fbx_tool/ tests/ --profile=black --line-length=120

# Lint code
flake8 fbx_tool/ tests/ --max-line-length=120 --extend-ignore=E203,W503

# Type checking
mypy fbx_tool/ --ignore-missing-imports --python-version=3.10

# Run all pre-commit hooks
pre-commit run --all-files
```

## Running the Application

```bash
# GUI mode
python fbx_tool/gui/main_window.py

# CLI mode (module entry point)
python -m fbx_tool

# CLI with specific file
python examples/run_analysis.py path/to/animation.fbx
```

## Building Executable

```bash
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool/gui/main_window.py
# Output: dist/FBX_Tool.exe
```

## Test Markers Reference

```python
@pytest.mark.unit  # Fast, isolated tests
@pytest.mark.integration  # Multi-component tests
@pytest.mark.gui  # GUI tests (requires display)
@pytest.mark.slow  # Long-running tests
@pytest.mark.fbx  # Tests requiring FBX SDK
```

## Coverage Requirements

From `pytest.ini`:
- **Enforced minimum:** 20% (`--cov-fail-under=20`)
- **Target for new modules:** 80%+
- **Reference standard:** gait_analysis.py (88% coverage)

## Code Quality Standards

From `pyproject.toml` and `.pre-commit-config.yaml`:
- **Black formatter:** 120 character line length
- **isort:** Black-compatible profile
- **flake8:** Max line 120, ignore E203, W503
- **mypy:** Type hints with `--ignore-missing-imports`
- **interrogate:** 50% docstring coverage minimum

## Pre-commit Hooks

All hooks in `.pre-commit-config.yaml` must pass:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security checks (bandit)
- Docstring coverage (interrogate: 50% minimum)
- File integrity checks
