# Repository Guidelines

## Project Structure & Module Organization

`fbx_tool/` is the main Python package. Core FBX loading and motion analysis live in `fbx_tool/analysis/`, the PyQt entry point is `fbx_tool/gui/main_window.py`, and viewer code lives in `fbx_tool/visualization/`. Use `examples/` for runnable CLI or visualization samples, `docs/` for architecture, setup, and testing notes, and `assets/` for icons or screenshots. Tests are organized under `tests/unit/`, `tests/integration/`, plus non-CI exploratory/debug scripts under `tests/exploratory/` and `tests/debug/`.

For Codex-based workflows, start with `docs/agentic/WORKFLOW.md`, use `docs/agentic/ROLE_ROUTING.md` to choose the review lens, use `docs/agentic/RELEASE_CHECKLIST.md` before merging, and use the task and handoff templates in `docs/agentic/` for multi-step work.

## Build, Test, and Development Commands

Use Python 3.10; the Autodesk FBX Python SDK is installed separately and is not bundled.

```powershell
.\setup-environment.ps1
.fbxenv\Scripts\activate
pip install -r requirements-dev.txt
python -m fbx_tool path\to\animation.fbx
python fbx_tool\gui\main_window.py
pytest
pytest tests\unit\test_gait_analysis.py -v
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool\gui\main_window.py
```

`pytest` uses `pytest.ini`, runs with coverage and xdist, and enforces the current 20% coverage floor. Build artifacts go to `dist/`; coverage HTML goes to `htmlcov/`.

## Coding Style & Naming Conventions

Format Python with Black at 120 columns and sort imports with isort using the Black profile. Pre-commit runs flake8 as a high-signal syntax and undefined-name gate; do not use it as a broad style formatter. Mypy runs with missing imports ignored for SDK-heavy code. Use `snake_case` for modules, functions, variables, and generated CSV/JSON output names. Test functions should describe behavior, for example `test_detect_stride_segments_normal_gait`.

## Testing Guidelines

Place fast isolated tests in `tests/unit/` and multi-component checks in `tests/integration/`. Use pytest markers from `pytest.ini`: `unit`, `integration`, `gui`, `slow`, and `fbx`. Mock FBX SDK behavior for unit tests; reserve real SDK or file-dependent cases for integration or `fbx`-marked tests. Prefer targeted runs while developing, then run `pytest` before handing off.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects and Conventional-style prefixes such as `fix:`, `docs:`, and `test:`. Keep commits scoped to one behavior or documentation change. Pull requests should include a concise summary, the commands run, linked issue context when available, and screenshots or sample output when GUI, viewer, or analysis results change.

## Security & Configuration Tips

Do not commit local FBX files, generated `output/`, `dist/`, `build/`, `htmlcov/`, or virtual environments. Keep Autodesk SDK installation steps in docs rather than vendoring SDK files. Run `pre-commit run --all-files` when changing shared analysis, packaging, or CI-sensitive files.
