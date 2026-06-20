# Agentic Baseline

Last checked: 2026-06-20

## Green Gate

Command:

```powershell
.\scripts\verify-fast.ps1 -PytestTarget 'tests/unit'
```

Result:

- Python 3.10 check passed.
- Unit tests: 485 passed, 32 skipped, 5 warnings.
- Runtime: approximately 1.5 seconds after collection in the local `.fbxenv`.

Use this as the default Codex handoff gate for ordinary code and documentation tasks.

## Known Non-Green Gate

Command:

```powershell
.\scripts\verify-fast.ps1 -IncludeStyle -PytestTarget 'tests/unit'
```

Result:

- Fails at Black before pytest.
- Black reports 12 files would be reformatted.

Current Black debt:

- `fbx_tool/analysis/dopesheet_export.py`
- `fbx_tool/analysis/fbx_loader.py`
- `fbx_tool/analysis/gait_summary.py`
- `tests/debug/inspect_animation_layers.py`
- `tests/debug/inspect_bones.py`
- `tests/debug/test_motion_states.py`
- `tests/exploratory/test_animation_extraction.py`
- `tests/exploratory/test_animation_variance.py`
- `tests/exploratory/test_chain_detection.py`
- `tests/exploratory/test_fbx_coordinate_system.py`
- `tests/exploratory/test_stack_1.py`
- `tests/integration/test_foot_contact_visualization_integration.py`

Do not treat these formatting failures as caused by unrelated future tasks unless those tasks edit the listed files.

## Working Tree Caveat

This baseline was recorded while the repository already had unrelated modified and untracked files. Future workers must run `git status --short` before editing and preserve unrelated user changes.

## CI Caveat

GitHub-hosted CI is an SDK-free hygiene gate only. The real test gate is local or self-hosted because unit tests import Autodesk's proprietary `fbx` module. Use `scripts/verify-fast.ps1` on a machine with Python 3.10 and the FBX SDK installed.
