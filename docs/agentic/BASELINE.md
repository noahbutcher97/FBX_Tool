# Agentic Baseline

Last checked: 2026-06-20

## Default Green Gate

Command:

```powershell
.\scripts\verify-fast.ps1 -PytestTarget 'tests/unit'
```

Result:

- Python 3.10 check passed.
- Unit tests: 485 passed, 32 skipped, 5 warnings.
- Runtime: approximately 1.5 seconds after collection in the local `.fbxenv`.

Use this as the default Codex handoff gate for ordinary code and documentation tasks.

## Strict Green Gate

Command:

```powershell
.\scripts\verify-fast.ps1 -IncludeStyle -PytestTarget 'tests/unit'
```

Result:

- Python 3.10 check passed.
- Black check passed.
- isort check passed.
- Unit tests: 485 passed, 32 skipped, 5 warnings.
- Runtime: approximately 1.4 seconds after collection in the local `.fbxenv`.

Use this when the task touches Python source, test files, examples, or scripts and the local FBX SDK environment is available.

## Working Tree Caveat

This baseline was recorded while the repository already had unrelated modified and untracked files. Future workers must run `git status --short` before editing and preserve unrelated user changes.

## CI Caveat

GitHub-hosted CI is an SDK-free hygiene gate only. The real test gate is local or self-hosted because unit tests import Autodesk's proprietary `fbx` module. Use `scripts/verify-fast.ps1` on a machine with Python 3.10 and the FBX SDK installed.
