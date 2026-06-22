# Agentic Baseline

Last checked: 2026-06-22

## Default Green Gate

Command:

```powershell
.\scripts\verify-fast.ps1 -PytestTarget 'tests/unit'
```

Result:

- Python 3.10 check passed.
- Unit tests: 509 passed.
- Runtime: 1.28 seconds after collection in the local `.fbxenv`.

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
- Unit tests: 509 passed.
- Runtime: 1.35 seconds after collection in the local `.fbxenv`.

Use this when the task touches Python source, test files, examples, or scripts and the local FBX SDK environment is available.

## Repo-Wide Hook Gate

Command:

```powershell
.\.fbxenv\Scripts\python.exe -m pre_commit run --all-files
```

Result:

- Black passed.
- isort passed.
- flake8 passed.
- mypy passed.
- bandit passed.
- General file hooks passed.
- interrogate passed.
- codespell passed.

Use this before release-sensitive, CI, hook, script, or repo-wide documentation changes.

## Working Tree State

This baseline was recorded with `git status --short` clean. Future workers must still run `git status --short` before editing and preserve unrelated user changes if any appear.

## CI Caveat

GitHub-hosted CI is an SDK-free hygiene gate. It runs `pre-commit run --all-files`, validates key PowerShell scripts parse, and verifies the required agentic docs are present. The real test gate is local or self-hosted because unit tests import Autodesk's proprietary `fbx` module. Use `scripts/verify-fast.ps1` on a machine with Python 3.10 and the FBX SDK installed.
