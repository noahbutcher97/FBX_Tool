# Developer Onboarding

Use this page for a quick orientation to the FBX Tool codebase. For contributor rules and Codex-based workflows, start with `AGENTS.md` and `docs/agentic/WORKFLOW.md`.

## Current Entry Points

1. `AGENTS.md` - repository structure, commands, style, tests, and PR expectations.
2. `docs/agentic/WORKFLOW.md` - agentic development loop and proof expectations.
3. `docs/agentic/BASELINE.md` - current local and CI verification gates.
4. `docs/agentic/RELEASE_CHECKLIST.md` - pre-merge and post-merge workflow.
5. `docs/development/FBX_SDK_FIXES.md` - FBX SDK API patterns to check before touching SDK calls.
6. `docs/testing/MOCK_SETUP_PATTERNS.md` - FBX SDK mock setup patterns for unit tests.

## Project Map

```text
fbx_tool/
  analysis/        Core FBX loading, motion extraction, and analysis modules
  gui/             PyQt application entry point and UI orchestration
  visualization/   OpenGL and matplotlib viewers
tests/
  unit/            Fast isolated tests, usually with SDK mocks
  integration/     Multi-component and real-file workflows
docs/
  agentic/         Current workflow, templates, baseline, and release checklist
  architecture/    System design and scene-management notes
  development/     SDK patterns, audits, and known implementation gaps
  testing/         Mocking and test-writing guidance
```

## Development Principles

- Use Python 3.10 and the local Autodesk FBX Python SDK; the SDK is not vendored.
- Prefer adaptive, data-driven analysis over hardcoded skeleton names, scales, or thresholds.
- Follow TDD for behavior changes: write the failing test, implement the smallest fix, then rerun focused and broader gates.
- Keep FBX scene ownership consistent with `docs/architecture/SCENE_MANAGEMENT.md`.
- Preserve unrelated user changes and keep branches scoped to one behavior or docs update.

## Verification

Use targeted tests while developing, then run the current baseline gate before handoff:

```powershell
.\scripts\verify-fast.ps1 -PytestTarget 'tests/unit'
```

For Python, CI, hook, or repo-wide documentation changes, also run:

```powershell
.\scripts\verify-fast.ps1 -IncludeStyle -PytestTarget 'tests/unit'
.\.fbxenv\Scripts\python.exe -m pre_commit run --all-files
```

GitHub-hosted CI is SDK-free. It validates pre-commit hooks, PowerShell script parsing, and required agentic docs. Local `verify-fast.ps1` remains the real unit-test gate because unit tests import Autodesk's proprietary `fbx` module.

## Finding Work

Use `docs/audits/MODULE_ERROR_LOGIC_AUDIT.md` and `docs/development/INCOMPLETE_MODULES.md` for implementation gaps, but verify current code and tests before treating older notes as active defects. Historical session notes are useful context, not a substitute for current repo inspection.
