# Agentic Development Workflow

This guide is the shared entry point for Codex-based and other agentic development workflows in FBX Tool. It captures repository rules that are independent of any one assistant runtime.

## Operating Principles

- Preserve user changes. Check `git status --short` before edits and do not revert unrelated work.
- Keep work scoped to the requested behavior. Prefer small, reviewable changes with targeted tests.
- Use Python 3.10.x only. The Autodesk FBX Python SDK is a separate local prerequisite and is not vendored.
- Follow TDD for new behavior: write a failing test, implement, then rerun the focused test and an appropriate broader gate.
- Prefer adaptive, data-driven algorithms over hardcoded skeleton names, fixed scales, or animation-specific thresholds.
- Use the scene manager for FBX loading instead of direct ad hoc scene ownership.

## Required Read Order

1. `AGENTS.md` for repository contribution basics.
2. `docs/agentic/WORKFLOW.md` for agentic workflow rules.
3. `docs/agentic/BASELINE.md` for current verification status and known gates.
4. `docs/agentic/ROLE_ROUTING.md` to pick the right review lens.
5. `docs/quick-reference/COMMANDS.md` for command details.
6. `docs/development/FBX_SDK_FIXES.md` before touching FBX SDK APIs.
7. `docs/architecture/SCENE_MANAGEMENT.md` before changing file loading or scene lifetime behavior.
8. `docs/testing/MOCK_SETUP_PATTERNS.md` before writing FBX SDK-heavy tests.

## Development Loop

1. State scope, forbidden scope, and expected proof before editing.
2. Inspect nearby code and tests before choosing an implementation pattern.
3. Add or update tests first when behavior changes.
4. Keep implementation changes close to the owning module.
5. Run the narrowest meaningful verification first.
6. Run `scripts/verify-fast.ps1` before handoff when dependencies are available.
7. Record skipped checks and the exact reason in the final handoff.

## Proof Expectations

Use concrete command output as evidence. For code changes, include at least one focused pytest command. For shared analysis, packaging, or workflow changes, prefer `scripts/verify-fast.ps1`; add `-IncludeStyle` when the task includes formatting cleanup or the style baseline is known green. Check `docs/agentic/BASELINE.md` before using `scripts/verify-full.ps1`; the repo-wide pre-commit phase may expose known cleanup debt.

Do not claim a clean baseline if the repo already has unrelated dirty files or pre-existing test failures. Separate "verified by this change" from "pre-existing project state."
