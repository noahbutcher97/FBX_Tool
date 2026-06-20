# Agentic Role Routing

Use these roles as review lenses or worker prompts. They are intentionally tool-neutral so Codex workers can apply them without depending on assistant-specific agent files.

## Project Guide

Use for FBX Tool conventions: Python 3.10, Autodesk FBX SDK constraints, scene manager usage, package layout, and existing documentation paths. Start here when a task is unclear.

## Test Architect

Use for TDD plans, fixture design, coverage improvement, and deciding whether a behavior belongs in `tests/unit/`, `tests/integration/`, or an `fbx`-marked test.

## Algorithm Architect

Use for gait, velocity, contact, transition, classification, and confidence algorithms. This lens should challenge hardcoded thresholds and require data-driven behavior.

## Data Quality Specialist

Use for NaN/inf handling, empty inputs, single-frame animations, low-confidence outputs, warnings, validation, and graceful degradation.

## Integration Engineer

Use when connecting analysis modules to CLI, GUI, scene loading, output files, or visualization. This lens owns contracts between modules.

## Code Auditor

Use for review passes before handoff. Focus on regressions, missing tests, pattern violations, dead code, and differences between claimed and verified behavior.

## Performance Optimizer

Use for memory pressure, large FBX files, repeated scene loads, caching, vectorization, and GUI responsiveness.

## Debug Resolver

Use after repeated failures or unclear symptoms. Require reproduction steps, the exact failing command, and evidence from logs or tracebacks before proposing fixes.

## Doc Curator

Use for documentation updates. Prefer evolving existing docs over creating new ones, and keep command references aligned with scripts and CI.
