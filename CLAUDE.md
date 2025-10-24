# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**🚨 FIRST TIME HERE?** → **[docs/onboarding/CLAUDE_START_HERE.md](docs/onboarding/CLAUDE_START_HERE.md)**

## Documentation Philosophy

**PREFER EVOLUTION OVER PROLIFERATION:**
- **Update existing documents** rather than creating new ones
- **Consolidate** related information into single, evolving files
- **Archive deprecated content** to `docs/audits/archive/` rather than deleting
- Maintain a **cohesive knowledge base** in `docs/` with minimal file count
- New files should only be created when adding **genuinely new categories** of information
- Keep documentation **current** - outdated docs are worse than no docs

## Specialized Agents

This project has 9 specialized agents for different development domains. Use `/agents` to see the full list.

### Domain-General Agents (8)

1. **algorithm-architect** - Algorithm design, data structures, statistical methods
   - Use when: Designing detection algorithms, choosing data structures, optimizing complexity
   - Examples: "Make jitter detection adaptive", "Design gait cycle algorithm", "Optimize this O(n²) loop"

2. **data-quality-specialist** - Validation, robustness, edge cases, confidence scoring
   - Use when: Handling edge cases, adding validation, implementing confidence scores
   - Examples: "Handle NaN values", "Add confidence to detection", "Validate user input"

3. **integration-engineer** - System integration, component interaction, pipelines
   - Use when: Connecting modules, designing data flow, refactoring coupling
   - Examples: "Integrate new module into pipeline", "GUI calls FBX SDK directly - fix this", "Design data contract"

4. **test-architect** - Testing methodology, TDD, coverage
   - Use when: Writing tests (TDD), improving coverage, test quality review
   - Examples: "Create tests for jump detection", "Improve coverage to 80%", "Tests are too minimal"

5. **code-auditor** - Pattern compliance, coverage analysis, code quality
   - Use when: Auditing code, finding hardcoded values, reviewing implementations
   - Examples: "Audit velocity_analysis", "Find hardcoded thresholds", "Review new feature"

6. **performance-optimizer** - Performance engineering, memory, caching
   - Use when: Performance issues, memory leaks, optimization needed
   - Examples: "GUI is slow", "Memory keeps growing", "Optimize this analysis"

7. **debug-resolver** - Debugging stuck situations, fresh perspective
   - Use when: Stuck after multiple attempts, mysterious failures
   - Examples: "Test still failing after 3 attempts", "Works sometimes, crashes others"

8. **doc-curator** - Documentation management, evolution over proliferation
   - Use when: Updating docs, resolving conflicts, archiving completed work
   - Examples: "Update docs after feature", "Consolidate threshold documentation", "Archive fixed issues"

### Project-Specific Agent (1)

9. **fbx-project-guide** - FBX Tool patterns, FBX SDK usage, project conventions
   - Use when: Need project guidance, FBX SDK help, pattern enforcement
   - Examples: "Correct FBX SDK usage?", "Should I use hardcoded threshold?", "Python version?"

### Agent Usage Best Practices

- **Be specific about domain**: "algorithm-architect for threshold design" not "help with thresholds"
- **Use proactively**: Don't wait to get stuck - consult debug-resolver early
- **Combine agents**: Use test-architect + algorithm-architect for TDD workflow
- **Trust specialized knowledge**: Agents have deep domain expertise
- **Check fbx-project-guide first**: For project-specific questions (FBX SDK, Python version, patterns)

---

## Quick Reference

### Project Overview

FBX Tool is a professional desktop application for analyzing FBX animation files with biomechanical motion processing and real-time 3D visualization. Works universally across any skeleton (Mixamo, Unity, Blender, custom).

**Critical Constraint:** Python 3.10.x ONLY (FBX SDK limitation)

### Essential Commands

See **[docs/quick-reference/COMMANDS.md](docs/quick-reference/COMMANDS.md)** for complete reference.

```bash
# Quick start
pytest                                    # Run all tests
black fbx_tool/ tests/                    # Format code
python fbx_tool/gui/main_window.py        # Launch GUI
```

## Architecture

### Analysis Pipeline

```
1. FBX Loading → Multi-stack ranking (fbx_loader.py)
2. Data Extraction → Dopesheet, joints, chains (utils.py)
3. Motion Analysis → Velocity, gait, contacts, root motion
4. Advanced Analysis → Pose validity, transitions, classification
```

**Detailed architecture:** [docs/architecture/](docs/architecture/)

### Key Patterns

1. **Scene Manager (MANDATORY)** - Use reference-counted scene loading
   ```python
   from fbx_tool.analysis.scene_manager import get_scene_manager
   with get_scene_manager().get_scene("file.fbx") as scene_ref:
       analyze(scene_ref.scene)
   ```
   See [docs/architecture/SCENE_MANAGEMENT.md](docs/architecture/SCENE_MANAGEMENT.md)

2. **Dynamic Chain Detection** - `utils.py:detect_chains_from_hierarchy()`
   - No hardcoded bone names
   - Works with any skeleton hierarchy

3. **Adaptive Thresholds** - Data-driven, not hardcoded
   - Use percentiles, coefficient of variation
   - Scale with animation properties
   - See [docs/architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md](docs/architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md)

## FBX SDK Critical Patterns

**IMPORTANT:** The FBX SDK has non-obvious APIs. **ALWAYS** consult **[docs/development/FBX_SDK_FIXES.md](docs/development/FBX_SDK_FIXES.md)** before using FBX SDK.

### Quick Examples

❌ **WRONG:** `time_span = scene.GetGlobalSettings().GetTimeSpan(fbx.FbxTime.eGlobal)`
✅ **CORRECT:** `time_span = anim_stack.GetLocalTimeSpan()`

❌ **WRONG:** `curve = node.GetAnimationCurve(0)`
✅ **CORRECT:** Access through property curve nodes with `GetCurveNode(anim_layer)`

## Test-Driven Development (TDD)

**MANDATORY:** All new features must follow strict TDD.

### TDD Workflow

1. **Write tests FIRST** - Define expected behavior
2. **Run tests** - Watch them fail (red)
3. **Implement robustly** - Not minimal code, but complete solution
4. **Refactor** - Keep tests green
5. **Add edge cases** - Expand coverage iteratively

### Writing Good Tests

**CRITICAL:** Tests must demand robust implementations, not just pass with placeholders.

**Detailed examples:** [docs/quick-reference/TDD_EXAMPLES.md](docs/quick-reference/TDD_EXAMPLES.md)

```python
# ❌ BAD: Too minimal
def test_detect_contacts():
    assert detect_foot_contacts([], []) is not None  # Passes with "return []"

# ✅ GOOD: Demands robust implementation
def test_detect_contacts_with_clear_ground_strike():
    positions = np.array([[0,20,0], [0,10,0], [0,0,0], [0,0,0], [0,5,0]])
    velocities = np.diff(positions[:, 1])
    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    assert len(contacts) == 1
    assert contacts[0]['start_frame'] == 2
    assert 0.0 <= contacts[0]['confidence'] <= 1.0
```

### Coverage Requirements

- **Minimum:** 20% (enforced by pytest.ini)
- **Target:** 80% for new modules
- **Reference:** `gait_analysis.py` (88% coverage)

### Test Organization

- `tests/unit/` - Fast, isolated tests
- `tests/integration/` - Multi-component tests
- `tests/conftest.py` - Shared fixtures

**Study:** `tests/unit/test_gait_analysis.py` for comprehensive patterns

## Design Principles

### 1. No Hardcoded Assumptions

❌ **BAD:** `THRESHOLD = 10.0` (breaks on different scales)
✅ **GOOD:** `threshold = np.percentile(velocities, 25)` (adaptive)

❌ **BAD:** `foot = bones[-2]` (assumes structure)
✅ **GOOD:** Detect by name matching with fallbacks

### 2. Separation of Concerns

- **Detection** (what) vs. **Classification** (what type)
- **Metrics** vs. **Thresholds**
- **Extraction** vs. **Analysis**

### 3. Confidence Scores

Every analysis returns confidence:
```python
return {
    'result': value,
    'confidence': 0.85,  # [0,1]
    'method': 'adaptive_percentile',
    'warnings': []
}
```

### 4. Graceful Degradation

- Empty data → Return zeros with low confidence
- Missing bones → Analyze available, log skipped
- Corrupt data → Skip problematic frames, report in summary

See [docs/development/EDGE_CASE_PATTERNS.md](docs/development/EDGE_CASE_PATTERNS.md) for patterns.

## Known Issues & Incomplete Modules

See **[docs/development/INCOMPLETE_MODULES.md](docs/development/INCOMPLETE_MODULES.md)** for details.

### Modules Requiring TDD (Write Tests First!)

- `directional_change_detection.py` - 0% coverage
- `motion_transition_detection.py` - 0% coverage
- `temporal_segmentation.py` - 0% coverage
- `motion_classification.py` - 0% coverage
- `constraint_violation_detection.py` - Contains TODO placeholders

### Algorithm Issues

From **[docs/development/ALGORITHM_ISSUES.md](docs/development/ALGORITHM_ISSUES.md)**:

**gait_analysis.py:**
- Line 165: Cycle rate calculation WRONG (contact rate, not cycle rate)
- Line 110: Stride length uses Y-axis instead of horizontal
- Line 112: Asymmetry always 0.0

**velocity_analysis.py:**
- Lines 702-707: NaN propagation in chain coherence
- Magic numbers (0.1, 0.01, 0.25, 0.4, 0.7) undocumented

**chain_analysis.py:**
- Lines 69-71: Undocumented IK score formula
- Lines 122-128: Temporal coherence may inflate scores

## Documentation Map

**Start here:** [docs/onboarding/README.md](docs/onboarding/README.md)

### Critical Reading (Read Before Coding!)

1. **[docs/development/FBX_SDK_FIXES.md](docs/development/FBX_SDK_FIXES.md)** - FBX SDK patterns (MUST READ!)
2. **[docs/development/INCOMPLETE_MODULES.md](docs/development/INCOMPLETE_MODULES.md)** - Current issues
3. **[docs/development/ALGORITHM_ISSUES.md](docs/development/ALGORITHM_ISSUES.md)** - Algorithm correctness issues
4. **[docs/development/EDGE_CASE_PATTERNS.md](docs/development/EDGE_CASE_PATTERNS.md)** - Edge case handling

### Quick Reference

- **[docs/quick-reference/COMMANDS.md](docs/quick-reference/COMMANDS.md)** - Dev commands
- **[docs/quick-reference/TDD_EXAMPLES.md](docs/quick-reference/TDD_EXAMPLES.md)** - Test patterns

### Architecture

- **[docs/architecture/SCENE_MANAGEMENT.md](docs/architecture/SCENE_MANAGEMENT.md)** - Scene manager system
- **[docs/architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md](docs/architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md)** - Adaptive design

### Testing

- **[docs/testing/MOCK_SETUP_PATTERNS.md](docs/testing/MOCK_SETUP_PATTERNS.md)** - FBX SDK mocking
- **pytest.ini** - Test config, markers, thresholds
- **tests/conftest.py** - Shared fixtures

### User Documentation

- **[docs/INSTALL.md](docs/INSTALL.md)** - Python 3.10 setup, FBX SDK
- **[docs/3D_VIEWER_GUIDE.md](docs/3D_VIEWER_GUIDE.md)** - OpenGL controls
- **[README.md](README.md)** - Features, quick start

### Changelog

- **[docs/changelog/SESSION_HISTORY.md](docs/changelog/SESSION_HISTORY.md)** - Major session updates
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Release history

## Common Pitfalls

1. **Python 3.11+** - FBX SDK only supports 3.10.x
2. **Wrong FBX SDK APIs** - Check [docs/development/FBX_SDK_FIXES.md](docs/development/FBX_SDK_FIXES.md) first!
3. **Implementing before testing** - TDD is mandatory
4. **Minimal tests** - Tests must demand robust code
5. **Hardcoded thresholds** - Use adaptive, data-driven values
6. **Assuming skeleton structure** - Support any naming convention
7. **Forgetting edge cases** - Empty, single frame, NaN/inf
8. **Not using scene manager** - All FBX loading must use reference counting
9. **Magic numbers** - Extract to named constants
10. **Silent failures** - Log warnings when skipping data

## Development Workflow

1. **Read docs** - Especially FBX_SDK_FIXES.md, INCOMPLETE_MODULES.md
2. **Pick task** - From incomplete modules or TODO
3. **Write tests FIRST** - Study `test_gait_analysis.py` for patterns
4. **Run tests** - `pytest tests/unit/test_module.py -v`
5. **Implement robustly** - No hardcoded values, adaptive algorithms
6. **Verify tests pass**
7. **Add edge cases** - Empty, single frame, NaN, extreme values
8. **Format** - `black . && isort . --profile=black`
9. **Check coverage** - `pytest --cov=fbx_tool.analysis.module_name` (target 80%+)
10. **Pre-commit** - `pre-commit run --all-files`
11. **Commit** - Descriptive message

## Module Entry Points

- **GUI:** `fbx_tool/gui/main_window.py`
- **CLI:** `fbx_tool/__main__.py` (enables `python -m fbx_tool`)
- **Examples:** `examples/run_analysis.py`
- **Core Utils:** `fbx_tool/analysis/utils.py` (I/O, chain detection)
- **FBX Loading:** `fbx_tool/analysis/fbx_loader.py`
- **Scene Manager:** `fbx_tool/analysis/scene_manager.py` (reference counting)

## Output Structure

Analysis results saved to `output/<fbx_filename>/`:

```
output/your_animation/
├── dopesheet.csv                      # Frame-by-frame rotations
├── joint_enhanced_relationships.csv   # Per-joint metrics
├── chain_confidence.csv               # IK confidence
├── chain_gait_segments.csv            # Stride timing
├── gait_summary.csv                   # Cycle metrics
├── velocity_analysis.csv              # Motion derivatives
├── foot_contacts.csv                  # Ground contacts
├── root_motion_analysis.csv           # Trajectory
├── procedural_metadata.json           # Adaptive thresholds
└── analysis_summary.json              # Complete summary
```

## Success Metrics

### ✅ You're doing well when:

- Tests written BEFORE implementation
- Coverage increases (80%+ target)
- Works across diverse assets (not just Mixamo)
- Code self-documenting
- Following FBX SDK patterns from docs
- Graceful edge case handling

### ❌ Course-correct when:

- Implementing before testing
- Hardcoded thresholds for specific animations
- Skipping edge cases
- Assuming character scale/hierarchy
- Wrong FBX SDK patterns
- Tests pass with placeholders

## Recent Updates

See **[docs/changelog/SESSION_HISTORY.md](docs/changelog/SESSION_HISTORY.md)** for complete history.

### Latest: Session 2025-10-19c - Critical Bug Fixes & GUI Improvements

- ✅ Fixed 5 critical crashes preventing analysis pipeline from running
  - **Foot Contact Analysis:** KeyError 'up_axis' - Now uses explicit `detect_full_coordinate_system()` call
  - **Directional Change Detection:** KeyError 'angular_velocity_y' - Updated to procedural field name
  - **Foot Contact Array Mismatch:** Removed obsolete np.diff alignment code
  - **GUI Batch Buttons:** Add to Batch buttons now re-enable after analysis completes
  - **Visualization Button:** Self-resolved (likely fixed by coordinate system improvements)
- ✅ Analysis pipeline now runs end-to-end without crashes
- ✅ Batch processing workflow fully functional
- 📄 Updated: `docs/development/INCOMPLETE_MODULES.md`, `docs/audits/MODULE_ERROR_LOGIC_AUDIT.md`

**User validation:** All crashes resolved, pipeline working as expected

### Session 2025-10-19b - Agent Architecture Refactoring

- ✅ Created 3 new domain-general agents (algorithm-architect, data-quality-specialist, integration-engineer)
- ✅ Renamed 4 agents (removed fbx- prefix from test-architect, doc-curator, debug-resolver, performance-optimizer)
- ✅ Split fbx-audit-codegen → code-auditor (auditing only)
- ✅ Merged fbx-gui-integrator → integration-engineer
- 📊 **Final suite: 9 agents** (8 domain-general, 1 project-specific)

### Session 2025-10-19a - Foot Contact Visualization Fix

- ✅ Fixed context-aware stuck bone detection in OpenGL viewer
- ✅ Added 22 unit tests + 6 integration tests for foot contact visualization
- ✅ User validation: "Okay finally this is working as expected"

### Session 2025-10-18 - Procedural Threshold System

- ✅ Cached derivatives in trajectory extraction (~3x speedup)
- ✅ Adaptive motion state detection (no hardcoded thresholds)
- ✅ Coefficient of variation for single-state detection
- ✅ Percentage-based minimum duration (frame-rate aware)
- ✅ Procedural metadata export system
- 📄 New: `docs/development/HARDCODED_CONSTANTS_AUDIT.md`

**Next priorities:**
1. **Adaptive Stuck Bone Detection** (Session 2025-10-19) - Make stuck transform classification relative to animation movement
   - Analyze average movement amount across animation per bone
   - Use statistical measures (mean, std, CV) to calculate stuck thresholds dynamically
   - Handle edge cases: slow-motion, minimal-movement animations
   - Current implementation uses fixed threshold: `y <= ground_height + 1.0`
   - Proposed: Calculate typical bone movement, then classify "stuck" relative to that
2. Jitter detection adaptive thresholds
3. Constraint confidence fix (misleading 1.0 when 0 chains)
4. Foot sliding threshold proceduralization

### Session 2025-10-17 - Scene Manager Architecture

- ✅ Reference-counted scene management (66-90% memory savings)
- ✅ Smart caching (visualizer keeps current ± 1 files)
- ✅ All 10 integration tests passing
- 📄 New: `docs/architecture/SCENE_MANAGEMENT.md`
- 📄 New: `docs/testing/MOCK_SETUP_PATTERNS.md`

**Breaking change:** All FBX loading must use scene manager (context managers preferred)
