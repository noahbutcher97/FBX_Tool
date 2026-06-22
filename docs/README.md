# FBX Tool Documentation

## Documentation Structure

This directory contains all project documentation organized by category.

### 📖 Getting Started

**New to the project? Start here:**

1. **[../AGENTS.md](../AGENTS.md)** - Contributor guide, commands, style, and PR expectations
2. **[agentic/WORKFLOW.md](agentic/WORKFLOW.md)** - Codex and agentic development workflow
3. **[INSTALL.md](INSTALL.md)** - Environment setup, Python 3.10 installation, FBX SDK setup
4. **[onboarding/README.md](onboarding/README.md)** - Developer orientation and project map
5. **[../README.md](../README.md)** - User-facing features and quick start guide

### 🏗️ Architecture

**Understanding the system design:**

- **[architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)** - Reference-counted scene manager architecture
  - Problem statement and motivation
  - Component overview (FBXSceneManager, FBXSceneReference)
  - Integration patterns (GUI, Visualizer, Analysis Worker)
  - Smart caching strategy
  - Workflow examples and performance impact

- **[architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md](architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md)** - Procedural design principles
  - Core principle: "Don't assume. Discover."
  - Data-driven thresholds and scale invariance
  - Confidence scoring patterns
  - Implementation status and future vision
  - Key learnings from Session 2025-10-18

- **[architecture/DATA_QUALITY_FRAMEWORK.md](architecture/DATA_QUALITY_FRAMEWORK.md)** - Data quality & confidence scoring (DESIGN DOC)
  - Quality scoring system for all analyses
  - Fallback motion detection strategies
  - Root bone → Next bone → Center of mass → Whole skeleton
  - AI/LLM integration metadata
  - Quality-driven adaptive thresholds

### 🧪 Testing

**Testing patterns and best practices:**

- **[testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md)** - FBX SDK mock patterns
  - Common mock setup issues and fixes
  - Complete integration test pattern
  - Mock checklist
  - Debugging strategies

### 🚀 Quick Reference

**Fast access to common commands and patterns:**

- **[quick-reference/COMMANDS.md](quick-reference/COMMANDS.md)** - Development commands
  - Environment setup
  - Testing commands (pytest options, markers)
  - Code quality (black, isort, flake8, mypy)
  - Running the application
  - Building executables
  - Coverage requirements

- **[quick-reference/TDD_EXAMPLES.md](quick-reference/TDD_EXAMPLES.md)** - Test-driven development patterns
  - TDD workflow steps
  - Good vs bad test examples
  - Test organization patterns
  - Using fixtures from conftest.py
  - Common test patterns (parametrized, confidence scores, adaptive thresholds)
  - Reference: test_gait_analysis.py breakdown

### 🔧 Development

**Development guidelines and issue tracking:**

- **[agentic/](agentic/)** - Current agentic workflow, task templates, handoff template, baseline, and release checklist
  - Start with `WORKFLOW.md`
  - Check `BASELINE.md` before claiming verification status
  - Use `RELEASE_CHECKLIST.md` before merging PRs

- **[development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)** - FBX SDK API patterns (MUST READ!)
  - Correct vs incorrect API usage
  - Time span access patterns
  - Animation curve access
  - Common pitfalls

- **[development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md)** - Current incomplete modules
  - Historical incomplete-module notes
  - Resolved placeholder behavior
  - Pointers to current audit status

- **[audits/MODULE_ERROR_LOGIC_AUDIT.md](audits/MODULE_ERROR_LOGIC_AUDIT.md)** - Current module status and follow-up priorities
  - Live source/test status by analysis module
  - Verification evidence and remaining risk areas
  - Current next-step recommendations

- **[agentic/TASK_TEMPLATE.md](agentic/TASK_TEMPLATE.md)** and **[agentic/HANDOFF_TEMPLATE.md](agentic/HANDOFF_TEMPLATE.md)** - Current task and handoff templates
  - Scope, proof, and forbidden-scope fields
  - Verification and skipped-check reporting
  - Return-format guidance for multi-step work

### 📊 Project Status

**Current state and improvement areas:**

- **[audits/MODULE_ERROR_LOGIC_AUDIT.md](audits/MODULE_ERROR_LOGIC_AUDIT.md)** - Current module audit findings
  - Source/test status by analysis module
  - Known follow-up areas
  - Current verification snapshot

- **[archive/ALGORITHM_ISSUES_2025-10-17_FIXED.md](archive/ALGORITHM_ISSUES_2025-10-17_FIXED.md)** - Historical algorithm issue notes

- **[archive/CODE_REVIEW_FINDINGS_2025-10-17_FIXED.md](archive/CODE_REVIEW_FINDINGS_2025-10-17_FIXED.md)** - Original code review (archived)

- **[development/EDGE_CASE_PATTERNS.md](development/EDGE_CASE_PATTERNS.md)** - Edge case handling patterns
  - Graceful degradation strategies
  - Validation patterns
  - Error handling recommendations

### 🎮 User Guides

**Application usage:**

- **[3D_VIEWER_GUIDE.md](3D_VIEWER_GUIDE.md)** - OpenGL viewer controls and keyboard shortcuts

## Documentation Categories

### By Role

#### For New Developers
1. Read [../AGENTS.md](../AGENTS.md)
2. Read [agentic/WORKFLOW.md](agentic/WORKFLOW.md)
3. Read [INSTALL.md](INSTALL.md)
4. Read [development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)
5. Review test examples in `../tests/unit/test_gait_analysis.py`

#### For Maintainers
1. Use [agentic/RELEASE_CHECKLIST.md](agentic/RELEASE_CHECKLIST.md) before merging PRs
2. Check [agentic/BASELINE.md](agentic/BASELINE.md) for current verification gates
3. Check [audits/MODULE_ERROR_LOGIC_AUDIT.md](audits/MODULE_ERROR_LOGIC_AUDIT.md) for current implementation gaps
4. Review [development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md) only for historical context
5. Consult [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md) for system design

#### For Users
1. Start with [../README.md](../README.md)
2. Install using [INSTALL.md](INSTALL.md)
3. Learn 3D viewer with [3D_VIEWER_GUIDE.md](3D_VIEWER_GUIDE.md)

### By Topic

#### Architecture & Design
- [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md) - Memory management
- [onboarding/README.md](onboarding/README.md) - Design principles
- [development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md) - API patterns

#### Testing & Quality
- [testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md) - Mock patterns
- [onboarding/README.md](onboarding/README.md) - TDD workflow
- `../tests/unit/test_gait_analysis.py` - Reference test examples

#### Issues & Improvements
- [audits/MODULE_ERROR_LOGIC_AUDIT.md](audits/MODULE_ERROR_LOGIC_AUDIT.md) - Current module status
- [development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md) - Historical incomplete-module notes
- [development/EDGE_CASE_PATTERNS.md](development/EDGE_CASE_PATTERNS.md) - Edge case handling

## Quick Reference

### Test Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fbx_tool

# Run specific test
pytest tests/unit/test_gait_analysis.py -v

# Run integration tests
pytest tests/integration/ -v
```

### Code Quality
```bash
# Format code
black fbx_tool/ tests/ --line-length=120

# Sort imports
isort fbx_tool/ tests/ --profile=black

# Run pre-commit hooks
pre-commit run --all-files
```

### Application
```bash
# GUI mode
python fbx_tool/gui/main_window.py

# CLI mode
python examples/run_analysis.py path/to/animation.fbx
```

## Contributing Documentation

When adding new documentation:

1. **Choose the right category:**
   - `architecture/` - System design, component relationships
   - `testing/` - Test patterns, mocking, quality
   - `development/` - API patterns, known issues, guidelines
   - `onboarding/` - Getting started, principles, workflows

2. **Follow the template:**
   - Overview section
   - Problem statement (if applicable)
   - Solution/pattern
   - Examples
   - Related documentation links

3. **Update this README** - Add your new doc to the appropriate section

4. **Cross-reference** - Link related docs together

5. **Keep it current** - Update docs when code changes

## Documentation TODOs

### Planned Documentation

- [ ] `architecture/ANALYSIS_PIPELINE.md` - Complete analysis flow
- [x] ~~`architecture/CACHING_STRATEGY.md`~~ - Covered in SCENE_MANAGEMENT.md
- [x] ~~`development/ADAPTIVE_THRESHOLDS.md`~~ - See `architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md` and `audits/MODULE_ERROR_LOGIC_AUDIT.md`
- [ ] `testing/INTEGRATION_TEST_PATTERNS.md` - Multi-component test examples
- [ ] `api/MODULES.md` - Module-by-module API reference
- [ ] `api/METADATA_SCHEMA.md` - procedural_metadata.json specification

### Documentation Improvements

- [ ] Add diagrams to scene management doc (architecture visualization)
- [ ] Create video tutorials for 3D viewer
- [ ] Add performance benchmarks to architecture docs
- [ ] Document all CSV output formats in detail

## Recent Updates

### 2025-10-18: Documentation Restructure & Procedural Thresholds
- ✅ **Restructured legacy assistant notes** - Reduced long-form session context and moved durable guidance into docs
- ✅ **Reorganized root directory** - Moved all docs to appropriate subdirectories
  - CODE_REVIEW_FINDINGS.md → development/CODE_REVIEW_FINDINGS.md (archived)
  - IMPROVEMENT_RECOMMENDATIONS.md → development/EDGE_CASE_PATTERNS.md
  - CHANGELOG.md → CHANGELOG.md
- ✅ **NEW:** [quick-reference/COMMANDS.md](quick-reference/COMMANDS.md) - All dev commands
- ✅ **NEW:** [quick-reference/TDD_EXAMPLES.md](quick-reference/TDD_EXAMPLES.md) - Test patterns
- ✅ **NEW:** [changelog/SESSION_HISTORY.md](changelog/SESSION_HISTORY.md) - Session updates archive
- ✅ Historical algorithm and hardcoded-constant notes were later archived under [archive/](archive/)
- ✅ Added a session handoff note, later superseded by [agentic task and handoff templates](agentic/TASK_TEMPLATE.md)
- ✅ Implemented adaptive motion state detection with CV check
- ✅ Cached derivatives for performance optimization
- ✅ Created procedural metadata export system
- **Estimated token savings:** ~15k tokens from autocompact buffer (33% reduction)

### 2025-10-17: Scene Management Documentation
- ✅ Added [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)
- ✅ Added [testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md)
- ✅ Created documentation structure README

### Previous Updates
- See git history for full changelog

## Getting Help

**Questions about:**
- **Setup/Installation** → [INSTALL.md](INSTALL.md)
- **Getting Started** → [../AGENTS.md](../AGENTS.md) and [onboarding/README.md](onboarding/README.md)
- **Agentic workflow** → [agentic/WORKFLOW.md](agentic/WORKFLOW.md)
- **FBX SDK** → [development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)
- **Testing** → [testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md)
- **Architecture** → [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)
- **Known Issues** → [audits/MODULE_ERROR_LOGIC_AUDIT.md](audits/MODULE_ERROR_LOGIC_AUDIT.md)

**Still stuck?** Check the [main README](../README.md) or review test examples in `../tests/`.
