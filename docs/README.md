# FBX Tool Documentation

## Documentation Structure

This directory contains all project documentation organized by category.

### 📖 Getting Started

**New to the project? Start here:**

1. **[INSTALL.md](INSTALL.md)** - Environment setup, Python 3.10 installation, FBX SDK setup
2. **[onboarding/README.md](onboarding/README.md)** - Project overview, TDD workflow, design principles
3. **[../README.md](../README.md)** - User-facing features and quick start guide

### 🏗️ Architecture

**Understanding the system design:**

- **[architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)** - Reference-counted scene manager architecture
  - Problem statement and motivation
  - Component overview (FBXSceneManager, FBXSceneReference)
  - Integration patterns (GUI, Visualizer, Analysis Worker)
  - Smart caching strategy
  - Workflow examples and performance impact

- **[architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md](architecture/PROCEDURAL_DESIGN_PHILOSOPHY.md)** - **NEW (2025-10-18)** Procedural design principles
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

- **[quick-reference/COMMANDS.md](quick-reference/COMMANDS.md)** - **NEW** Development commands
  - Environment setup
  - Testing commands (pytest options, markers)
  - Code quality (black, isort, flake8, mypy)
  - Running the application
  - Building executables
  - Coverage requirements

- **[quick-reference/TDD_EXAMPLES.md](quick-reference/TDD_EXAMPLES.md)** - **NEW** Test-driven development patterns
  - TDD workflow steps
  - Good vs bad test examples
  - Test organization patterns
  - Using fixtures from conftest.py
  - Common test patterns (parametrized, confidence scores, adaptive thresholds)
  - Reference: test_gait_analysis.py breakdown

### 🔧 Development

**Development guidelines and issue tracking:**

- **[development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)** - FBX SDK API patterns (MUST READ!)
  - Correct vs incorrect API usage
  - Time span access patterns
  - Animation curve access
  - Common pitfalls

- **[development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md)** - Current incomplete modules
  - Modules requiring TDD implementation
  - Active issues and root causes
  - Fix requirements

- **[development/HARDCODED_CONSTANTS_AUDIT.md](development/HARDCODED_CONSTANTS_AUDIT.md)** - **NEW (2025-10-18)** Proceduralization tracker
  - Comprehensive audit of all hardcoded constants
  - Status tracking (Fixed/Partial/Not Fixed)
  - Priority levels (P0/P1/P2)
  - Impact assessment and fix order

- **[development/NEXT_SESSION_TODO.md](development/NEXT_SESSION_TODO.md)** - **NEW (2025-10-18)** Session handoff
  - Urgent verification tasks
  - Priority-ordered todo list
  - Known issues and fixes in progress
  - Test protocol and quick commands

### 📊 Project Status

**Current state and improvement areas:**

- **[development/ALGORITHM_ISSUES.md](development/ALGORITHM_ISSUES.md)** - **Detailed** algorithm correctness issues
  - Complete code review findings for velocity_analysis.py, gait_analysis.py, chain_analysis.py
  - Line-specific issues with fixes
  - Priority categorization (MUST FIX, SHOULD FIX, NICE TO HAVE)

- **[development/CODE_REVIEW_FINDINGS.md](development/CODE_REVIEW_FINDINGS.md)** - Original code review (archived)

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
1. Read [INSTALL.md](INSTALL.md)
2. Read [onboarding/README.md](onboarding/README.md)
3. Read [development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)
4. Review test examples in `../tests/unit/test_gait_analysis.py`

#### For Maintainers
1. Check [development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md) for active issues
2. Review [development/ALGORITHM_ISSUES.md](development/ALGORITHM_ISSUES.md) for priority fixes
3. Consult [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md) for system design
4. Use [testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md) for test writing

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
- [development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md) - Known issues
- [development/ALGORITHM_ISSUES.md](development/ALGORITHM_ISSUES.md) - Algorithm correctness
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
python -m fbx_tool path/to/animation.fbx
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
- [x] ~~`architecture/CACHING_STRATEGY.md`~~ - ✅ Covered in SCENE_MANAGEMENT.md & CLAUDE.md Session 2025-10-18
- [x] ~~`development/ADAPTIVE_THRESHOLDS.md`~~ - ✅ See HARDCODED_CONSTANTS_AUDIT.md & CLAUDE.md
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
- ✅ **Restructured CLAUDE.md** - Reduced from ~9.9k tokens to ~3k tokens (70% reduction)
- ✅ **Reorganized root directory** - Moved all docs to appropriate subdirectories
  - CODE_REVIEW_FINDINGS.md → development/CODE_REVIEW_FINDINGS.md (archived)
  - IMPROVEMENT_RECOMMENDATIONS.md → development/EDGE_CASE_PATTERNS.md
  - CHANGELOG.md → CHANGELOG.md
- ✅ **NEW:** [quick-reference/COMMANDS.md](quick-reference/COMMANDS.md) - All dev commands
- ✅ **NEW:** [quick-reference/TDD_EXAMPLES.md](quick-reference/TDD_EXAMPLES.md) - Test patterns
- ✅ **NEW:** [changelog/SESSION_HISTORY.md](changelog/SESSION_HISTORY.md) - Session updates archive
- ✅ **NEW:** [development/ALGORITHM_ISSUES.md](development/ALGORITHM_ISSUES.md) - Detailed algorithm fixes
- ✅ Added [development/HARDCODED_CONSTANTS_AUDIT.md](development/HARDCODED_CONSTANTS_AUDIT.md)
- ✅ Added [development/NEXT_SESSION_TODO.md](development/NEXT_SESSION_TODO.md)
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
- **Getting Started** → [onboarding/README.md](onboarding/README.md)
- **FBX SDK** → [development/FBX_SDK_FIXES.md](development/FBX_SDK_FIXES.md)
- **Testing** → [testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md)
- **Architecture** → [architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)
- **Known Issues** → [development/INCOMPLETE_MODULES.md](development/INCOMPLETE_MODULES.md)

**Still stuck?** Check the [main README](../README.md) or review test examples in `../tests/`.
