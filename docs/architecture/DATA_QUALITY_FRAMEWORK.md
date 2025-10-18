# Data Quality & Confidence Scoring Framework

## Overview

A comprehensive framework for assessing animation data quality and providing confidence scores for all analysis results. This is critical for effective AI communication and fallback analysis strategies.

## Problem Statement

**Current Issues:**
1. Analysis fails completely when root bone detection fails (no fallback)
2. No confidence scores on results (can't tell good data from questionable data)
3. AI/LLM integration would benefit from quality metadata to guide interpretation
4. No way to communicate "I detected this, but I'm only 60% confident"

**Use Cases:**
- Root bone detection fails → Fallback to center of mass calculation
- Poor root motion data → Analyze whole-skeleton movement instead
- Noisy velocity data → Flag low confidence in motion classification
- AI queries → "What is the animation doing?" with confidence context

---

## Architecture

### 1. Data Quality Scoring System

#### Quality Dimensions

Each analysis result should have quality scores across these dimensions:

```python
{
    "data_completeness": 0.0-1.0,    # How much required data was available
    "data_consistency": 0.0-1.0,      # Internal consistency of data
    "detection_confidence": 0.0-1.0,  # Confidence in detected features
    "analysis_reliability": 0.0-1.0,  # Overall reliability of analysis
    "overall_quality": 0.0-1.0        # Weighted average of above
}
```

#### Quality Levels

```python
QUALITY_EXCELLENT = 0.9-1.0   # High confidence, complete data
QUALITY_GOOD = 0.7-0.9        # Reliable results, minor gaps
QUALITY_FAIR = 0.5-0.7        # Usable but questionable
QUALITY_POOR = 0.3-0.5        # Low confidence, use with caution
QUALITY_UNRELIABLE = 0.0-0.3  # Do not trust, fallback recommended
```

---

### 2. Fallback Motion Detection

#### Hierarchy of Motion Detection Strategies

When root bone detection fails or produces poor quality data, use these fallbacks in order:

##### Strategy 1: Root Bone (Primary)
- **Detection:** Pattern matching + skeleton node verification
- **Quality Indicators:**
  - Found skeleton node matching patterns ✓
  - Bone has animation curves ✓
  - Movement exceeds threshold ✓
- **Confidence:** 0.9-1.0

##### Strategy 2: Next Lowest Bone (First Fallback)
- **Detection:** Find first child of root, or spine base
- **Quality Indicators:**
  - Bone diverges from initial position
  - Rotation changes over time
  - Forces/velocities are non-zero
- **Confidence:** 0.6-0.8
- **Adjustments:**
  - Subtract parent bone motion (if available)
  - Account for skeletal offset from true root

##### Strategy 3: Center of Mass (Second Fallback)
- **Calculation:** Weighted average of all skeleton bone positions
- **Quality Indicators:**
  - All bones contributing to calculation
  - Consistent COM movement
  - No erratic jumps
- **Confidence:** 0.5-0.7
- **Advantages:**
  - Always works (no root bone required)
  - Robust to poor hierarchy structure

##### Strategy 4: Whole-Skeleton Motion Analysis (Final Fallback)
- **Approach:** Analyze aggregate skeleton movement
- **Metrics:**
  - Bounding box displacement
  - Average bone velocity across skeleton
  - Variance in bone positions over time
- **Quality Indicators:**
  - Sufficient skeleton bones detected
  - Movement patterns are coherent
- **Confidence:** 0.3-0.6
- **Use Cases:**
  - No clear root bone
  - Complex rigs (vehicles, creatures)
  - Hierarchical motion (riding, carrying)

---

### 3. Implementation Plan

#### Phase 1: Quality Scoring Infrastructure

**File:** `fbx_tool/analysis/quality_scoring.py`

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class QualityScore:
    """Quality assessment for analysis results."""
    data_completeness: float      # 0.0-1.0
    data_consistency: float       # 0.0-1.0
    detection_confidence: float   # 0.0-1.0
    analysis_reliability: float   # 0.0-1.0
    overall_quality: float        # Weighted average

    quality_level: str           # EXCELLENT/GOOD/FAIR/POOR/UNRELIABLE
    warnings: list[str]          # Human-readable warnings
    fallback_used: Optional[str] # If fallback strategy was used

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        pass

    def get_narrative(self) -> str:
        """Human-readable quality summary."""
        pass

def compute_quality_score(
    data_completeness: float,
    data_consistency: float,
    detection_confidence: float,
    analysis_reliability: float,
    weights: Optional[Dict[str, float]] = None
) -> QualityScore:
    """Compute overall quality score with configurable weights."""
    pass

def assess_root_bone_quality(root_bone, trajectory_data) -> QualityScore:
    """Assess quality of root bone detection and trajectory."""
    pass

def assess_motion_data_quality(positions, velocities, accelerations) -> QualityScore:
    """Assess quality of motion derivative calculations."""
    pass
```

#### Phase 2: Fallback Root Motion Detection

**File:** `fbx_tool/analysis/utils.py` (extend existing)

```python
def detect_root_with_fallback(scene):
    """
    Detect root bone with fallback strategies.

    Returns:
        tuple: (bone_node, detection_method, confidence_score)
    """
    # Try primary method
    root_bone = _detect_root_bone(scene)
    if root_bone and _validate_root_quality(root_bone, scene):
        return (root_bone, "root_bone_pattern", 0.95)

    # Fallback 1: Next lowest bone
    next_bone = _find_next_lowest_bone(scene)
    if next_bone:
        confidence = _assess_next_bone_confidence(next_bone, scene)
        if confidence > 0.6:
            return (next_bone, "next_lowest_bone", confidence)

    # Fallback 2: Center of mass
    com_node = _create_virtual_com_node(scene)
    if com_node:
        return (com_node, "center_of_mass", 0.6)

    # Fallback 3: Whole skeleton analysis
    return (None, "whole_skeleton", 0.4)

def _find_next_lowest_bone(scene):
    """Find the first skeleton bone in hierarchy."""
    pass

def _assess_next_bone_confidence(bone, scene):
    """Assess confidence in using this bone as root proxy."""
    # Check movement magnitude
    # Check animation curve presence
    # Check hierarchy position
    pass

def _create_virtual_com_node(scene):
    """Create virtual node representing center of mass."""
    # Calculate weighted COM from all bones
    # Create trajectory from COM over time
    pass
```

#### Phase 3: Whole-Skeleton Motion Analysis

**File:** `fbx_tool/analysis/skeleton_motion_analysis.py` (new)

```python
def analyze_skeleton_motion(scene, output_dir="output/"):
    """
    Fallback analysis when root bone detection fails.

    Analyzes aggregate skeleton movement using:
    - Bounding box displacement
    - Average bone velocities
    - Skeleton center of mass
    - Coherence metrics
    """
    pass

def compute_skeleton_bounding_box(scene, frame):
    """Compute 3D bounding box for all bones at given frame."""
    pass

def compute_skeleton_center_of_mass(scene, frame, bone_weights=None):
    """Compute weighted center of mass for skeleton."""
    pass

def assess_motion_coherence(bone_velocities):
    """
    Measure how coherent bone movements are.

    High coherence = bones moving together (walking, running)
    Low coherence = independent motion (dancing, gesturing)
    """
    pass
```

#### Phase 4: Integrate Quality Scores Into All Analyses

**Update existing modules to return quality scores:**

```python
# Root motion analysis
def analyze_root_motion(scene, output_dir="output/"):
    # ... existing analysis ...

    # Add quality assessment
    quality = assess_root_motion_quality(
        root_bone=root_bone,
        detection_method=detection_method,
        trajectory_data=trajectory_data,
        total_distance=total_distance
    )

    return {
        "root_bone_name": root_bone_name,
        "total_distance": total_distance,
        # ... existing fields ...
        "quality": quality.to_dict(),
        "detection_method": detection_method,
        "confidence": quality.overall_quality
    }

# Gait analysis
def analyze_gait(scene, output_dir="output/"):
    # ... existing analysis ...

    quality = assess_gait_quality(
        stride_segments=stride_segments,
        foot_contact_confidence=foot_contact_confidence
    )

    return stride_segments, gait_summary, quality.to_dict()
```

---

### 4. Quality Indicators by Analysis Type

#### Root Motion Quality Indicators

```python
def assess_root_motion_quality(root_bone, detection_method, trajectory_data, total_distance):
    """
    Quality indicators:
    - Detection method used (pattern=1.0, fallback=0.6, COM=0.5)
    - Movement magnitude (distance > 0.1 = good)
    - Trajectory smoothness (low jerk = good)
    - Animation curve presence
    """

    completeness = 1.0 if root_bone else 0.5

    # Detection confidence based on method
    detection_confidence = {
        "root_bone_pattern": 0.95,
        "next_lowest_bone": 0.7,
        "center_of_mass": 0.6,
        "whole_skeleton": 0.4
    }.get(detection_method, 0.3)

    # Consistency: check for smooth trajectory
    jerks = compute_jerk_magnitudes(trajectory_data)
    consistency = 1.0 - min(np.mean(jerks) / 1000.0, 1.0)

    # Reliability: based on movement magnitude
    reliability = min(total_distance / 10.0, 1.0)  # Expect at least 10 units

    return compute_quality_score(
        data_completeness=completeness,
        data_consistency=consistency,
        detection_confidence=detection_confidence,
        analysis_reliability=reliability
    )
```

#### Gait Analysis Quality Indicators

```python
def assess_gait_quality(stride_segments, foot_contact_confidence):
    """
    Quality indicators:
    - Feet detected successfully
    - Contact events have high confidence
    - Stride segments are regular
    - Temporal coherence
    """
    pass
```

#### Velocity Analysis Quality Indicators

```python
def assess_velocity_quality(velocities, accelerations, jerks):
    """
    Quality indicators:
    - Low noise in derivatives
    - Smooth acceleration curves
    - Few jerk spikes
    - Consistent frame rate
    """
    pass
```

---

### 5. AI Integration Layer

#### LLM-Friendly Quality Metadata

**File:** `fbx_tool/analysis/ai_metadata.py` (new)

```python
def generate_ai_metadata(analysis_results):
    """
    Generate LLM-friendly metadata for analysis results.

    Returns structured data optimized for AI interpretation:
    - Natural language quality summary
    - Confidence-weighted facts
    - Caveats and warnings
    - Recommended interpretations
    """

    metadata = {
        "summary": generate_natural_language_summary(analysis_results),
        "confidence_level": analysis_results["quality"]["quality_level"],
        "high_confidence_facts": extract_high_confidence_facts(analysis_results),
        "low_confidence_facts": extract_low_confidence_facts(analysis_results),
        "warnings": analysis_results["quality"]["warnings"],
        "fallbacks_used": list_fallbacks_used(analysis_results),
        "recommended_interpretation": generate_interpretation_guide(analysis_results)
    }

    return metadata

def generate_natural_language_summary(analysis_results):
    """
    Generate natural language summary of animation.

    Example:
    "This animation shows a character walking forward with HIGH confidence (0.92).
     Root motion was detected using the standard hips bone. The character travels
     approximately 15.3 units over 2.5 seconds at an average speed of 6.1 units/sec."
    """
    pass

def extract_high_confidence_facts(analysis_results, threshold=0.8):
    """
    Extract only facts with confidence above threshold.

    Example:
    [
        {"fact": "Character is walking", "confidence": 0.92},
        {"fact": "Movement is forward", "confidence": 0.89},
        {"fact": "Gait is bipedal", "confidence": 0.95}
    ]
    """
    pass
```

---

### 6. CSV Output Format with Quality Scores

#### Updated Root Motion Summary CSV

```csv
root_bone,detection_method,total_distance,displacement,mean_velocity,dominant_direction,quality_level,overall_confidence,data_completeness,detection_confidence,warnings
mixamorig:Hips,root_bone_pattern,15.32,14.87,6.13,forward,EXCELLENT,0.92,1.0,0.95,
RootNode,next_lowest_bone,2.14,2.01,0.86,forward,FAIR,0.65,0.8,0.7,"Using fallback: next lowest bone"
VirtualCOM,center_of_mass,8.45,7.92,3.38,forward,FAIR,0.58,0.6,0.6,"No root bone found; using center of mass"
```

#### New Quality Assessment CSV

**File:** `quality_assessment.csv`

```csv
analysis_type,quality_level,overall_confidence,data_completeness,data_consistency,detection_confidence,analysis_reliability,warnings,fallbacks_used
root_motion,EXCELLENT,0.92,1.0,0.95,0.95,0.85,"",
gait,GOOD,0.78,0.9,0.85,0.7,0.75,"Foot sliding detected on left foot",
velocity,FAIR,0.62,0.8,0.6,0.8,0.5,"High jerk spikes detected","derivative_smoothing"
```

---

## Implementation Priority

### High Priority (Immediate Value)
1. ✅ **Fix root bone detection** (COMPLETED)
2. **Add basic quality scoring to root motion** (Next)
3. **Implement next-lowest-bone fallback** (Quick win)

### Medium Priority (Near-term)
4. **Add quality scores to all existing analyses**
5. **Implement center-of-mass fallback**
6. **Create quality assessment CSV output**

### Lower Priority (Future Enhancement)
7. **Whole-skeleton motion analysis**
8. **AI metadata generation layer**
9. **Adaptive thresholds based on quality scores**

---

## Benefits

### For Users
- Analyses don't fail completely when root bone missing
- Clear confidence indicators on all results
- Warnings when data quality is poor

### For AI Integration
- LLMs can weight responses by confidence
- Natural language quality summaries
- Structured metadata for intelligent interpretation

### For Developers
- Reusable quality scoring framework
- Fallback strategies prevent analysis failures
- Quality-guided adaptive thresholds

---

## Testing Strategy

### Unit Tests
- Quality score calculation
- Fallback detection logic
- Confidence assessment functions

### Integration Tests
- Root motion with fallbacks
- Quality score generation for all analyses
- AI metadata generation

### Real-world Tests
- Mixamo animations (standard rigs)
- Custom rigs (non-standard hierarchies)
- Broken/incomplete animations
- Non-humanoid skeletons (creatures, vehicles)

---

## Related Documentation

- `docs/development/FBX_SDK_FIXES.md` - Root bone detection patterns
- `docs/architecture/SCENE_MANAGEMENT.md` - Scene lifecycle
- `docs/testing/MOCK_SETUP_PATTERNS.md` - Testing quality scoring

---

## Future Enhancements

### Adaptive Quality Thresholds
Use quality scores to adjust analysis thresholds:
```python
if quality.overall_quality > 0.8:
    # High quality data - use strict thresholds
    VELOCITY_THRESHOLD = 0.1
else:
    # Lower quality - use relaxed thresholds
    VELOCITY_THRESHOLD = 0.5
```

### Quality-Driven Recommendations
```python
if quality.overall_quality < 0.5:
    recommendations.append("Consider using center-of-mass analysis instead")
if "foot_sliding" in warnings:
    recommendations.append("Ground plane detection may be unreliable")
```

### Machine Learning Integration
- Train quality classifiers on labeled data
- Predict analysis reliability from scene metadata
- Auto-detect optimal fallback strategy per file
