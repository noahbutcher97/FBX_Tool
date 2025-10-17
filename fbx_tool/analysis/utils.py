"""
Utilities Module
Common helper functions for file I/O, bone hierarchy traversal, and data processing.
"""
import os
import csv
import numpy as np
import fbx


# ============================================================================
# File I/O Utilities
# ============================================================================

def ensure_output_dir(filepath):
    """
    Ensure the output directory exists for a given filepath.
    
    Args:
        filepath (str): Full path to output file.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def safe_overwrite(filepath):
    """
    Safely remove existing file to force overwrite.
    
    Args:
        filepath (str): Path to file to overwrite.
    
    Raises:
        RuntimeError: If file is locked or permission denied.
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            raise RuntimeError(
                f"Cannot overwrite {filepath} - file may be open in another program. "
                "Please close it and try again."
            )


def prepare_output_file(filepath):
    """
    Prepare output file by ensuring directory exists and clearing old file.
    
    Args:
        filepath (str): Path to output file.
    """
    ensure_output_dir(filepath)
    safe_overwrite(filepath)


def write_csv(filepath, header, rows):
    """
    Write CSV file with header and rows.
    
    Args:
        filepath (str): Output CSV path.
        header (list): Column headers.
        rows (list of lists): Data rows.
    """
    prepare_output_file(filepath)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ============================================================================
# FBX Scene Utilities
# ============================================================================

def get_animation_info(scene):
    """
    Extract animation timing information from FBX scene.
    Automatically selects the animation stack with actual animation data.

    Args:
        scene (fbx.FbxScene): FBX scene object.

    Returns:
        dict: Animation info with start, stop, frame_rate, frame_time, stack_name.
    """
    stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if stack_count == 0:
        raise RuntimeError("No animation stack found in scene.")

    # Try to find a stack with actual animation data
    # Priority: look for "mixamo.com" stack, then use longest duration stack
    selected_stack = None
    selected_index = 0
    max_duration = 0

    for i in range(stack_count):
        stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
        stack_name = stack.GetName()
        time_span = stack.GetLocalTimeSpan()
        duration = time_span.GetStop().GetSecondDouble() - time_span.GetStart().GetSecondDouble()

        # Prefer mixamo.com stack (contains actual Mixamo animation)
        if "mixamo" in stack_name.lower():
            selected_stack = stack
            selected_index = i
            break

        # Otherwise use the longest duration stack
        if duration > max_duration:
            max_duration = duration
            selected_stack = stack
            selected_index = i

    if not selected_stack:
        selected_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        selected_index = 0

    scene.SetCurrentAnimationStack(selected_stack)
    take_info = selected_stack.GetLocalTimeSpan()
    start = take_info.GetStart().GetSecondDouble()
    stop = take_info.GetStop().GetSecondDouble()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    frame_rate = fbx.FbxTime.GetFrameRate(time_mode)
    frame_time = 1.0 / frame_rate

    print(f"Using animation stack {selected_index}: '{selected_stack.GetName()}' ({stop - start:.2f}s)")

    return {
        "start": start,
        "stop": stop,
        "frame_rate": frame_rate,
        "frame_time": frame_time,
        "duration": stop - start,
        "stack_name": selected_stack.GetName(),
        "stack_index": selected_index
    }


def build_bone_hierarchy(scene):
    """
    Build a dictionary mapping child bones to their parents.
    
    Args:
        scene (fbx.FbxScene): FBX scene object.
    
    Returns:
        dict: {child_bone_name: parent_bone_name or None}
    """
    root = scene.GetRootNode()
    hierarchy = {}
    
    def traverse(node, parent=None):
        hierarchy[node.GetName()] = parent.GetName() if parent else None
        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i), node)
    
    traverse(root)
    return hierarchy


def collect_bone_names(scene):
    """
    Collect all bone names in the scene hierarchy.
    
    Args:
        scene (fbx.FbxScene): FBX scene object.
    
    Returns:
        list: List of bone names in hierarchy order.
    """
    root = scene.GetRootNode()
    bone_names = []
    
    def traverse(node):
        bone_names.append(node.GetName())
        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i))
    
    for i in range(root.GetChildCount()):
        traverse(root.GetChild(i))
    
    return bone_names


# ============================================================================
# Data Processing Utilities
# ============================================================================

def convert_numpy_to_native(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Any object (dict, list, numpy type, etc.)
    
    Returns:
        Converted object with native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_to_native(k): convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj


def format_float(value, precision=6):
    """
    Format float to string with specified precision.
    
    Args:
        value (float): Value to format.
        precision (int): Number of decimal places.
    
    Returns:
        str: Formatted string.
    """
    return f"{value:.{precision}f}"


# ============================================================================
# Chain Definition Utilities
# ============================================================================

def get_standard_chains():
    """
    Return standard bone chain definitions for humanoid rigs.

    DEPRECATED: Use detect_chains_from_hierarchy() for dynamic detection.

    Returns:
        dict: {chain_name: [bone_names]}
    """
    return {
        "LeftLeg": ["thigh_l", "calf_l", "foot_l", "ball_l"],
        "RightLeg": ["thigh_r", "calf_r", "foot_r", "ball_r"],
        "LeftArm": ["clavicle_l", "upperarm_l", "lowerarm_l", "hand_l"],
        "RightArm": ["clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"],
        "Spine": ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head"]
    }


def detect_chains_from_hierarchy(hierarchy, min_chain_length=3):
    """
    Automatically detect bone chains from skeleton hierarchy.
    Detects all linear chains (sequences of bones where each has only one child).

    Args:
        hierarchy (dict): {child: parent} bone hierarchy mapping
        min_chain_length (int): Minimum bones in a chain to be detected

    Returns:
        dict: {chain_name: [bone_names]} detected chains
    """
    # Build reverse hierarchy (parent -> children)
    children_map = {}
    for child, parent in hierarchy.items():
        if parent:
            children_map.setdefault(parent, []).append(child)

    # Find chain roots (bones that aren't the only child of their parent)
    chains = {}
    visited = set()

    def trace_chain(bone):
        """Trace a linear chain starting from bone."""
        chain = [bone]
        current = bone

        while True:
            kids = children_map.get(current, [])
            # Stop if we hit a branch (multiple children) or a leaf (no children)
            if len(kids) != 1:
                break
            current = kids[0]
            chain.append(current)

        return chain

    # Identify all chain starts
    for bone in hierarchy.keys():
        if bone in visited:
            continue

        # Skip if this bone is the only child (it's part of a chain, not a root)
        parent = hierarchy.get(bone)
        if parent:
            siblings = children_map.get(parent, [])
            if len(siblings) == 1:
                continue  # Part of an existing chain

        # Trace the chain from this root
        chain = trace_chain(bone)

        if len(chain) >= min_chain_length:
            # Mark all bones in this chain as visited
            visited.update(chain)

            # Generate a chain name based on bone names
            chain_name = _generate_chain_name(chain)
            chains[chain_name] = chain

    return chains


def _generate_chain_name(chain):
    """
    Generate a descriptive name for a bone chain.

    Args:
        chain (list): List of bone names in the chain

    Returns:
        str: Generated chain name
    """
    first_bone = chain[0].lower()
    last_bone = chain[-1].lower()

    # Common naming patterns
    if 'leg' in first_bone or 'thigh' in first_bone or 'upleg' in first_bone:
        if 'left' in first_bone or '_l' in first_bone:
            return "LeftLeg"
        elif 'right' in first_bone or '_r' in first_bone:
            return "RightLeg"
        return "Leg"

    elif 'arm' in first_bone or 'shoulder' in first_bone or 'clavicle' in first_bone:
        if 'left' in first_bone or '_l' in first_bone:
            return "LeftArm"
        elif 'right' in first_bone or '_r' in first_bone:
            return "RightArm"
        return "Arm"

    elif 'spine' in first_bone or 'hips' in first_bone or 'pelvis' in first_bone:
        return "Spine"

    elif 'neck' in first_bone or 'head' in first_bone:
        return "Neck"

    # Default: use first bone name
    return chain[0]


def validate_chain(chain, bone_names):
    """
    Validate that all bones in a chain exist in the skeleton.
    
    Args:
        chain (list): List of bone names in chain.
        bone_names (list): Available bone names in skeleton.
    
    Returns:
        list: Valid bones found in the skeleton.
    """
    return [bone for bone in chain if bone in bone_names]


# ============================================================================
# Math Utilities
# ============================================================================

def compute_velocity(values):
    """
    Compute velocity (first derivative) from position values.
    
    Args:
        values (array-like): Position values over time.
    
    Returns:
        np.ndarray: Velocity values.
    """
    return np.diff(values, prepend=values[0])


def compute_acceleration(velocity):
    """
    Compute acceleration (second derivative) from velocity values.
    
    Args:
        velocity (array-like): Velocity values over time.
    
    Returns:
        np.ndarray: Acceleration values.
    """
    return np.diff(velocity, prepend=velocity[0])


def detect_inversions(values):
    """
    Detect direction inversions (sign changes in derivative).
    
    Args:
        values (array-like): Time series values.
    
    Returns:
        np.ndarray: Boolean array where True indicates inversion.
    """
    vel = compute_velocity(values)
    return np.array([(vel[i-1] * vel[i]) < 0 for i in range(1, len(vel))])
