"""
FBX Loader Module
Handles loading FBX files and extracting scene metadata safely.

Supports multi-stack FBX files by evaluating and ranking animation stacks
based on activity metrics (animated bones, keyframe density, duration).
"""
import fbx
import numpy as np

def load_fbx(path):
    """
    Load an FBX file and return the scene object and manager.

    IMPORTANT: Caller is responsible for destroying the manager when done:
        scene, manager = load_fbx(path)
        # ... use scene ...
        cleanup_fbx_scene(scene, manager)

    Args:
        path (str): Full path to the FBX file.

    Returns:
        tuple: (fbx.FbxScene, fbx.FbxManager) - The loaded scene and its manager

    Raises:
        FileNotFoundError: If the FBX file does not exist.
        RuntimeError: If FBX SDK fails to load or parse the file.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"FBX file not found: {path}")

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")

    if not importer.Initialize(path, -1, manager.GetIOSettings()):
        error = importer.GetStatus().GetErrorString()
        importer.Destroy()
        manager.Destroy()
        raise RuntimeError(f"FBX SDK failed to initialize: {error}")

    scene = fbx.FbxScene.Create(manager, "Scene")
    if not importer.Import(scene):
        error = importer.GetStatus().GetErrorString()
        importer.Destroy()
        manager.Destroy()
        raise RuntimeError(f"FBX SDK failed to import scene: {error}")

    importer.Destroy()

    # Return both scene AND manager - caller must destroy manager when done
    return scene, manager


def cleanup_fbx_scene(scene, manager):
    """
    Properly cleanup FBX scene and manager to prevent memory leaks.

    Args:
        scene: FBX scene object (can be None)
        manager: FBX manager object (can be None)
    """
    # Scene is destroyed automatically when manager is destroyed
    # Just need to destroy the manager
    if manager is not None:
        manager.Destroy()

def get_scene_metadata(scene):
    """
    Extract useful metadata from an FBX scene.
    
    Args:
        scene (fbx.FbxScene): The FBX scene object.
    
    Returns:
        dict: Metadata including frame rate, time range, bone count, etc.
    """
    anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if anim_stack_count == 0:
        return {"has_animation": False}
    
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    scene.SetCurrentAnimationStack(anim_stack)
    take_info = anim_stack.GetLocalTimeSpan()
    start = take_info.GetStart().GetSecondDouble()
    stop = take_info.GetStop().GetSecondDouble()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    frame_rate = fbx.FbxTime.GetFrameRate(time_mode)
    
    root = scene.GetRootNode()
    bone_count = count_bones(root)
    
    return {
        "has_animation": True,
        "start_time": start,
        "stop_time": stop,
        "duration": stop - start,
        "frame_rate": frame_rate,
        "bone_count": bone_count,
        "anim_stack_name": anim_stack.GetName()
    }

def count_bones(node):
    """Recursively count bones in the hierarchy."""
    count = 1
    for i in range(node.GetChildCount()):
        count += count_bones(node.GetChild(i))
    return count


def evaluate_stack_activity(scene, anim_stack):
    """
    Evaluate animation stack activity and quality metrics.

    Ranks stacks by:
    - Number of animated curves
    - Number of animated bones (weighted by importance)
    - Animation duration
    - Keyframe density

    Args:
        scene: FBX scene object
        anim_stack: Animation stack to evaluate

    Returns:
        dict: Activity metrics and quality score
    """
    # Core skeletal bones that indicate meaningful animation
    # Higher weight = more important for ranking
    CORE_BONE_PATTERNS = {
        'hips': 3.0, 'pelvis': 3.0, 'spine': 2.5, 'chest': 2.0,
        'neck': 2.0, 'head': 1.5,
        'shoulder': 2.0, 'arm': 1.5, 'elbow': 1.5, 'hand': 1.0,
        'leg': 2.0, 'thigh': 2.0, 'knee': 1.5, 'foot': 2.0, 'ankle': 2.0
    }

    # Get time span
    time_span = anim_stack.GetLocalTimeSpan()
    start = time_span.GetStart().GetSecondDouble()
    stop = time_span.GetStop().GetSecondDouble()
    duration = stop - start

    if duration <= 0:
        return {
            'animated_curves': 0,
            'animated_bones': 0,
            'core_bone_weight': 0.0,
            'duration': 0.0,
            'keyframe_count': 0,
            'keyframe_density': 0.0,
            'activity_score': 0.0
        }

    # Set as current stack to evaluate curves
    scene.SetCurrentAnimationStack(anim_stack)

    # Count animated curves and bones
    animated_curves = 0
    animated_bones_set = set()
    total_keyframes = 0
    core_bone_weight = 0.0

    def traverse_and_count(node):
        nonlocal animated_curves, animated_bones_set, total_keyframes, core_bone_weight

        # Check if this node has animation on its properties
        # Most skeletal animation lives in transform properties
        has_animation = False

        # Get the animation layer (first layer)
        # FbxAnimStack contains FbxAnimLayer objects
        layer_count = anim_stack.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId))
        if layer_count == 0:
            # No animation layer, recurse to children
            for i in range(node.GetChildCount()):
                traverse_and_count(node.GetChild(i))
            return

        anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)

        # Check translation curves
        curve_node = node.LclTranslation.GetCurveNode(anim_layer)
        if curve_node:
            has_animation = True
            # Count curves for X, Y, Z
            for channel in range(curve_node.GetChannelsCount()):
                curve = curve_node.GetCurve(channel)
                if curve:
                    animated_curves += 1
                    total_keyframes += curve.KeyGetCount()

        # Check rotation curves
        curve_node = node.LclRotation.GetCurveNode(anim_layer)
        if curve_node:
            has_animation = True
            for channel in range(curve_node.GetChannelsCount()):
                curve = curve_node.GetCurve(channel)
                if curve:
                    animated_curves += 1
                    total_keyframes += curve.KeyGetCount()

        # Check scaling curves
        curve_node = node.LclScaling.GetCurveNode(anim_layer)
        if curve_node:
            has_animation = True
            for channel in range(curve_node.GetChannelsCount()):
                curve = curve_node.GetCurve(channel)
                if curve:
                    animated_curves += 1
                    total_keyframes += curve.KeyGetCount()

        if has_animation:
            bone_name = node.GetName().lower()
            animated_bones_set.add(node.GetName())

            # Check if this is a core bone (higher weight)
            for pattern, weight in CORE_BONE_PATTERNS.items():
                if pattern in bone_name:
                    core_bone_weight += weight
                    break  # Only count highest match

        # Recurse to children
        for i in range(node.GetChildCount()):
            traverse_and_count(node.GetChild(i))

    # Traverse scene hierarchy
    root = scene.GetRootNode()
    traverse_and_count(root)

    # Compute metrics
    animated_bone_count = len(animated_bones_set)
    keyframe_density = total_keyframes / duration if duration > 0 else 0

    # Activity score (weighted combination)
    # Higher score = more meaningful animation
    # Weights are empirically tuned
    activity_score = (
        (animated_curves * 0.1) +           # More curves = more animation
        (animated_bone_count * 1.0) +       # More bones = fuller animation
        (core_bone_weight * 2.0) +          # Core bones = high value
        (duration * 0.5) +                  # Longer = more content
        (keyframe_density * 0.01)           # Denser = higher quality
    )

    return {
        'animated_curves': animated_curves,
        'animated_bones': animated_bone_count,
        'core_bone_weight': core_bone_weight,
        'duration': duration,
        'keyframe_count': total_keyframes,
        'keyframe_density': keyframe_density,
        'activity_score': activity_score
    }


def rank_animation_stacks(scene):
    """
    Evaluate and rank all animation stacks in the scene.

    Returns stacks sorted by activity score (descending).

    Args:
        scene: FBX scene object

    Returns:
        list: Ranked stacks as dicts with {
            'stack': FbxAnimStack object,
            'name': str,
            'index': int,
            'rank': str (primary/secondary/tertiary),
            'metrics': activity metrics dict
        }
    """
    anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))

    if anim_stack_count == 0:
        return []

    # Evaluate all stacks
    stack_evaluations = []
    for i in range(anim_stack_count):
        stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
        metrics = evaluate_stack_activity(scene, stack)

        stack_evaluations.append({
            'stack': stack,
            'name': stack.GetName(),
            'index': i,
            'metrics': metrics
        })

    # Sort by activity score (descending)
    stack_evaluations.sort(key=lambda x: x['metrics']['activity_score'], reverse=True)

    # Assign ranks
    rank_labels = ['primary', 'secondary', 'tertiary']
    for i, evaluation in enumerate(stack_evaluations):
        if i < len(rank_labels):
            evaluation['rank'] = rank_labels[i]
        else:
            evaluation['rank'] = f'rank_{i+1}'

    return stack_evaluations


def get_scene_metadata(scene, prefer_stack_name=None):
    """
    Extract useful metadata from an FBX scene with intelligent stack selection.

    Evaluates all animation stacks and selects the most active/meaningful one,
    or allows manual selection by name.

    Args:
        scene (fbx.FbxScene): The FBX scene object
        prefer_stack_name (str, optional): Preferred stack name (e.g., "Mixamo")

    Returns:
        dict: Metadata including:
            - has_animation: bool
            - start_time, stop_time, duration, frame_rate: timing info
            - bone_count: skeleton size
            - anim_stack_name: selected stack name
            - anim_stack_rank: stack ranking (primary/secondary/tertiary)
            - anim_stack_count: total number of stacks
            - all_stacks: list of all ranked stacks with metrics
    """
    # Rank all animation stacks
    ranked_stacks = rank_animation_stacks(scene)

    if not ranked_stacks:
        return {
            "has_animation": False,
            "anim_stack_count": 0
        }

    # Select stack (prefer by name if specified, otherwise use primary)
    selected_stack_info = None

    if prefer_stack_name:
        # Try to find stack by name
        for stack_info in ranked_stacks:
            if prefer_stack_name.lower() in stack_info['name'].lower():
                selected_stack_info = stack_info
                break

    # Fall back to primary (highest ranked)
    if not selected_stack_info:
        selected_stack_info = ranked_stacks[0]

    # Set as current stack
    anim_stack = selected_stack_info['stack']
    scene.SetCurrentAnimationStack(anim_stack)

    # Extract timing information
    take_info = anim_stack.GetLocalTimeSpan()
    start = take_info.GetStart().GetSecondDouble()
    stop = take_info.GetStop().GetSecondDouble()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    frame_rate = fbx.FbxTime.GetFrameRate(time_mode)

    # Count bones
    root = scene.GetRootNode()
    bone_count = count_bones(root)

    # Compile lightweight stack summary (without FBX objects for serialization)
    stack_summary = []
    for stack_info in ranked_stacks:
        stack_summary.append({
            'name': stack_info['name'],
            'rank': stack_info['rank'],
            'index': stack_info['index'],
            'activity_score': stack_info['metrics']['activity_score'],
            'animated_bones': stack_info['metrics']['animated_bones'],
            'duration': stack_info['metrics']['duration']
        })

    return {
        "has_animation": True,
        "start_time": start,
        "stop_time": stop,
        "duration": stop - start,
        "frame_rate": frame_rate,
        "bone_count": bone_count,
        "anim_stack_name": selected_stack_info['name'],
        "anim_stack_rank": selected_stack_info['rank'],
        "anim_stack_index": selected_stack_info['index'],
        "anim_stack_count": len(ranked_stacks),
        "all_stacks": stack_summary  # Summary of all available stacks
    }
