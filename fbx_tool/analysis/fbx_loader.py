"""
FBX Loader Module
Handles loading FBX files and extracting scene metadata safely.
"""
import fbx

def load_fbx(path):
    """
    Load an FBX file and return the scene object.
    
    Args:
        path (str): Full path to the FBX file.
    
    Returns:
        fbx.FbxScene: The loaded FBX scene.
    
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
    return scene

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
