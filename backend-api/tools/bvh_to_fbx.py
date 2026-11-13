"""OPTIONAL: convert a generated .bvh into .fbx using Blender (free, https://www.blender.org).

Python cannot write FBX on its own (the format is proprietary), so we let Blender do it:

    blender --background --python tools/bvh_to_fbx.py -- outputs/<id>.bvh outputs/<id>.fbx

Check the result's orientation/scale in your game engine; adjust axis_* / global_scale if needed.
"""
import sys

import bpy  # only available inside Blender's bundled Python

args = sys.argv[sys.argv.index("--") + 1:]
if len(args) != 2:
    raise SystemExit("usage: blender --background --python bvh_to_fbx.py -- input.bvh output.fbx")
src, dst = args

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(
    filepath=src,
    axis_forward="-Z",
    axis_up="Y",
    global_scale=0.01,          # our BVH is in centimetres
    rotate_mode="NATIVE",
    update_scene_fps=True,
    update_scene_duration=True,
)
bpy.ops.export_scene.fbx(filepath=dst, bake_anim=True, add_leaf_bones=False, object_types={"ARMATURE"})
print("Wrote", dst)
