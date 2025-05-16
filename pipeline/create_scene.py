# -*- coding: utf-8 -*-
import bpy
import math
import os
import json
import sys
from mathutils import Vector


def clear_scene():
    """Clear all objects and materials from the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    # If no objects are selected, delete will raise an error, so add a check
    if bpy.context.selected_objects:
        bpy.ops.object.delete()


    # Clean up unused data blocks more safely
    mats = list(bpy.data.materials)
    for block in mats: bpy.data.materials.remove(block)
    meshes = list(bpy.data.meshes)
    for block in meshes: bpy.data.meshes.remove(block)
    lights = list(bpy.data.lights)
    for block in lights: bpy.data.lights.remove(block)
    cameras = list(bpy.data.cameras)
    for block in cameras: bpy.data.cameras.remove(block)
    worlds = list(bpy.data.worlds)
    for block in worlds: bpy.data.worlds.remove(block)






def create_camera(camera_data):
    """Create a camera based on the configuration."""
   
    # Check for necessary keys
    if not all(k in camera_data for k in ['name', 'location', 'rotation_euler']):
        # Error message removed, return None is sufficient
        return None


    camera = bpy.data.cameras.new(name=camera_data["name"])
    camera_ob = bpy.data.objects.new(name=camera_data["name"], object_data=camera)


    # Set location and rotation (convert to radians)
    camera_ob.location = Vector(camera_data["location"])
    camera_ob.rotation_euler = [math.radians(r) for r in camera_data["rotation_euler"]]


    # Set camera parameters
    camera.lens = camera_data.get("lens", 50)
    camera.sensor_width = camera_data.get("sensor_width", 36)
    camera.sensor_height = camera_data.get("sensor_height", 36)
    camera.clip_start = camera_data.get("clip_start", 0.1)
    camera.clip_end = camera_data.get("clip_end", 100)


    # Add camera to the scene
    bpy.context.collection.objects.link(camera_ob)
    bpy.context.scene.camera = camera_ob
    return camera_ob




def determine_shape(name):
    """Infer shape type from the object name."""
    name_lower = name.lower()
    if 'cube' in name_lower:
        return 'CUBE'
    elif 'sphere' in name_lower:
        return 'SPHERE'
    elif 'cylinder' in name_lower:
        return 'CYLINDER'
    else:
        # Warning removed, default to CUBE
        return 'CUBE'


def create_object(obj_data):
    """Create a single object (mimicking prompt2.md logic)."""
    # Check for necessary keys
    required_keys = ['name', 'location', 'rotation_euler', 'material', 'size_params']
    if not all(key in obj_data for key in required_keys):
        # Error message removed
        return None


    shape = determine_shape(obj_data["name"])
    location = Vector(obj_data["location"])
    rotation_rad = [math.radians(r) for r in obj_data["rotation_euler"]]
    size_params = obj_data["size_params"]


 
    # Create object directly using size_params, adding defaults for missing keys
    if shape == 'CUBE':
        size = size_params.get('size', 1.0) # Default size = 1.0
        bpy.ops.mesh.primitive_cube_add(
            size=size, location=location, rotation=rotation_rad,
            enter_editmode=False, align='WORLD'
        )
    elif shape == 'SPHERE':
        radius = size_params.get('radius', 0.5) # Default radius = 0.5
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, location=location, rotation=rotation_rad,
            enter_editmode=False, align='WORLD'
        )
    elif shape == 'CYLINDER':
        radius = size_params.get('radius', 0.5) # Default radius = 0.5
        depth = size_params.get('depth', 1.0)   # Default depth = 1.0
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=depth, location=location, rotation=rotation_rad,
            enter_editmode=False, align='WORLD'
        )
    else:
        # Error message removed
        return None
    

    obj = bpy.context.active_object
    if not obj:
        # Error message removed
        return None


    obj.name = obj_data["name"]


    # Create and assign material
    material = create_material(obj_data["material"])
    if material: # Ensure material creation was successful
        assign_material(obj, material)


    return obj


def create_material(mat_data):
    """Create a material."""
    # Check for necessary keys
    if not all(k in mat_data for k in ['name', 'base_color', 'metallic', 'roughness']):
        # Error message removed
        return None


    mat_name = mat_data["name"]
    if mat_name in bpy.data.materials:
        # Reuse existing material
        return bpy.data.materials[mat_name]


    # Create new material
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links


    # Clear potentially existing default nodes
    while nodes: nodes.remove(nodes[0])


    # Create nodes
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)


    # Set material properties (increase robustness)
    color = mat_data["base_color"]
    if isinstance(color, (list, tuple)) and len(color) == 3:
        color = list(color) + [1.0]
    elif not (isinstance(color, (list, tuple)) and len(color) == 4):
        # Warning removed, use default black
        color = [0.0, 0.0, 0.0, 1.0]
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Metallic'].default_value = float(mat_data["metallic"])
    principled.inputs['Roughness'].default_value = float(mat_data["roughness"])


    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)


    # Link nodes
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat








def assign_material(obj, material):
    """Assign a material to an object."""
    if not obj or not obj.data:
        # Warning removed
        return
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)




def setup_lighting(lighting_data):
    """Set up lighting based on configuration."""


    # Create Sun light
    sun_rotation_rad = [math.radians(a) for a in lighting_data.get("sun_rotation_euler_degrees", [60, 0, 30])]
    sun_energy = lighting_data.get("sun_energy", 3.0)
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.rotation_euler = sun_rotation_rad
    sun.data.energy = sun_energy


    # Set up environment light (world background)
    env_color = lighting_data.get("environment_color", [0.8, 0.8, 0.8, 1.0])
    env_strength = lighting_data.get("environment_strength", 1.2)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True


    # Safely get/create Background and Output nodes and link them
    bg_node = None
    output_node = None
    if world.node_tree:
        output_node = world.node_tree.nodes.get('World Output')
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                bg_node = node
                break
    else:
        # Should not happen if use_nodes is True, but handle defensively
        world.node_tree = bpy.data.node_groups.new(name="World Nodes", type='ShaderNodeTree')


    if not output_node: output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    if not bg_node: bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')


    bg_output = bg_node.outputs.get('Background')
    surf_input = output_node.inputs.get('Surface')
    if bg_output and surf_input:
        is_linked = any(link.from_node == bg_node and link.to_node == output_node for link in world.node_tree.links)
        if not is_linked: world.node_tree.links.new(bg_output, surf_input)


    bg_node.inputs['Color'].default_value = env_color
    bg_node.inputs['Strength'].default_value = env_strength




def main():
    """Main function."""
    json_path = None
    output_dir = None
    # Parse command line arguments
    try:
        if "--" in sys.argv:
            args_index = sys.argv.index("--") + 1
            args = sys.argv[args_index:]
            if len(args) >= 2:
                json_path = args[0]
                output_dir = args[1]
            else:
                 raise ValueError("JSON path and output directory must be provided.")
        else:
             raise ValueError("Missing '--' separator.")


    except ValueError:
        # Error message removed, print usage string to stderr
        print("Usage: blender --background --python create_scene.py -- <json_path> <output_dir>", file=sys.stderr)
        sys.exit(1) # Error exit


    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f: # Specify encoding
        data = json.load(f)
    if "lighting" not in data:
            # Default lighting if missing
            data["lighting"] = { "sun_energy": 3.0, "sun_rotation_euler_degrees": [60, 0, 30], "environment_color": [0.8, 0.8, 0.8, 1.0], "environment_strength": 1.2 }
    if "objects" not in data or not isinstance(data["objects"], list):
            # Error message removed
            sys.exit(1)
    if "camera" not in data:
        # Error message removed
        sys.exit(1)
    # Validate size_params existence for all objects
    for i, obj_data in enumerate(data["objects"]):
        if "size_params" not in obj_data:
            # Error message removed
            sys.exit(1)


    # --- Execute Blender operations ---
    # Clean scene
    clear_scene()


    # Create camera
    if not create_camera(data["camera"]): sys.exit(1) # Exit if camera creation fails


    # Create objects
    successful_objects = 0
    for obj_data in data["objects"]:
        created_obj = create_object(obj_data)
        if created_obj:
                successful_objects += 1
    # Warning removed if no objects created


    # Set up lighting
    setup_lighting(data["lighting"])


    # Set render parameters
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 480
    scene.render.resolution_y = 320
    scene.cycles.samples = 256 # Increased samples slightly for potentially better quality
    scene.render.image_settings.file_format = 'PNG'


    # Calculate output path based on JSON filename and check for existing files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_png_path = os.path.join(output_dir, f"{base_name}.png")
    run_number = 1
    while os.path.exists(output_png_path):
        output_png_path = os.path.join(output_dir, f"{base_name}_run{run_number}.png")
        run_number += 1


    scene.render.filepath = output_png_path


    # Perform rendering
    bpy.ops.render.render(write_still=True)




if __name__ == "__main__":
    main()

