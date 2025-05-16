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
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    for block in list(bpy.data.materials): bpy.data.materials.remove(block)
    for block in list(bpy.data.meshes): bpy.data.meshes.remove(block)
    for block in list(bpy.data.lights): bpy.data.lights.remove(block)
    for block in list(bpy.data.cameras): bpy.data.cameras.remove(block)
    for block in list(bpy.data.worlds): bpy.data.worlds.remove(block)

def create_camera(camera_data):
    """Create a camera based on the configuration."""
    if not all(k in camera_data for k in ['name', 'location', 'rotation_euler']):
        return None

    camera = bpy.data.cameras.new(name=camera_data["name"])
    camera_ob = bpy.data.objects.new(name=camera_data["name"], object_data=camera)

    camera_ob.location = Vector(camera_data["location"])
    camera_ob.rotation_euler = [math.radians(r) for r in camera_data["rotation_euler"]]

    camera.lens = camera_data.get("lens", 50)
    camera.sensor_width = camera_data.get("sensor_width", 36)
    camera.sensor_height = camera_data.get("sensor_height", 36)
    camera.clip_start = camera_data.get("clip_start", 0.1)
    camera.clip_end = camera_data.get("clip_end", 100)

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
        return 'CUBE'

def create_object(obj_data):
    """Create a single object."""
    required_keys = ['name', 'location', 'rotation_euler', 'material', 'size_params']
    if not all(key in obj_data for key in required_keys):
        return None

    shape = determine_shape(obj_data["name"])
    location = Vector(obj_data["location"])
    rotation_rad = [math.radians(r) for r in obj_data["rotation_euler"]]
    size_params = obj_data["size_params"]

    if shape == 'CUBE':
        size = size_params.get('size', 1.0)
        bpy.ops.mesh.primitive_cube_add(size=size, location=location, rotation=rotation_rad)
    elif shape == 'SPHERE':
        radius = size_params.get('radius', 0.5)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, rotation=rotation_rad)
    elif shape == 'CYLINDER':
        radius = size_params.get('radius', 0.5)
        depth = size_params.get('depth', 1.0)
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation_rad)
    else:
        return None

    obj = bpy.context.active_object
    if not obj:
        return None

    obj.name = obj_data["name"]

    material = create_material(obj_data["material"])
    if material:
        assign_material(obj, material)

    return obj

def create_material(mat_data):
    """Create a material."""
    if not all(k in mat_data for k in ['name', 'base_color', 'metallic', 'roughness']):
        return None

    mat_name = mat_data["name"]
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    while nodes: nodes.remove(nodes[0])

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    color = mat_data["base_color"]
    if isinstance(color, (list, tuple)) and len(color) == 3:
        color = list(color) + [1.0]
    elif not (isinstance(color, (list, tuple)) and len(color) == 4):
        color = [0.0, 0.0, 0.0, 1.0]

    principled.inputs['Base Color'].default_value = color
    principled.inputs['Metallic'].default_value = float(mat_data["metallic"])
    principled.inputs['Roughness'].default_value = float(mat_data["roughness"])

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat

def assign_material(obj, material):
    """Assign a material to an object."""
    if not obj or not obj.data:
        return
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

def setup_lighting(lighting_data):
    """Set up lighting based on configuration."""
    sun_rotation_rad = [math.radians(a) for a in lighting_data.get("sun_rotation_euler_degrees", [60, 0, 30])]
    sun_energy = lighting_data.get("sun_energy", 3.0)
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.rotation_euler = sun_rotation_rad
    sun.data.energy = sun_energy

    env_color = lighting_data.get("environment_color", [0.8, 0.8, 0.8, 1.0])
    env_strength = lighting_data.get("environment_strength", 1.2)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg_node = next((n for n in nodes if n.type == 'BACKGROUND'), nodes.new(type='ShaderNodeBackground'))
    output_node = nodes.get("World Output") or nodes.new(type='ShaderNodeOutputWorld')

    bg_node.inputs['Color'].default_value = env_color
    bg_node.inputs['Strength'].default_value = env_strength

    if not any(link.from_node == bg_node and link.to_node == output_node for link in links):
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

def main():
    """Main function for batch processing a directory of JSON files."""
    json_dir = None
    output_dir = None

    # Parse command line arguments
    try:
        if "--" in sys.argv:
            args_index = sys.argv.index("--") + 1
            args = sys.argv[args_index:]
            if len(args) >= 2:
                json_dir = args[0]
                output_dir = args[1]
            else:
                raise ValueError("JSON directory and output directory must be provided.")
        else:
            raise ValueError("Missing '--' separator.")
    except ValueError:
        print("Usage: blender --background --python create_scene.py -- <json_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(json_dir):
        print(f"Error: JSON directory '{json_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    # Sort files numerically based on the number in the filename (e.g., CLEVR_val_XXXXXX.json)
    def get_filenumber(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            return -1 # Fallback for unexpected filenames

    json_files = sorted(raw_json_files, key=get_filenumber)

    if not json_files:
        print(f"No .json files found in directory: {json_dir}")
        sys.exit(1)

    num_total_files = len(json_files)
    num_skipped = 0
    num_processed = 0
    num_invalid_json = 0
    num_render_errors = 0 # For other errors during render setup/execution

    print(f"\nFound {num_total_files} JSON files in directory: {json_dir}")

    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        output_png_path = os.path.join(output_dir, f"{base_name}.png")

        if os.path.exists(output_png_path):
            print(f"Skipping: Output image '{output_png_path}' already exists for '{json_file}'.")
            num_skipped += 1
            continue

        json_path = os.path.join(json_dir, json_file)
        print(f"Processing: {json_path} -> {output_png_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data.setdefault("lighting", {
                "sun_energy": 3.0,
                "sun_rotation_euler_degrees": [60, 0, 30],
                "environment_color": [0.8, 0.8, 0.8, 1.0],
                "environment_strength": 1.2
            })

            if "objects" not in data or not isinstance(data["objects"], list) or "camera" not in data:
                print(f"Invalid JSON structure in file: {json_file}. Skipping render.", file=sys.stderr)
                num_invalid_json +=1
                continue

            # Basic check for size_params - can be expanded if needed
            valid_objects = True
            for obj_data_idx, obj_data in enumerate(data["objects"]):
                if "size_params" not in obj_data:
                    print(f"Missing 'size_params' in object #{obj_data_idx} in file: {json_file}. Skipping render.", file=sys.stderr)
                    valid_objects = False
                    break
            if not valid_objects:
                num_invalid_json +=1
                continue

            clear_scene()

            if not create_camera(data["camera"]):
                print(f"Camera creation failed in file: {json_file}. Skipping render.", file=sys.stderr)
                num_render_errors +=1
                continue

            for obj_data in data["objects"]:
                create_object(obj_data) # Assume create_object handles its own errors gracefully or returns None

            setup_lighting(data["lighting"])

            scene = bpy.context.scene
            scene.render.engine = 'CYCLES'
            scene.render.resolution_x = 480
            scene.render.resolution_y = 320
            scene.cycles.samples = 256 # Reduced for faster processing if needed, adjust as per quality requirements
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = output_png_path
            
            bpy.ops.render.render(write_still=True)
            print(f"Rendered: {output_png_path}")
            num_processed += 1
        
        except json.JSONDecodeError as e_json:
            print(f"Error decoding JSON from '{json_file}': {e_json}. Skipping render.", file=sys.stderr)
            num_invalid_json += 1
        except Exception as e_render:
            print(f"An error occurred while processing or rendering '{json_file}': {e_render}. Skipping render.", file=sys.stderr)
            num_render_errors += 1

    print(f"\n--- Rendering Summary ---")
    print(f"Total JSON files found: {num_total_files}")
    print(f"Skipped (output image already existed): {num_skipped}")
    print(f"Processed (rendered) in this run: {num_processed}")
    if num_invalid_json > 0:
        print(f"Skipped (invalid JSON structure or missing critical data): {num_invalid_json}")
    if num_render_errors > 0:
        print(f"Errors (render setup or execution failed): {num_render_errors}")
    print(f"-------------------------\n")

if __name__ == "__main__":
    main()