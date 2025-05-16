import os
import pdb # Import PDB for debugging
import sys # Import sys for sys.exit

import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import time # Added for potential retries, good practice

import torch
from transformers import CLIPTokenizer, CLIPModel
from segment_anything import sam_model_registry, SamPredictor
from openai import OpenAI # Added for LLM evaluation

from mapping_utils import read_mapping_table, lookup_object
from eval_utils import evaluate_spatial_relations, calculate_iou, dice_score, bbox_edge_distance_score
from eval_utils import evaluate_spatial_rule_sets

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# clip_model_path = "/data/zhangzy/clever_proj/hugging_face/transformers/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
clip_model_path = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
clip_model.eval()

sam_checkpoint = "/home/hansirui_3rd/zhangzy/data/Clevr_proj/code/segment-anything/weights/sam_vit_h_4b8939.pth"

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)


def compute_clip_text_similarity(text1, text2):
    """
    使用 Hugging Face CLIP 计算两个文本的语义相似度
    """
    inputs = clip_tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return (text_features[0] @ text_features[1]).item()


# def construct_extrinsic_matrix(position, rotation):
#     position = np.array(position)
#     rotation = np.array(rotation)
#     R_world = R.from_euler('xyz', rotation).as_matrix()
#     R_w2c = R_world.T
#     fix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#     R_fixed = fix @ R_w2c
#     T_fixed = -R_fixed @ position
#     E = np.eye(4)
#     E[:3, :3], E[:3, 3] = R_fixed, T_fixed
#     return E

def construct_extrinsic_matrix(position, rotation):
    """
    Constructs a 4x4 extrinsic matrix from camera position and rotation in degrees.

    Args:
        position: (x, y, z) camera position in world coordinates
        rotation: (rx, ry, rz) Euler angles in degrees (xyz order)

    Returns:
        4x4 numpy array representing the extrinsic matrix (camera-to-world)
    """
    position = np.array(position)
    rotation = np.array(rotation)

    # Convert Euler angles (in degrees) to rotation matrix
    R_world = R.from_euler('xyz', rotation, degrees=True).as_matrix()

    # World to camera: transpose of rotation
    R_w2c = R_world.T

    # Fix axes: Blender/OpenGL-style to OpenCV-style
    fix = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
    R_fixed = fix @ R_w2c

    # Translation
    T_fixed = -R_fixed @ position

    # Assemble extrinsic matrix
    E = np.eye(4)
    E[:3, :3] = R_fixed
    E[:3, 3] = T_fixed

    return E

def get_camera_matrices(camera):
    render_resolution = camera.get("render_resolution", [480, 320])

    fx = camera["lens"] * render_resolution[0] / camera["sensor_width"]
    fy = camera["lens"] * render_resolution[1] / camera["sensor_height"]
    cx = render_resolution[0] / 2
    cy = render_resolution[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    E = construct_extrinsic_matrix(camera["location"], camera["rotation_euler"])
    return K, E


def project_to_camera(location, K, E):
    P_world = np.append(location, 1)
    P_camera = E @ P_world
    X, Y, Z = P_camera[:3]
    if Z == 0:
        return np.array([0, 0])
    p = K @ [X / Z, Y / Z, 1]
    return p[:2]


def get_mask(image, point):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(np.array([point]), np.array([1]), multimask_output=False)
    return masks



def save_mask(img_array, output_filename):
    """
    保存图片到指定路径。
    
    参数：
      img_array: 输入的图像数组，可以是形状为 (1, H, W)、(H, W) 或 (H, W, channels) 的 numpy 数组。
      output_filename: 保存图片的路径和文件名。
    """
    # 如果图像数组形状为 (1, H, W) 则去除多余的维度
    if img_array.ndim == 3 and img_array.shape[0] == 1:
        img_array = np.squeeze(img_array, axis=0)

    # 保存图片
    cv2.imwrite(output_filename, img_array)

def mask_to_bbox(mask):
    mask_2d = mask[0]
    y_idx, x_idx = np.where(mask_2d > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return None
    return int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())


def evaluate_with_llm(client: OpenAI, model_name: str, system_prompt: str, pred_json_str: str, gt_json_str: str):
    """
    Evaluates scene descriptions using an LLM.
    If an error occurs during API call or parsing, the error will propagate up.
    Returns a tuple: (dictionary with scores, Python dictionary of LLM response for saving).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Predicted JSON:\n```json\n{pred_json_str}\n```\n\nGround Truth (GT) JSON:\n```json\n{gt_json_str}\n```"
        }
    ]
    llm_scores = {
        "llm_obj_fidelity": np.nan, # Initialize with NaN, will be overwritten on success
        "llm_scene_layout": np.nan,
        "llm_overall_visual": np.nan,
    }
    llm_output_for_saving = None # This will hold the dict to be saved on success
    
    # Errors from API call or parsing will now propagate up
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0, 
    )
    answer_from_api = chat_response.choices[0].message.content
    
    string_to_parse = answer_from_api
    if isinstance(string_to_parse, str):
        string_to_parse = string_to_parse.strip()
        if string_to_parse.startswith("```json") and string_to_parse.endswith("```"):
            first_newline = string_to_parse.find('\n')
            last_backticks = string_to_parse.rfind('```')
            if first_newline != -1 and last_backticks != -1 and first_newline < last_backticks:
                string_to_parse = string_to_parse[first_newline + 1:last_backticks].strip()
        elif string_to_parse.startswith("```") and string_to_parse.endswith("```"):
            string_to_parse = string_to_parse[3:-3].strip()
    
    parsed_answer = json.loads(string_to_parse) # This is the dict we want to save if successful
    llm_output_for_saving = parsed_answer 

    llm_scores["llm_obj_fidelity"] = parsed_answer.get("GPT4_1_JSON_Object_Appearance_Fidelity", {}).get("score")
    llm_scores["llm_scene_layout"] = parsed_answer.get("GPT4_1_JSON_Scene_Layout_Accuracy", {}).get("score")
    llm_scores["llm_overall_visual"] = parsed_answer.get("GPT4_1_JSON_Overall_Visual_Quality_and_Similarity", {}).get("score")
    
    for k, v in llm_scores.items():
        try:
            llm_scores[k] = float(v)
        except (ValueError, TypeError):
            # If a score is present but not a number (e.g., "N/A" string from LLM), keep it as NaN.
            # This specific try-except is for individual score conversion, not overall API call.
            llm_scores[k] = np.nan 
            print(f"Warning: Could not parse LLM score for {k} as float. Value: {v}. Setting to NaN.")
            
    return llm_scores, llm_output_for_saving


def evaluate_case(case_id, paths, openai_client: OpenAI = None, llm_model_name: str = None, llm_system_prompt: str = None):
    output_case_dir = os.path.join(paths["output_dir"], case_id)
    os.makedirs(output_case_dir, exist_ok=True)

    pred_json_file_path = os.path.join(paths["pred_json_dir"], f"{case_id}.json")
    gt_json_file_path = os.path.join(paths["gt_json_dir"], f"{case_id}.json")
    map_file_path = os.path.join(paths["map_dir"], f"{case_id}.txt")
    pred_render_img_path = os.path.join(paths["render_dir"], f"{case_id}.png")
    gt_actual_img_path = os.path.join(paths["gt_image_dir"], f"{case_id}.png")

    try:
        # All main processing logic for the case goes here
        with open(pred_json_file_path) as f:
            pred_data = json.load(f)
        with open(gt_json_file_path) as f:
            gt_data = json.load(f)
        
        map_file = map_file_path
        pred_img_path = pred_render_img_path
        gt_img_path = gt_actual_img_path
        
        mapping = read_mapping_table(map_file)
        pred_objects = pred_data["objects"]
        gt_objects = gt_data["scenes"][0]["objects"]
        gt_relation = gt_data["scenes"][0]["relationships"]

        K, E = get_camera_matrices(pred_data["camera"])
        pred_image = cv2.imread(pred_img_path)
        gt_image = cv2.imread(gt_img_path)

        coords_gt, coords_pred = [], []
        dist_list, clip_scores = [], []
        ious, dices, bboxes = [], [], []
        matched, pred_obj_list = 0, []

        for i, obj in enumerate(gt_objects):
            gt_coord = np.array(obj["pixel_coords"][:2])
            desc = f"{obj['color']} {obj['size']} {obj['material']} {obj['shape']}"
            lookup_result = lookup_object(desc, mapping)
            if lookup_result is None:
                continue
            pred_name, pred_id = lookup_result
            pred_obj = next((o for o in pred_objects if o["name"] == pred_name), None)
            if not pred_obj:
                continue
            matched += 1
            pred_coord = project_to_camera(pred_obj["location"], K, E)
            coords_gt.append(gt_coord)
            coords_pred.append(pred_coord)
            pred_obj_list.append([pred_obj["location"], pred_id])
            dist_list.append(np.linalg.norm(gt_coord - pred_coord))
            gt_mask = get_mask(gt_image, gt_coord).astype(np.uint8) * 255
            gt_mask_path = os.path.join(output_case_dir, f"gt_mask_{i}.png")
            save_mask(gt_mask, gt_mask_path)
            pred_mask = get_mask(pred_image, pred_coord).astype(np.uint8) * 255
            pred_mask_path = os.path.join(output_case_dir, f"pred_mask_{i}.png")
            save_mask(pred_mask, pred_mask_path)
            ious.append(calculate_iou(gt_mask, pred_mask))
            dices.append(dice_score(gt_mask, pred_mask))
            bboxes.append(bbox_edge_distance_score(mask_to_bbox(gt_mask), mask_to_bbox(pred_mask)))
            pred_tokens = pred_obj["name"].lower().split(" ")
            size_keywords = {"small", "large", "tiny", "big"}
            shape_keywords = {"cube", "sphere", "cylinder", "block", "ball", "tube"}
            material_keywords = {"metal", "rubber", "shiny", "matte", "metallic", "nonmetallic", "glossy", "dull", "plastic"}
            color_keywords = {"blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow", "black", "white", "orange", "pink"}
            color_token, size_token, material_token, shape_token = "none", "none", "none", "none"
            assigned_indices = set()
            for i_tok, token in enumerate(pred_tokens):
                if token in size_keywords and size_token == "none": size_token, _ = token, assigned_indices.add(i_tok)
                elif token in shape_keywords and shape_token == "none": shape_token, _ = token, assigned_indices.add(i_tok)
                elif token in material_keywords and material_token == "none": material_token, _ = token, assigned_indices.add(i_tok)
            found_color = False
            for i_tok, token in enumerate(pred_tokens):
                if i_tok not in assigned_indices and token in color_keywords and color_token == "none": color_token, _, found_color = token, assigned_indices.add(i_tok), True; break
            if not found_color and color_token == "none":
                for i_tok, token in enumerate(pred_tokens):
                    if i_tok not in assigned_indices: color_token, _ = token, assigned_indices.add(i_tok); break
            material_token_for_comparison = material_token
            if material_token in {"shiny", "metallic", "glossy"}: material_token_for_comparison = "metal"
            elif material_token in {"matte", "nonmetallic", "dull", "plastic"}: material_token_for_comparison = "rubber"
            elif material_token == "metal": material_token_for_comparison = "metal"
            elif material_token == "rubber": material_token_for_comparison = "rubber"
            clip_scores.append([
                compute_clip_text_similarity(color_token, obj["color"]),
                compute_clip_text_similarity(size_token, obj["size"]),
                compute_clip_text_similarity(material_token_for_comparison, obj["material"]),
                compute_clip_text_similarity(shape_token, obj["shape"]),
                compute_clip_text_similarity(pred_obj["name"], desc)
            ])

        if matched == 0:
            metrics = {
                "pixel_dist": np.nan, "count_acc": 0.0, "relation_acc": np.nan,
                "iou": np.nan, "dice": np.nan, "bbox": np.nan,
                "clip_color": np.nan, "clip_size": np.nan, "clip_material": np.nan,
                "clip_shape": np.nan, "clip_all": np.nan,
                "llm_obj_fidelity": np.nan, "llm_scene_layout": np.nan, "llm_overall_visual": np.nan
            }
        else:
            clip_scores_np = np.array(clip_scores)
            metrics = {
                "pixel_dist": np.mean(dist_list) / (480 * 320) if dist_list else np.nan,
                "count_acc": matched / len(gt_objects) if len(gt_objects) > 0 else 0,
                "relation_acc": evaluate_spatial_rule_sets(pred_obj_list, gt_relation)['overall']['accuracy'] if pred_obj_list else 0,
                "iou": np.mean(ious) if ious else np.nan,
                "dice": np.mean(dices) if dices else np.nan,
                "bbox": np.mean(bboxes) if bboxes else np.nan,
                "clip_color": np.mean(clip_scores_np[:, 0]) if clip_scores_np.size > 0 else np.nan,
                "clip_size": np.mean(clip_scores_np[:, 1]) if clip_scores_np.size > 0 else np.nan,
                "clip_material": np.mean(clip_scores_np[:, 2]) if clip_scores_np.size > 0 else np.nan,
                "clip_shape": np.mean(clip_scores_np[:, 3]) if clip_scores_np.size > 0 else np.nan,
                "clip_all": np.mean(clip_scores_np[:, 4]) if clip_scores_np.size > 0 else np.nan,
            }

        if openai_client and llm_model_name and llm_system_prompt:
            pred_json_str = json.dumps(pred_data)
            gt_json_str = json.dumps(gt_data)
            llm_eval_scores, llm_response_to_save = evaluate_with_llm(
                openai_client, llm_model_name, llm_system_prompt, pred_json_str, gt_json_str
            )
            metrics.update(llm_eval_scores)
            if llm_response_to_save is not None:
                llm_response_save_path = os.path.join(output_case_dir, f"llm_response_{case_id}.json")
                try:
                    with open(llm_response_save_path, 'w', encoding='utf-8') as f:
                        json.dump(llm_response_to_save, f, ensure_ascii=False, indent=4)
                except Exception as e_save_llm_resp:
                    print(f"Error saving LLM response for case {case_id} to {llm_response_save_path}: {e_save_llm_resp}")
        else:
            metrics["llm_obj_fidelity"] = np.nan
            metrics["llm_scene_layout"] = np.nan
            metrics["llm_overall_visual"] = np.nan

        case_metrics_save_path = os.path.join(output_case_dir, f"case_metrics_{case_id}.json")
        try:
            with open(case_metrics_save_path, 'w', encoding='utf-8') as f:
                serializable_metrics = {k: (v.item() if isinstance(v, np.generic) else (None if isinstance(v, float) and np.isnan(v) else v)) for k, v in metrics.items()}
                json.dump(serializable_metrics, f, ensure_ascii=False, indent=4)
        except Exception as e_save_metrics:
            print(f"Error saving case metrics for case {case_id} to {case_metrics_save_path}: {e_save_metrics}")

        return metrics

    except ValueError as ve:
        if "matmul: Input operand 1 has a mismatch in its core dimension" in str(ve):
            print(f"\n--- Skipping case {case_id} due to matrix dimension error ---")
            print(f"Error type: {type(ve).__name__}")
            print(f"Error message: {str(ve)}")
            return None
        else:
            raise

    except Exception as e:
        print(f"\n--- Exception occurred while processing case: {case_id} ---")
        print(f"  Predicate JSON: {pred_json_file_path}")
        print(f"  Ground Truth JSON: {gt_json_file_path}")
        print(f"  Predicted Image: {pred_render_img_path}")
        print(f"  Ground Truth Image: {gt_actual_img_path}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Entering PDB to debug. Type 'c' to continue to the next line in PDB, 'q' to quit PDB (and the script)." )
        pdb.post_mortem(e.__traceback__)
        print(f"--- Exited PDB for case: {case_id}. Re-raising original exception. ---")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--gt_json_dir", required=True)
    parser.add_argument("--pred_json_dir", required=True)
    parser.add_argument("--gt_image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_api_base", type=str, default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--llm_model_name", type=str, default=None)
    parser.add_argument("--llm_system_prompt_path", type=str, default="/home/hansirui_3rd/zhangzy/data/Clevr_proj/prompts/gpt4o_as_evaluator.txt")
    args = parser.parse_args()

    paths = vars(args)
    os.makedirs(paths["output_dir"], exist_ok=True)

    # Initialize OpenAI client (same as original)
    openai_client = None
    if args.llm_model_name:
        if not args.openai_api_key:
            print("Error: LLM evaluation requires OpenAI API key.", file=sys.stderr)
            sys.exit(1)
        try:
            openai_client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
            openai_client = None

    # Load LLM prompt (same as original)
    llm_prompt_to_use = None
    if openai_client and args.llm_system_prompt_path:
        try:
            with open(args.llm_system_prompt_path, 'r', encoding='utf-8') as f:
                llm_prompt_to_use = f.read()
        except Exception as e:
            print(f"Error loading LLM prompt: {e}")

    # Prepare case list (same as original)
    raw_case_filenames = [f for f in os.listdir(paths["pred_json_dir"]) if f.endswith(".json")]
    sorted_case_filenames = sorted(raw_case_filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    case_ids = [os.path.splitext(f)[0] for f in sorted_case_filenames]

    # Initialize counters
    num_total_cases = len(case_ids)
    num_skipped_loaded = 0
    num_processed_this_run = 0
    num_failed_load_reevaluated = 0
    num_failed_evaluation = 0
    num_skipped_matrix_error = 0
    num_skipped_missing_files = 0  # New counter for missing files

    all_collected_metrics = []

    # Main processing loop
    for case_id in tqdm(case_ids, desc="Evaluating cases"):
        tqdm.write(f"Case '{case_id}': Processing...")

        # Check required files
        required_files = {
            "map_file": os.path.join(paths["map_dir"], f"{case_id}.txt"),
            "pred_json": os.path.join(paths["pred_json_dir"], f"{case_id}.json"),
            "gt_json": os.path.join(paths["gt_json_dir"], f"{case_id}.json"),
            "pred_image": os.path.join(paths["render_dir"], f"{case_id}.png"),
            "gt_image": os.path.join(paths["gt_image_dir"], f"{case_id}.png"),
        }

        missing_flag = False
        for file_type, path in required_files.items():
            if not os.path.exists(path):
                tqdm.write(f"!! Missing {file_type}: {path}")
                missing_flag = True
        if missing_flag:
            num_skipped_missing_files += 1
            continue

        output_case_dir = os.path.join(paths["output_dir"], case_id)
        os.makedirs(output_case_dir, exist_ok=True)
        case_metrics_file = os.path.join(output_case_dir, f"case_metrics_{case_id}.json")

        # Check if the case metrics file already exists
        if os.path.exists(case_metrics_file):
            try:
                with open(case_metrics_file, 'r', encoding='utf-8') as f:
                    loaded_metrics = json.load(f)
                all_collected_metrics.append(loaded_metrics)
                num_skipped_loaded += 1
                tqdm.write(f"Case '{case_id}': Loaded from cache.")
                continue
            except Exception as e_load:
                tqdm.write(f"Failed to load metrics for case {case_id}: {str(e_load)}")
                num_failed_load_reevaluated += 1

        # Process the case if metrics file does not exist or failed to load
        try:
            current_case_metrics = evaluate_case(
                case_id, 
                paths,
                openai_client=openai_client,
                llm_model_name=args.llm_model_name,
                llm_system_prompt=llm_prompt_to_use
            )

            if current_case_metrics is None:
                num_skipped_matrix_error += 1
                tqdm.write(f"Case '{case_id}': Skipped due to matrix error.")
                continue

            all_collected_metrics.append(current_case_metrics)
            num_processed_this_run += 1
            tqdm.write(f"Case '{case_id}': Processed successfully.")

        except FileNotFoundError as fe:
            tqdm.write(f"Case '{case_id}': Key file missing: {str(fe)}")
            num_skipped_missing_files += 1
        except Exception as e:
            tqdm.write(f"Case '{case_id}': Processing failed: {str(e)}")
            num_failed_evaluation += 1

    # Final statistics
    print(f"\n=== Evaluation Summary ===")
    print(f"Total cases: {num_total_cases}")
    print(f"Skipped (missing files): {num_skipped_missing_files}")
    print(f"Skipped (matrix errors): {num_skipped_matrix_error}")
    print(f"Loaded from cache: {num_skipped_loaded}")
    print(f"Newly processed: {num_processed_this_run}")
    print(f"Failed evaluations: {num_failed_evaluation}")
    print(f"Failed cache loads: {num_failed_load_reevaluated}")
    print("==========================")

    # Existing metric calculation and output (same as original)
    if all_collected_metrics:
        metric_keys = ["pixel_dist", "count_acc", "relation_acc", "iou", "dice", "bbox",
                      "clip_color", "clip_size", "clip_material", "clip_shape", "clip_all",
                      "llm_obj_fidelity", "llm_scene_layout", "llm_overall_visual"]
        
        avg_metrics = {}
        for key in metric_keys:
            values_for_key = []
            for m in all_collected_metrics:
                val = m.get(key, np.nan)
                if val is None:
                    values_for_key.append(np.nan)
                elif isinstance(val, (int, float, np.generic)):
                    values_for_key.append(float(val))
                else:
                    values_for_key.append(np.nan)
            
            valid_values = [v for v in values_for_key if not np.isnan(v)]
            avg_metrics[key] = np.mean(valid_values) if valid_values else np.nan
            
        header = "| Pixel Distance | Count ACC | Relation ACC | Mask IOU | Mask DICE | BBox Score | CLIP Color | CLIP Size | CLIP Material | CLIP Shape | CLIP All | LLM Obj | LLM Layout | LLM Overall |"
        separator = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        values = f"| {avg_metrics['pixel_dist']:.10f} | {avg_metrics['count_acc']:.10f} | {avg_metrics['relation_acc']:.10f} | {avg_metrics['iou']:.10f} | {avg_metrics['dice']:.10f} | {avg_metrics['bbox']:.10f} | {avg_metrics['clip_color']:.10f} | {avg_metrics['clip_size']:.10f} | {avg_metrics['clip_material']:.10f} | {avg_metrics['clip_shape']:.10f} | {avg_metrics['clip_all']:.10f} | {avg_metrics['llm_obj_fidelity']:.10f} | {avg_metrics['llm_scene_layout']:.10f} | {avg_metrics['llm_overall_visual']:.10f} |"
        
        print("\n" + "\n".join([header, separator, values]))

if __name__ == "__main__":
    main()