import os

import json
import argparse
import torch
# Import CLIPTokenizer and CLIPModel from transformers
from transformers import CLIPTokenizer, CLIPModel 
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment # Import Hungarian algorithm solver


# --- CLIP Setup using specific local path ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the specific snapshot path
# clip_model_path = "/data/zhangzy/clever_proj/hugging_face/transformers/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
clip_model_path = "openai/clip-vit-base-patch32"
try:
    # Load tokenizer and model from the specific path
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval() # Set model to evaluation mode
    print(f"CLIP model loaded successfully from path: {clip_model_path} on {device}.")
except Exception as e:
    print(f"Error loading CLIP model from path '{clip_model_path}': {e}")
    print("Please ensure the path is correct and contains the necessary files.")
    raise e


# --- Similarity Function using HF CLIP Tokenizer and Model ---
@torch.no_grad()
def compute_clip_text_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """
    Computes CLIP similarity between two text strings using HF Transformers.
    Handles None inputs by returning 0 similarity.
    """
    if text1 is None or text2 is None:
        return 0.0 # Treat missing attributes as dissimilar

    try:
        # Use CLIPTokenizer
        inputs = clip_tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True).to(device)
        # Get text features from CLIPModel
        text_features = clip_model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features[0] @ text_features[1]).item()
        # Clamp similarity to be non-negative for easier interpretation/combination
        return max(0.0, similarity)
    except Exception as e:
        print(f"Warning: Error computing HF CLIP similarity for '{text1}' vs '{text2}': {e}")
        return 0.0 # Return low similarity on error


# --- Attribute Extraction ---

def extract_attributes_pred(pred_obj: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts attributes from the predicted object JSON structure."""
    attributes = {"color": None, "size": None, "shape": None, "material": None}
    name = pred_obj.get("name")

    # Try extracting from 'name' first as it seems structured
    if name:
        parts = name.lower().split()
        # Basic extraction based on expected name structure
        # This might need refinement based on actual variations
        if len(parts) >= 4:
            attributes["color"] = parts[0]
            attributes["size"] = parts[1]
            attributes["material"] = parts[2] # 'matte' or 'shiny' expected here
            attributes["shape"] = parts[3]
        elif len(parts) == 3: # Handle cases like "large shiny sphere"
             attributes["size"] = parts[0]
             attributes["material"] = parts[1]
             attributes["shape"] = parts[2]
             # color might be missing or derived from material base_color if needed

    # Refine/Overwrite using other fields if needed (more robust)
    # Size (Example refinement - adapt based on actual size_params logic)
    size_params = pred_obj.get("size_params", {})
    if "size" in size_params:
        # Assuming simple size mapping, adjust as needed
        attributes["size"] = "large" if size_params["size"] >= 1.5 else "small"
    elif "radius" in size_params:
         attributes["size"] = "large" if size_params["radius"] >= 0.75 else "small"

    # Material (Example refinement)
    material_details = pred_obj.get("material", {})
    metallic = material_details.get("metallic", 0.0)
    # roughness = material_details.get("roughness", 0.5) # Could also use roughness
    if metallic is not None:
        attributes["material"] = "metal" if metallic > 0.5 else "rubber" # Map shiny/matte to GT terms

    # Shape (Example refinement - if name parsing fails)
    if attributes["shape"] is None and name:
        if "cylinder" in name: attributes["shape"] = "cylinder"
        elif "sphere" in name: attributes["shape"] = "sphere"
        elif "cube" in name: attributes["shape"] = "cube"

    # Color (Could potentially map base_color RGB to name, but complex)
    # For now, relying on name parsing or leave as None if not in name

    return attributes


def extract_attributes_gt(gt_obj: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts attributes from the ground truth object JSON structure."""
    return {
        "color": gt_obj.get("color"),
        "size": gt_obj.get("size"),
        "shape": gt_obj.get("shape"),
        "material": gt_obj.get("material")
    }


# --- Similarity Calculation ---

def calculate_object_similarity(pred_attrs: Dict[str, Optional[str]], gt_attrs: Dict[str, Optional[str]]) -> float:
    """Calculates overall similarity based on attribute CLIP scores."""
    scores = []
    # Compare common attributes
    for key in ["color", "size", "shape", "material"]:
        score = compute_clip_text_similarity(pred_attrs.get(key), gt_attrs.get(key))
        scores.append(score)

    # Return the average similarity score
    return sum(scores) / len(scores) if scores else 0.0


# --- Matching Logic using Hungarian Algorithm ---

def find_best_matches(pred_objects: List[Dict], gt_objects: List[Dict], similarity_threshold: float = 0.5) -> Dict[int, int]:
    """
    Finds the best GT match for each predicted object using the Hungarian algorithm
    (linear_sum_assignment) to achieve a globally optimal one-to-one assignment.
    Returns a dictionary mapping predicted object index to GT object index.
    Matches below the similarity_threshold are discarded (-1).
    """
    num_pred = len(pred_objects)
    num_gt = len(gt_objects)
    if num_pred == 0 or num_gt == 0:
        return {}

    # Extract attributes for all objects first
    pred_attributes = [extract_attributes_pred(p) for p in pred_objects]
    gt_attributes = [extract_attributes_gt(g) for g in gt_objects]

    # Calculate similarity matrix
    similarity_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            similarity_matrix[i, j] = calculate_object_similarity(pred_attributes[i], gt_attributes[j])

    # Convert similarity to cost for minimization problem
    # Cost = 1 - Similarity (since similarity is capped at 1 and higher is better)
    cost_matrix = 1 - similarity_matrix

    # Solve the assignment problem using Hungarian algorithm
    # Finds the assignment that minimizes the total cost (maximizes total similarity)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the final mapping, applying the threshold
    matches = {i: -1 for i in range(num_pred)} # Initialize all predictions as unmatched
    for r, c in zip(row_ind, col_ind):
        # Check if the similarity of the optimal match meets the threshold
        if similarity_matrix[r, c] >= similarity_threshold:
            matches[r] = c

    return matches


# --- File I/O ---

def save_mapping_to_file(mapping: Dict[int, int], pred_objects: List[Dict], gt_objects: List[Dict], filename: str):
    """Saves the mapping as a Markdown table with GT Desc, Pred Desc, Matched GT Index (0-based), ordered by GT object index."""
    with open(filename, "w", encoding="utf-8") as f:
        # Write Markdown table header
        f.write("| column1 | column2 | column3 |\n")
        f.write("|---------|---------|---------|\n")
        
        # Create an inverted mapping (gt_idx -> pred_idx) for efficient lookup
        inverted_mapping = {v: k for k, v in mapping.items() if v != -1}
        
        # Iterate through GT objects based on their original order
        for gt_idx, gt_obj in enumerate(gt_objects):
            # Check if this GT object was matched by any prediction
            if gt_idx in inverted_mapping:
                pred_idx = inverted_mapping[gt_idx]
                
                # Ensure pred_idx is valid
                if 0 <= pred_idx < len(pred_objects):
                    # Get prediction description (Column 2)
                    pred_name = pred_objects[pred_idx].get("name", f"Unnamed Pred Object {pred_idx}")
                    # Get GT description (Column 1 - using spaces)
                    gt_desc = f"{gt_obj.get('color','?')} {gt_obj.get('size','?')} {gt_obj.get('material','?')} {gt_obj.get('shape','?')}"
                    # Get 0-based MATCHED GT index (Column 3 - Changed back to 0-based)
                    gt_index_0based = gt_idx 
                    # Write the Markdown table row 
                    f.write(f"| {gt_desc} | {pred_name} | {gt_index_0based} |\n") # Use 0-based index
                else:
                    # Log error if pred_idx is somehow invalid
                    print(f"[Warning] Invalid Pred Index {pred_idx} found for GT Index {gt_idx} in file {filename}. Skipping output row.")
            # else: This GT object was not matched, so we don't write a row for it.


# --- Main Execution ---

def main(map_dir: str, pred_dir: str, gt_dir: str):
    os.makedirs(map_dir, exist_ok=True)

    raw_pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    
    # Sort files numerically based on the number in the filename (e.g., CLEVR_val_XXXXXX.json)
    def get_filenumber(filename):
        # Extracts XXXXXX from CLEVR_val_XXXXXX.json
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            return -1 # Fallback for unexpected filenames, sorts them first or last depending on use

    pred_files = sorted(raw_pred_files, key=get_filenumber)

    num_total_files = len(pred_files)
    num_skipped = 0
    num_processed = 0
    num_gt_missing = 0
    num_json_errors = 0
    num_other_errors = 0

    print(f"\nFound {num_total_files} JSON files in prediction directory: {pred_dir}")

    for filename in pred_files:
        output_filename_base = os.path.splitext(filename)[0]
        output_map_file = os.path.join(map_dir, f"{output_filename_base}.txt")

        if os.path.exists(output_map_file):
            print(f"[Info] Skipping '{filename}': Output mapping '{output_map_file}' already exists.")
            num_skipped += 1
            continue

        pred_path = os.path.join(pred_dir, filename)
        gt_filename_base = os.path.basename(filename)
        gt_path = os.path.join(gt_dir, gt_filename_base)

        if not os.path.exists(gt_path):
            print(f"[Warning] GT file not found for prediction '{filename}': {gt_path}. Skipping.")
            num_gt_missing += 1
            continue

        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            pred_objects = pred_data.get("objects", [])
            gt_objects = []
            if "scenes" in gt_data and len(gt_data["scenes"]) > 0:
                 gt_objects = gt_data["scenes"][0].get("objects", [])
            else:
                 print(f"[Warning] Could not find 'scenes' or 'objects' in GT file: {gt_path}. Skipping.")
                 # Still save an empty map file to mark as "processed" for resume logic
                 save_mapping_to_file({}, [], [], output_map_file)
                 num_gt_missing +=1 # Count as a form of missing GT data for this purpose
                 continue

            if not pred_objects or not gt_objects:
                print(f"[Info] Skipping {filename}: No objects found in prediction or ground truth. Creating empty map file.")
                save_mapping_to_file({}, [], [], output_map_file) 
                num_processed +=1 # Count as processed because we created the output map file
                continue

            mapping = find_best_matches(pred_objects, gt_objects, similarity_threshold=0.5)
            save_mapping_to_file(mapping, pred_objects, gt_objects, output_map_file)
            print(f"[✓] Processed '{filename}' and saved mapping to: {output_map_file}")
            num_processed += 1

        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON for {filename}: {e}")
            num_json_errors += 1
        except Exception as e:
            print(f"[Error] Failed processing {filename}: {e}")
            num_other_errors += 1

    print(f"\n--- Matching Summary ---")
    print(f"Total JSON files found: {num_total_files}")
    print(f"Skipped (output map already existed): {num_skipped}")
    print(f"Processed in this run: {num_processed}")
    if num_gt_missing > 0:
        print(f"Skipped (GT file missing or invalid structure): {num_gt_missing}")
    if num_json_errors > 0:
        print(f"Errors (JSON parsing failed): {num_json_errors}")
    if num_other_errors > 0:
        print(f"Errors (other processing issues): {num_other_errors}")
    print(f"------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast attribute-based object matching (Hungarian Algorithm) between predicted and GT JSON files using HF CLIP.")
    parser.add_argument("--map_dir", type=str, required=True, help="Directory to save the output mapping files.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted JSON files.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth JSON files (CLEVR format).")
    args = parser.parse_args()

    main(args.map_dir, args.pred_dir, args.gt_dir) 