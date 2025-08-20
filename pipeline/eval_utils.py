import numpy as np
from typing import List, Dict, Any


def evaluate_spatial_rule_sets(pred_objects, gt_relations):
    """
    Evaluate spatial relationships by treating each (subject, relation) pair as a rule unit.
    The whole set of target objects must match exactly with the predicted set to count as correct.

    Returns:
        dict: accuracy per relation and overall accuracy
    """
    def get_relative_direction(obj1_loc, obj2_loc):
        dx = obj2_loc[0] - obj1_loc[0]
        dy = obj2_loc[1] - obj1_loc[1]
        threshold = 0.1
        rels = []
        if dx > threshold:
            rels.append("right")
        elif dx < -threshold:
            rels.append("left")
        if dy > threshold:
            rels.append("behind")
        elif dy < -threshold:
            rels.append("front")
        return rels

    # Build ID to location map
    id_to_loc = {obj_id: loc for loc, obj_id in pred_objects}
    object_ids = list(id_to_loc.keys())
    num_objects = len(object_ids)

    relation_types = ["right", "left", "front", "behind"]
    results = {}
    total_correct = 0
    total_total = 0

    for rel in relation_types:
        correct = 0
        total = num_objects

        for i in range(num_objects):
            if i not in id_to_loc:
                continue

            pred_targets = []
            for j in range(num_objects):
                if j == i or j not in id_to_loc:
                    continue
                rels = get_relative_direction(id_to_loc[i], id_to_loc[j])
                if rel in rels:
                    pred_targets.append(j)

            pred_set = set(pred_targets)
            gt_set = set(gt_relations[rel][i]) if i < len(gt_relations[rel]) else set()

            if pred_set == gt_set:
                correct += 1

        acc = correct / total if total > 0 else 0.0
        results[rel] = {
            "correct": correct,
            "total": total,
            "accuracy": acc
        }

        total_correct += correct
        total_total += total

    results["overall"] = {
        "correct": total_correct,
        "total": total_total,
        "accuracy": total_correct / total_total if total_total > 0 else 0.0
    }

    return results

def evaluate_spatial_relations(pred_objects, gt_relations):
    """
    Evaluate the accuracy of predicted object locations against ground truth spatial relationships.
    Only counts as correct when all GT-defined relationships for a pair of objects match exactly.
    
    Args:
        pred_objects: List of [location, id], where location is [x,y,z] and id is the object index
        gt_relations: Dictionary containing ground truth relationships ("right", "left", "front", "behind")
    
    Returns:
        float: Accuracy of GT-defined spatial relationships
    """
    def get_relative_direction(obj1_loc, obj2_loc):
        """Helper function to determine relative direction between two objects"""
        # Calculate relative position
        dx = obj2_loc[0] - obj1_loc[0]  # Positive dx means obj2 is to the right of obj1
        dy = obj2_loc[1] - obj1_loc[1]  # Positive dy means obj2 is behind obj1
        
        # Define thresholds for relationship determination
        threshold = 0.1
        
        relations = []
        # Right/Left based on x-coordinate
        if dx > threshold:
            relations.append("right")
        elif dx < -threshold:
            relations.append("left")
            
        # Front/Behind based on y-coordinate (smaller y means more front)
        if dy > threshold:
            relations.append("behind")
        elif dy < -threshold:
            relations.append("front")
            
        return relations

    # Create a mapping from id to location
    id_to_loc = {obj[1]: obj[0] for obj in pred_objects}
    
    total_gt_relations = 0  # Count of GT-defined relationships
    correct_gt_relations = 0  # Count of correctly predicted GT relationships
    
    # For each object
    for obj_id in range(len(gt_relations["right"])):
        if obj_id not in id_to_loc:
            continue
            
        obj_loc = id_to_loc[obj_id]
        
        # For each other object
        for other_id in range(len(gt_relations["right"])):
            if other_id == obj_id or other_id not in id_to_loc:
                continue
                
            other_loc = id_to_loc[other_id]
            
            # Get predicted relationships for this pair
            pred_rels = get_relative_direction(obj_loc, other_loc)
            
            # Check if there are any GT relationships for this pair
            has_gt_relation = False
            is_correct = True  # Will be set to False if any GT relation doesn't match
            
            # Check each relationship type
            for rel_type in ["right", "left", "front", "behind"]:
                gt_has_rel = other_id in gt_relations[rel_type][obj_id]
                pred_has_rel = rel_type in pred_rels
                
                if gt_has_rel:
                    has_gt_relation = True
                    if gt_has_rel != pred_has_rel:
                        is_correct = False
                        break
            
            # Only count pairs that have GT relationships
            if has_gt_relation:
                total_gt_relations += 1
                if is_correct:
                    correct_gt_relations += 1
    
    return correct_gt_relations / total_gt_relations if total_gt_relations > 0 else 0.0


def calculate_iou(mask1, mask2):

    if mask1.ndim == 3 and mask1.shape[0] == 1:
        mask1 = np.squeeze(mask1, axis=0)
    if mask2.ndim == 3 and mask2.shape[0] == 1:
        mask2 = np.squeeze(mask2, axis=0)
    
    if mask1.dtype != bool:
        mask1 = mask1 > 0
    if mask2.dtype != bool:
        mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    

    if union == 0:
        return 1.0 
    
    iou = intersection / union
    return iou


def dice_score(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute Dice Score between prediction and ground truth masks.

    Parameters:
    - pred_mask: np.ndarray of shape (1, H, W), binary (0 or 1)
    - gt_mask: np.ndarray of shape (1, H, W), binary (0 or 1)
    - smooth: small constant to avoid division by zero

    Returns:
    - dice: float, Dice coefficient
    """
    pred = pred_mask.astype(np.bool_)
    gt = gt_mask.astype(np.bool_)

    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def bbox_edge_distance_score(bbox1, bbox2, eps=1e-6):
    """
    Compute a similarity score based on center-to-edge distances.

    Parameters:
    - bbox1, bbox2: tuple (x_min, y_min, x_max, y_max) or None
    - eps: small value to avoid division by zero

    Returns:
    - score: float in [0, 1], higher is better, or np.nan if a bbox is None
    """
    if bbox1 is None or bbox2 is None:
        return np.nan

    def center_and_distances(bbox):
        if bbox is None: # Should ideally be caught by the check above, but good for robustness
            return None 
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        left   = cx - x_min
        right  = x_max - cx
        top    = cy - y_min
        bottom = y_max - cy
        return np.array([left, right, top, bottom])

    d1 = center_and_distances(bbox1)
    d2 = center_and_distances(bbox2)

    if d1 is None or d2 is None: # If center_and_distances somehow still got None (e.g., if called directly)
        return np.nan

    # L1 relative distance score
    diff = np.abs(d1 - d2)
    denom = np.abs(d1).sum() + eps
    score = 1.0 - diff.sum() / denom
    score = np.clip(score, 0.0, 1.0)
    return score

