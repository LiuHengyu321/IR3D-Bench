def read_mapping_table(file_path='./1.txt'):
    """
    Read the mapping table from the file and return a dictionary.
    
    Args:
        file_path (str): Path to the mapping table file
        
    Returns:
        dict: A dictionary with view_val_s.json descriptions as keys and tuples of (9.json object names, index) as values
    """
    mapping = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Skip header and separator lines (first 2 lines)
    for line in lines[2:]:
        if line.strip():
            parts = line.strip().split('|')
            if len(parts) >= 4:  # Should have 4 parts: empty, description, object name, index
                key = parts[1].strip()
                value = parts[2].strip()
                index = int(parts[3].strip())
                
                # Convert "None" string to None value
                if value.lower() == "none":
                    value = None
                    
                mapping[key] = (value, index)
    
    return mapping

def lookup_object(description, mapping=None):
    """
    Look up a view_val_s.json object description to find its corresponding 9.json object name and index.
    
    Args:
        description (str): The object description from view_val_s.json
        mapping (dict, optional): A pre-loaded mapping dictionary. If None, will load from file.
        
    Returns:
        tuple or None: A tuple of (object_name, index) in 9.json or None if not found
    """
    if mapping is None:
        mapping = read_mapping_table()
    
    return mapping.get(description, None)



# Example usage
if __name__ == "__main__":
    # Read the mapping table
    mapping = read_mapping_table()
    print("Loaded mapping:", mapping)
    
    # Example lookup
    description = "brown large rubber cylinder"
    result = lookup_object(description)
    if result:
        object_name, index = result
        print(f"对象 '{description}' 映射到: {object_name}，索引号为: {index}")
    else:
        print(f"No mapping found for '{description}'")
    
    # Example evaluation
    pred_objects = [
        [[1.0, 2.0, 0.5], 0],  # [location, id]
        [[2.0, 2.5, 0.5], 1],
        [[0.0, 1.0, 0.5], 2],
    ]
    
    gt_relations = {
        "right": [[1], [2], []],
        "behind": [[], [0], []],
        "front": [[1], [], [0]],
        "left": [[2], [], [1]]
    }
    
    accuracy = evaluate_spatial_relations(pred_objects, gt_relations)
    print(f"Spatial relationship accuracy: {accuracy:.2%}") 