import os
import json
from typing import List, Dict, Any

def process_train_files() -> List[Dict[str, Any]]:
    """
    Process all tree.json files in the current directory and format them into the required structure.
    """
    formatted_data = []
    
    # Walk through all directories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    # Check if file is empty
                    if os.path.getsize(file_path) == 0:
                        continue
                        
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Process each tree structure
                    if isinstance(data, dict) and 'base_question' in data and 'tree' in data:
                        question_data = {
                            "question": data['base_question'],
                            "variants": []
                        }
                        
                        # Keep track of variants we've seen for this question
                        seen_variants = {data['base_question']}
                        
                        # Process all variants from the tree structure
                        def process_node(node):
                            # Add variants from current level
                            for variant in node.get('variants', []):
                                # Skip if we've seen this variant before
                                # Check if this variant is symbolically equivalent to any we've seen
                                from sympy import simplify, sympify
                                try:
                                    variant_expr = sympify(variant)
                                    if any(simplify(variant_expr - sympify(seen)) == 0 for seen in seen_variants):
                                        continue
                                except:
                                    # If parsing fails, fall back to string comparison
                                    if variant in seen_variants:
                                        continue
                                    
                                seen_variants.add(variant)
                                variant_data = {
                                    "variant": variant,
                                    "difficulty": f"level_{node['level']}"
                                }
                                question_data["variants"].append(variant_data)
                            
                            # Process children recursively
                            for child in node.get('children', []):
                                process_node(child)
                        
                        # Process the tree starting from root
                        for root_node in data['tree']:
                            process_node(root_node)
                        
                        if question_data["variants"]:  # Only add if there are valid variants
                            formatted_data.append(question_data)
                
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            print(f"Processed {file_path}")
    
    return formatted_data

def main():
    # Process all train files
    formatted_data = process_train_files()
    
    # Create output directory if it doesn't exist
    os.makedirs('/home/ubuntu/test/TTRL/test_trees', exist_ok=True)
    
    # Write formatted data to output file
    output_path = os.path.join('/home/ubuntu/test/TTRL/test_trees', 'ttrl.json')
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=4)
    
    print(f"Processed data written to {output_path}")

    # Print number of questions processed
    print(f"Total number of questions processed: {len(formatted_data)}")

if __name__ == "__main__":
    main()
