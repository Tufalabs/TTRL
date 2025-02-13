import os
import json
from typing import List, Dict, Any

def process_train_files() -> List[Dict[str, Any]]:
    """
    Process all train.json files in the current directory and format them into the required structure.
    """
    formatted_data = []
    
    # Walk through all directories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('train.json'):
                file_path = os.path.join(root, file)
                
                try:
                    # Check if file is empty
                    if os.path.getsize(file_path) == 0:
                        continue
                        
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Process each integral and its variants
                    if isinstance(data, dict) and 'integral' in data and 'variants' in data:
                        question_data = {
                            "question": data['integral'],
                            "variants": []
                        }
                        
                        # Process all variants regardless of difficulty level
                        for difficulty, variants in data['variants'].items():
                            for variant in variants:
                                if variant.get('variant') and variant.get('reasoning'):
                                    variant_data = {
                                        "variant": variant['variant'],
                                        "reasoning": variant['reasoning'],
                                        "difficulty": variant.get('requested_difficulty', 'unknown')
                                    }
                                    question_data["variants"].append(variant_data)
                        
                        if question_data["variants"]:  # Only add if there are valid variants
                            formatted_data.append(question_data)
                
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
    
    return formatted_data

def main():
    # Process all train files
    formatted_data = process_train_files()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Write formatted data to output file
    output_path = os.path.join('/home/ubuntu/o1-replication/TTRL/variant_results', 'ttrl.json')
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=4)
    
    print(f"Processed data written to {output_path}")

    # Print number of questions processed
    print(f"Total number of questions processed: {len(formatted_data)}")

if __name__ == "__main__":
    main()
