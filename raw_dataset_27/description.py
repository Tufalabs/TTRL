import os
import json

# Count filled vs empty directories
dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
filled_dirs = []
empty_dirs = []

for d in dirs:
    if len(os.listdir(d)) > 0:
        filled_dirs.append(d)
    else:
        empty_dirs.append(d)

print(f"Found {len(filled_dirs)} filled directories and {len(empty_dirs)} empty directories")

# Analyze variations in train.json for filled directories
total_variations = 0
easy_variations = 0
equivalent_variations = 0
harder_variations = 0

num_files_with_easier = 0
num_files_with_equivalent = 0
num_files_with_harder = 0

num_files_with_ground_truth = 0

all_questions = []

for dir_name in filled_dirs:
    train_file = os.path.join(dir_name, 'train.json')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            
        if "easier" in train_data["variants"]:
            num_files_with_easier += 1
            total_variations += len(train_data["variants"]["easier"])
            easy_variations += len(train_data["variants"]["easier"])
            
        if "equivalent" in train_data["variants"]:
            num_files_with_equivalent += 1
            total_variations += len(train_data["variants"]["equivalent"])
            equivalent_variations += len(train_data["variants"]["equivalent"])
            
        if "harder" in train_data["variants"]:
            num_files_with_harder += 1
            total_variations += len(train_data["variants"]["harder"])
            harder_variations += len(train_data["variants"]["harder"])
    
    test_file = os.path.join(dir_name, 'test.json')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            if "question" in test_data:
                all_questions.append(test_data["question"])
            
            if test_data["ground_truth_solution"] != None:
                num_files_with_ground_truth += 1

# Save all questions to a single JSON file
questions_file = 'all_test_questions.json'
with open(questions_file, 'w') as f:
    json.dump(all_questions, f, indent=2)

print(f"\nCollected {len(all_questions)} questions and saved to {questions_file}")

print(f"\nAnalysis of variations in train.json files:")
print(f"Total variations: {total_variations}")
print(f"Easy variations: {easy_variations}")
print(f"Equivalent variations: {equivalent_variations}")
print(f"Harder variations: {harder_variations}")

print(f"Average number of variations per file: {total_variations / len(filled_dirs)}")
print(f"Average number of easier variations per file: {easy_variations / num_files_with_easier}")
print(f"Average number of equivalent variations per file: {equivalent_variations / num_files_with_equivalent}")
print(f"Average number of harder variations per file: {harder_variations / num_files_with_harder}")

print(f"Files with easier variations: {num_files_with_easier} / {len(filled_dirs)}")
print(f"Files with equivalent variations: {num_files_with_equivalent} / {len(filled_dirs)}")
print(f"Files with harder variations: {num_files_with_harder} / {len(filled_dirs)}")

print(f"Files with ground truth: {num_files_with_ground_truth} / {len(filled_dirs)}")