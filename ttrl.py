import json
import os
import pandas as pd
from typing import List, Dict
from utils import run_inference, variants_to_parquet, run_rl, verify_response

def main(path_to_questions: str, parquet_output_path: str, model_dir: str, max_new_tokens: int, project_name: str):
    # 1. Read the integration test questions from a file with the following format:
    #
    # [ 
    #     {
    #         "question": "integrate(1/(x**2 - x + 1), x)",
    #         "variants": [
    #             {
    #                 "variant": "integrate(1/(x**2 - x + 1), x)",
    #                 "reasoning": "The integral can be transformed using the substitution u = x - 1/2, which simplifies the denominator.",
    #                 "difficulty": "easier"
    #             }
    #         ]
    #     },
    #     {
    #         "question": "integrate(1/(x**2 - x + 1), x)",
    #         "variants": [
    #             {
    #                 "variant": "integrate(1/(x**2 - x + 1), x)",
    #                 "reasoning": "The integral can be transformed using the substitution u = x - 1/2, which simplifies the denominator.",
    #                 "difficulty": "easier"
    #             }
    #         ]
    #     }
    # ]
    import time
    start_time = time.time()

    with open(path_to_questions, 'r') as f:
        data = json.load(f)
    
    # 2. Loop through each question and put the variants into a parquet file to be ready for the RL    
    for q_id, question_data in enumerate(data):
        variants_to_parquet(data=question_data, output_path=parquet_output_path, question_id=q_id)
        print(f"Saved variants for question {q_id} to {parquet_output_path}/variants_q{q_id}.parquet and {parquet_output_path}/variants_q{q_id}.json")
        print(f"Saved val test of question pass@16 {q_id} to {parquet_output_path}/test_q{q_id}.parquet and {parquet_output_path}/test_q{q_id}.json")

    num_correct = 0
    num_total = len(data)
    results = [] 
    
    # 3. For each question, run GRPO using CustomTinyZero and the numerical_integration reward function
    for q_id, question_data in enumerate(data):
        print(f"RUNNING GRPO FOR QUESTION {q_id}")
        experiment_name = f"llama3.2_3b_ttrl_integration_q{q_id}"
        last_checkpoint_path = run_rl(model_dir=model_dir, train_parquet_path=f"{parquet_output_path}/variants_q{q_id}.parquet", test_parquet_path=f"{parquet_output_path}/test_q{q_id}.parquet", project_name=project_name, experiment_name=experiment_name, max_new_tokens=max_new_tokens)
        print(f"Last checkpoint path: {last_checkpoint_path}")
        
        instruction_following = "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer> wrapped in a <answer> tags. Importantly, put * between terms you want to multiply! Show your full working out before solving, don't include any constants of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags."
        prompt = f"{question_data['question']}\n{instruction_following}"
        print("--------------------------------")
        print(f"Question {q_id}")
        print(f"Prompt: {prompt}")
        
        # Run multiple passes and track correctness
        passes = [1, 5, 10]
        correct_by_pass = {p: 0 for p in passes}
        responses = []
        
        # Generate responses with base model
        print("Generating responses with base model")
        base_responses = []
        base_correct_by_pass = {p: 0 for p in passes}
        
        for i in range(max(passes)):
            base_response = run_inference(model_dir=model_dir, prompt=prompt, max_new_tokens=max_new_tokens)
            base_responses.append(base_response)
            print(f"Base Model Response {i+1}: {base_response}")
            is_correct = verify_response(response=base_response, ground_truth=question_data["question"], num_tests=5, timeout_secs=10)
            print(f"Base Model Pass {i+1} correct: {is_correct}")
            
            # Check if correct for each pass@k
            for p in passes:
                if i < p and is_correct:
                    base_correct_by_pass[p] += 1
                    break
                    
        # Generate responses with RL model
        print("Generating responses with RL model")
        rl_responses = []
        rl_correct_by_pass = {p: 0 for p in passes}
        
        for i in range(max(passes)):
            rl_response = run_inference(model_dir=last_checkpoint_path, prompt=prompt, max_new_tokens=max_new_tokens)
            rl_responses.append(rl_response)
            print(f"RL Model Response {i+1}: {rl_response}")
            is_correct = verify_response(response=rl_response, ground_truth=question_data["question"], num_tests=5, timeout_secs=10)
            print(f"RL Model Pass {i+1} correct: {is_correct}")
            
            # Check if correct for each pass@k
            for p in passes:
                if i < p and is_correct:
                    rl_correct_by_pass[p] += 1
                    break
        
        print("--------------------------------")
        
        # Store results for this question
        results.append({
            "question_id": q_id,
            "question": question_data["question"],
            "base_model": {
                "responses": base_responses,
                "pass@1": base_correct_by_pass[1] > 0,
                "pass@5": base_correct_by_pass[5] > 0, 
                "pass@10": base_correct_by_pass[10] > 0
            },
            "rl_model": {
                "responses": rl_responses,
                "pass@1": rl_correct_by_pass[1] > 0,
                "pass@5": rl_correct_by_pass[5] > 0,
                "pass@10": rl_correct_by_pass[10] > 0
            }
        })
        
        # Track overall correct answers for pass@1
        if rl_correct_by_pass[1] > 0:
            num_correct += 1
    
    print(f"Number of correct answers (RL model, pass@1): {num_correct}")
    print(f"Number of total answers: {num_total}")
    print(f"Accuracy (RL model, pass@1): {num_correct / num_total}")
    
    # Calculate pass@k accuracies for both models
    base_pass_k_correct = {p: sum(1 for r in results if r['base_model'][f'pass@{p}']) for p in passes}
    rl_pass_k_correct = {p: sum(1 for r in results if r['rl_model'][f'pass@{p}']) for p in passes}
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results to JSON file
    results_json = {
        "summary": {
            "num_total": num_total,
            "total_time_seconds": total_time,
            "base_model": {
                "pass@1": {
                    "num_correct": base_pass_k_correct[1],
                    "accuracy": base_pass_k_correct[1] / num_total
                },
                "pass@5": {
                    "num_correct": base_pass_k_correct[5],
                    "accuracy": base_pass_k_correct[5] / num_total
                },
                "pass@10": {
                    "num_correct": base_pass_k_correct[10],
                    "accuracy": base_pass_k_correct[10] / num_total
                }
            },
            "rl_model": {
                "pass@1": {
                    "num_correct": rl_pass_k_correct[1],
                    "accuracy": rl_pass_k_correct[1] / num_total
                },
                "pass@5": {
                    "num_correct": rl_pass_k_correct[5],
                    "accuracy": rl_pass_k_correct[5] / num_total
                },
                "pass@10": {
                    "num_correct": rl_pass_k_correct[10],
                    "accuracy": rl_pass_k_correct[10] / num_total
                }
            }
        },
        "questions": results
    }
    
    with open(os.path.join(os.path.dirname(f"/home/ubuntu/o1-replication/TTRL/checkpoints/{project_name}"), "results.json"), "w") as f:
        json.dump(results_json, f, indent=4)

if __name__ == "__main__":
    PATH_TO_QUESTIONS = '/home/ubuntu/o1-replication/TTRL/test_trees/ttrl.json'
    PARQUET_OUTPUT_PATH = '/home/ubuntu/o1-replication/TTRL/test_trees_parquet'
    MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"
    # MODEL_DIR = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MAX_NEW_TOKENS = 1024
    PROJECT_NAME = "verl_grpo_ttrl_trees"
    
    main(path_to_questions=PATH_TO_QUESTIONS, parquet_output_path=PARQUET_OUTPUT_PATH, model_dir=MODEL_DIR, max_new_tokens=MAX_NEW_TOKENS, project_name=PROJECT_NAME)
