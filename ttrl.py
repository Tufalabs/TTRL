import json
import os
import pandas as pd
from typing import List, Dict
from utils import run_inference, variants_to_parquet

def main(path_to_questions: str, parquet_output_path: str):
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

    with open(path_to_questions, 'r') as f:
        data = json.load(f)
    
    # 2. Loop through each question and put the variants into a parquet file to be ready for the RL    
    for q_id, question_data in enumerate(data):
        variants_to_parquet(data=question_data, output_path=parquet_output_path, question_id=q_id)
        print(f"Saved variants for question {q_id} to {parquet_output_path}/variants_q{q_id}.parquet and {parquet_output_path}/variants_q{q_id}.json")

if __name__ == "__main__":
    PATH_TO_QUESTIONS = '/home/ubuntu/o1-replication/TTRL/test_dataset_27/ttrl27.json'
    PARQUET_OUTPUT_PATH = '/home/ubuntu/o1-replication/TTRL/variants_set_27'
    
    main(path_to_questions=PATH_TO_QUESTIONS, parquet_output_path=PARQUET_OUTPUT_PATH)
