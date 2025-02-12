import os
import base_datasets
import sympy as sp
from base_datasets.mit_bee_qualifiying_questions import BASE_QUESTIONS as QUESTIONS
import concurrent.futures

def extract_integrand(integral_text: str) -> str:
    """Extract the integrand from an integration problem text."""
    import re
    integrand_pattern = r"integrate\((.+),\s*x\)"
    m = re.search(integrand_pattern, integral_text)
    if not m:
        raise ValueError("Could not extract the integrand from the provided integral text.")
    return m.group(1)

# Create a global process pool executor to reuse worker processes
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
TIMEOUT_SECONDS = 3  # Maximum allowed seconds for integration

def integrate_wrapper(integrand, x):
    """
    A top-level function to compute sp.integrate(integrand, x).
    This function is picklable and can be used with the executor.
    """
    return sp.integrate(integrand, x)

def run_integration(integrand, x, timeout=TIMEOUT_SECONDS):
    """
    Run sp.integrate(integrand, x) using a persistent process pool.
    If it takes longer than 'timeout' seconds, a TimeoutError is raised.
    """
    future = executor.submit(integrate_wrapper, integrand, x)
    return future.result(timeout=timeout)

if __name__ == '__main__':
    samples = []
    # Define an instruction for the incorrect questions.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before solving, don't include any constants of integration."
    )
    
    x = sp.symbols('x')
    
    # Loop over each incorrect question.
    for idx, question in enumerate(QUESTIONS):
        # Build the prompt by combining the question with the instruction.
        prompt_content = f"{question}\n{instruction_following}"
        
        # Compute the ground truth using sympy
        try:
            integrand_expr = sp.sympify(extract_integrand(question))
            try:
                computed_expr = run_integration(integrand_expr, x)
                if computed_expr is None:
                    computed_expr = run_integration(integrand_expr, x, meijerg=True)
                ground_truth = str(computed_expr) if computed_expr is not None else ""
                print(f"Ground truth for question {idx}: {ground_truth}")
            except concurrent.futures.TimeoutError:
                print(f"Warning: Integration timed out for question {idx}")
                ground_truth = ""
            except Exception as e:
                print(f"Warning: Integration failed for question {idx}: {e}")
                ground_truth = ""
        except Exception as e:
            print(f"Warning: Could not compute ground truth for question {idx}: {e}")
            ground_truth = ""
        
        # Build a sample dictionary
        sample = {
            "data_source": "integration",
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "question_index": idx,
                "question_id": question
            }
        }
        samples.append(sample)
    
    # Define a local output directory and ensure it exists.
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the samples to a JSON file
    import json
    with open(os.path.join(output_dir, 'integration_test.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Save the samples to a Parquet file
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join(output_dir, 'integration_test.parquet'))
    
    print("Incorrect questions dataset created and saved to JSON and Parquet successfully.")
