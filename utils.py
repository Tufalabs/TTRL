import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
from typing_extensions import Unpack
import os, sys, subprocess
import re
import random
import sympy as sp
import mpmath as mp
import signal
from math_utils import is_equiv, last_boxed_only_string, remove_boxed

def run_inference(
    prompt: str,
    model_dir: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    device: str = "cuda",
    system_prompt: Optional[str] = None
) -> str:
    """
    Run inference using Llama 3.2 3B model with HuggingFace API.
    
    Args:
        prompt: Input text prompt
        model_dir: HuggingFace model directory
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run inference on ("cuda" or "cpu")
        system_prompt: System prompt prefix (optional)
    
    Returns:
        Generated response text
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Format prompt with system prompt if provided
    if system_prompt:
        full_prompt = f"<|system|>{system_prompt}<|assistant|>{prompt}"
    else:
        full_prompt = prompt

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up prompt from response
    response = response[len(full_prompt):].strip()

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response

# Timeout exception for integration evaluation
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Integration evaluation timed out.")

# Set the signal handler (Note: signal.alarm works on Unix-like systems)
signal.signal(signal.SIGALRM, timeout_handler)

def extract_candidate_solution(solution_str: str, method: str = 'strict') -> str:
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    """
    if not solution_str or not isinstance(solution_str, str):
        return None
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    candidate = None
    if method == 'strict':
        try:
            #matches = re.findall(r"\\boxed{(.*?)}", solution_str, re.IGNORECASE | re.DOTALL)
            #candidate = matches[-1].strip() if matches else None
            string_in_last_boxed = last_boxed_only_string(solution_str)
            candidate = remove_boxed(string_in_last_boxed)
        except Exception:
            return None
    else:
        candidate = solution_str.strip()

    # Filter out candidates that contain the word 'integrate' (in any case)
    if candidate and re.search(r'\bintegrate\b', candidate, re.IGNORECASE):
        return None

    return candidate

def preprocess_candidate_solution(solution: str) -> str:
    """
    Preprocesses a solution string to remove common LaTeX delimiters and extraneous terms.
    Returns "0" if the solution is empty or not a string.
    """
    if not solution or not isinstance(solution, str):
        return "0"  # Return a safe default that will parse to 0
        
    try:
        # Remove LaTeX delimiters and dollar signs.
        solution = solution.replace(r"\(", "").replace(r"\)", "")
        solution = solution.replace("$", "")
        # Replace some common LaTeX commands with sympy-compatible ones.
        solution = solution.replace("\\arctan", "atan")
        solution = solution.replace("\\arccos", "acos")
        solution = solution.replace("\\arcsin", "asin")
        solution = solution.replace("\\sqrt", "sqrt")
        solution = solution.replace("\\ln", "log")
        solution = solution.replace("arctan", "atan")
        solution = solution.replace("arccos", "acos")
        solution = solution.replace("arcsin", "asin")
        solution = solution.replace("^", "**")

        # Replace e** notation with exp()
        solution = re.sub(r'e\*\*([^*]+)', r'exp(\1)', solution)
        # Remove any trailing "+ C" or similar constant expressions.
        solution = re.sub(r"\+?\s*C\b", "", solution)
        return solution.strip() or "0"  # Return "0" if empty after processing
    except Exception:
        return "0"

def verify_response(response: str, ground_truth: str, num_tests: int = 5, timeout_secs: int = 10, tol: float = 1e-2) -> bool:
    """
    Verify the response is correct by comparing it to the question.
    """
    candidate = extract_candidate_solution(response, method='strict')
    if not candidate:
        return False

    candidate = preprocess_candidate_solution(candidate)
    ground_truth_processed = preprocess_candidate_solution(ground_truth)
    
    x = sp.symbols('x')
    locals_dict = {
        'x': x,
        'C': 0,
        'integrate': sp.integrate,
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'exp': sp.exp
    }
    
    try:
        candidate_expr = sp.parse_expr(candidate, local_dict=locals_dict)
        # Extract the integrand from ground_truth by removing 'integrate(' and splitting at the comma.
        integrand_str = ground_truth_processed.replace('integrate(', '').split(',')[0]
        integrand_expr = sp.parse_expr(integrand_str, local_dict=locals_dict)
        
        # Create lambda functions for numerical evaluation.
        candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
        integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
        
        is_correct = True
        successful_tests = 0
        for test_num in range(num_tests):
            a_val = random.uniform(-10, 10)
            b_val = random.uniform(-10, 10)
          
            if abs(b_val - a_val) < 1e-3:
                # Skip tests where the evaluation points are too close.
                continue
                
            try:
                # Set an alarm for the timeout.
                signal.alarm(timeout_secs)
                candidate_diff = candidate_func(b_val) - candidate_func(a_val)
                definite_integral = mp.quad(integrand_func, [a_val, b_val])
                # Cancel the alarm.
                signal.alarm(0)
                
                if abs(candidate_diff - definite_integral) > tol:
                    is_correct = False
                    break
                successful_tests += 1
            except TimeoutException as te:
                print(f"Test {test_num + 1}: Timeout during evaluation: {te}")
                signal.alarm(0)
                continue  # Skip this test and try the next one.
            except Exception as e:
                signal.alarm(0)
                print(f"Test {test_num + 1}: Error during evaluation: {str(e)}")
                continue
        
        if not is_correct:
            return False
    
        return successful_tests > 0
    except Exception as e:
        print(f"Error during computation: {str(e)}")
        return False

def variants_to_parquet(data, output_path: str, question_id: str) -> None:
    """
    Expects data in the following format:
    
    {
        "question": "integrate(1/(x**2 - x + 1), x)",
        "variants": [
            {
                "variant": "integrate(1/(x**2 - x + 1), x)",
                "reasoning": "The integral can be transformed using the substitution u = x - 1/2, which simplifies the denominator.",
                "difficulty": "easier"
            }
        ]
    }
    """
    
    samples = []
    test_samples = []
    # Define an instruction for the incorrect questions.
    instruction_following = "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer> wrapped in a <answer> tags. Importantly, put * between terms you want to multiply! Show your full working out before solving, don't include any constants of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags."

    # Loop over each question.
    for idx, question in enumerate(data["variants"]):
        # Build the prompt by combining the question with the instruction.
        prompt_content = f"{question['variant']}\n{instruction_following}"
        
        # Build a sample dictionary
        sample = {
            "data_source": "integration_numeric",
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": question["variant"]
            },
            "extra_info": {
                "question_index": idx,
                "question_id": question["variant"]
            }
        }
        samples.append(sample)

    # Create test samples using the base question
    base_prompt = f"{data['question']}\n{instruction_following}"
    for i in range(16):
        test_sample = {
            "data_source": "integration_numeric",
            "prompt": [{
                "role": "user", 
                "content": base_prompt
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['question']
            },
            "extra_info": {
                "question_index": i,
                "question_id": data['question']
            }
        }
        test_samples.append(test_sample)
    
    # Define a local output directory and ensure it exists.
    os.makedirs(output_path, exist_ok=True)

    # Save the samples to JSON files
    import json
    with open(os.path.join(output_path, f'variants_q{question_id}.json'), 'w') as f:
        json.dump(samples, f, indent=4)
    with open(os.path.join(output_path, f'test_q{question_id}.json'), 'w') as f:
        json.dump(test_samples, f, indent=4)
    
    # Save the samples to Parquet files
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join(output_path, f'variants_q{question_id}.parquet'))
    
    test_df = pd.DataFrame(test_samples)
    test_df.to_parquet(os.path.join(output_path, f'test_q{question_id}.parquet'))
    
    print(f"Variants saved to {output_path}/variants_q{question_id}.parquet")
    print(f"Test samples saved to {output_path}/test_q{question_id}.parquet")

def run_rl(model_dir: str, train_parquet_path: str, test_parquet_path: str, project_name: str, experiment_name: str, max_new_tokens: int):
    """
    Run RL using CustomTinyZero and the numerical_integration reward function
    """

    # Set environment variable
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

    # Define paths and configuration
    base_model = model_dir

    # Create checkpoint directory
    checkpoint_dir = f"/home/ubuntu/test/TTRL/checkpoints/{project_name}/{experiment_name}"

    os.makedirs(checkpoint_dir, exist_ok=True, mode=0o777)
    log_file = os.path.join(checkpoint_dir, "logfile.txt")

    # Copy current script to experiment directory
    import shutil
    shutil.copy2(__file__, os.path.join(checkpoint_dir, os.path.basename(__file__)))

    # Build command arguments - removed 'cd' and '&&' and use shell=True instead
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator=grpo",
        f"data.train_files={train_parquet_path}",
        f"data.val_files={test_parquet_path}",
        "data.train_batch_size=256",
        "data.val_batch_size=16", 
        "data.max_prompt_length=2048",
        f"data.max_response_length={max_new_tokens}",
        f"actor_rollout_ref.model.path={base_model}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size=16",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
        "critic.ppo_max_token_len_per_gpu=2048",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=2",
        "actor_rollout_ref.ref.ulysses_sequence_parallel_size=2",
        "critic.ulysses_sequence_parallel_size=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.grad_offload=False", 
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.critic_warmup=0",
        "trainer.logger=['console','wandb']",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=5",
        "trainer.test_freq=10",
        f"trainer.default_hdfs_dir={checkpoint_dir}",
        f"trainer.default_local_dir={checkpoint_dir}",
        "trainer.total_epochs=8",
        "actor_rollout_ref.actor.optim.total_training_steps=50"
    ]

    # Run the command from the correct directory and tee output to log file
    with open(log_file, 'a') as f:
        process = subprocess.Popen(
            ' '.join(cmd),
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            shell=True,
            cwd="/home/ubuntu/test/CustomTinyZero"  # Set working directory here
        )
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            f.buffer.write(line)
    
    last_checkpoint_path = os.path.join(checkpoint_dir, "actor")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return last_checkpoint_path
    

def generate_variants(question: Dict, output_dir: str):
    """
    An experimental pipeline that:
    - Accepts an integral (or uses a preset one) and a set of difficulty targets.
    - For each requested difficulty (e.g., easier, equivalent, or harder) it generates several variants.
    - Each LLM prompt now generates up to 10 variants at once. If the user requests more than 10 variants,
    the work is split into multiple calls.
    - A second LLM prompt asks for a difficulty evaluation (easier/harder/equivalent) as a double-check.
    - Variants judged as "harder" are filtered out when not desired.
    - All results are saved to "variants.json".

    To be implemented.
    """

    pass

if __name__ == "__main__":
    prompt = "<|system|>: Assume you can adopt various mathematical personas such as a calculus professor who loves elegant simplifications, a creative mathematician who enjoys unusual substitutions, a student who prefers working with polynomials and rational functions, a theoretical physicist who likes trigonometric and exponential forms, an engineer who favors practical, computational approaches, a number theorist fascinated by prime numbers and rational coefficients, a geometry enthusiast who thinks in terms of geometric transformations.\n\nGiven the integral: integrate(1/(x**2 - x + 1), x)\nYour task is to generate 4 variant(s) that are equivalent than the original.\n\n1. Analyze the original integral and identify its key characteristics.\n2. Consider the following transformation ideas: apply algebraic identities that preserve complexity and rewrite the integrand in a different form without changing overall complexity and distribute terms differently. You may use them or devise your own modifications.\n3. For each variant, provide a brief reasoning from the perspective of a distinct persona and then present the variant in valid Python sympy syntax.\n\nReturn your answer in the following exact format for each variant:\n====\nVariant <number>:\nReasoning: <your explanation>\nVariant: integrate(<integrand>, x)\n====\nEnsure each variant is clearly separated by the delimiter '===='. <|assistant|>"
    response = run_inference(prompt=prompt, system_prompt=None)
    print(response)
