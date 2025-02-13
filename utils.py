import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
import os

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

    return response

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
    # Define an instruction for the incorrect questions.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before solving, don't include any constants of integration."
    )

    # Loop over each incorrect question.
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
    
    # Define a local output directory and ensure it exists.
    output_dir = '/home/ubuntu/o1-replication/TTRL/variants_set_27'
    os.makedirs(output_dir, exist_ok=True)

    # Save the samples to a JSON file
    import json
    with open(os.path.join(output_dir, f'variants_q{question_id}.json'), 'w') as f:
        json.dump(samples, f, indent=4)
    
    # Save the samples to a Parquet file
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join(output_dir, f'variants_q{question_id}.parquet'))
    
    print(f"Variants saved to {output_dir}/variants_q{question_id}.parquet")


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
