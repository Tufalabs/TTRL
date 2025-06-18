"""
self_improve.py

An experimental pipeline that:
- Accepts an integral (or uses a preset one) and a set of difficulty targets.
- For each requested difficulty (e.g., easier, equivalent, or harder) it generates several variants.
- Each LLM prompt now generates up to 10 variants at once. If the user requests more than 10 variants,
  the work is split into multiple calls.
- Each variant's antiderivative is attempted symbolically via Sympy.
- A second LLM prompt asks for a difficulty evaluation (easier/harder/equivalent) as a double-check.
- The antiderivative solution (i.e. the integration result) is computed, and it is evaluated at three random points.
- Points that produce complex values are skipped.
- Variants judged as "harder" are filtered out when not desired.
- All results are saved to "variants.json".

If the integration (antiderivative computation) takes more than 5 seconds, it is skipped and the variant is
returned with a solution of None (and an empty evaluation dictionary), rather than being thrown out.
"""

import concurrent.futures
import json
import math
import os
import random
import re
from datetime import datetime
from typing import Optional

import sympy as sp

from TTRL.utils import run_inference

TIMEOUT_SECONDS = 1  # Maximum allowed seconds for integration

# New flag: if set to False, we will not compute the symbolic solution
CALCULATE_SYMBOLIC = False  # Set to False to disable symbolic integration computation.

# Create a global process pool executor to reuse worker processes.
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)


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


def verify_integral(integral_str: str) -> bool:
    """
    Verify the integral by checking that the derivative of the antiderivative equals the original integrand.
    Uses symbolic simplification to check for an exact zero difference.
    Note: In this revised version we no longer use this result to drop variants.
    """
    x = sp.symbols("x")
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return False

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception as e:
            print(
                "Integration timed out in verify_integral; returning non-verified"
                " result."
            )
            return False

        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
        return diff_expr == 0
    except Exception as e:
        print("Error verifying integral:", e)
        return False


def compute_solution_and_evals(
    integral_str: str,
    num_points: int = 3,
    lower: float = -10,
    upper: float = 10,
    tol: float = 1e-6,
):
    """
    Given an integral string of the form "integrate(<integrand>, x)", compute the antiderivative (solution)
    and evaluate that solution at up to `num_points` random values of x.
    If the antiderivative cannot be computed within TIMEOUT_SECONDS, returns None and an empty dict.

    Returns:
        solution_str: The antiderivative as a string (or None if not computable or if symbolic integration is disabled).
        evaluations: A dictionary mapping each random x value (rounded) to the numerical evaluation.
    """
    # If symbolic integration is disabled, immediately return None
    if not CALCULATE_SYMBOLIC:
        return None, {}

    x = sp.symbols("x")
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception as e:
            print(
                "Integration timed out in compute_solution_and_evals; marking as too"
                " hard."
            )
            return None, {}
        solution_str = str(antideriv)
        evaluations = {}
        attempts = 0
        max_attempts = num_points * 10
        while len(evaluations) < num_points and attempts < max_attempts:
            attempts += 1
            test_val = random.uniform(lower, upper)
            eval_val = antideriv.evalf(subs={x: test_val})
            if hasattr(eval_val, "as_real_imag"):
                re_val, im_val = eval_val.as_real_imag()
                if abs(im_val) < tol:
                    evaluations[round(test_val, 3)] = float(re_val)
            else:
                evaluations[round(test_val, 3)] = float(eval_val)
        return solution_str, evaluations
    except Exception as e:
        print("Error computing solution/evaluations:", e)
        return None, {}


def parse_variants(text: str) -> list:
    """
    Parse the LLM response text and extract a list of variant dictionaries.
    The expected format for each variant is:

    ====
    Variant <number>:
    Reasoning: <explanation>
    Variant: integrate(<integrand>, x)
    ====
    """
    variants = []
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            reasoning_match = re.search(
                r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL
            )
            variant_match = re.search(r"Variant:\s*(integrate\([^,]+,\s*x\))", block)
            print(f"variant_match: {variant_match}")
            if variant_match:
                variant_expr = variant_match.group(1).strip()
                reasoning_text = (
                    reasoning_match.group(1).strip() if reasoning_match else ""
                )
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants


def process_single_variant(
    model: str,
    output_dir: str,
    original_integral: str,
    difficulty: str,
    variant_data: dict,
) -> dict:
    """
    Process one variant dictionary:
      - Attempt to compute its antiderivative (solution) and numerical evaluations.
        If integration takes too long (or otherwise fails) the solution is set to None.
      - Attempt a verification by differentiating the computed solution if available.
      - Ask the LLM for a difficulty evaluation.
    """
    variant_integral = variant_data.get("variant")
    if not variant_integral:
        return None

    # Always attempt to compute the solution; if the integration is too hard (or disabled), solution will be None.
    solution, evaluations = compute_solution_and_evals(variant_integral)

    # Try to verify the computed solution only if available.
    x = sp.symbols("x")
    verification = None
    pattern = r"integrate\((.+),\s*x\)"
    match = re.search(pattern, variant_integral)
    if match:
        try:
            integrand_str = match.group(1)
            integrand = sp.sympify(integrand_str)
            if solution is not None:
                antideriv = sp.sympify(solution)
                diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
                verification = diff_expr == 0
            else:
                verification = None
        except Exception as e:
            verification = None
    else:
        verification = None

    evaluation = "unknown"
    prompt_evaluation = (
        f"Original integral: {original_integral}\nVariant integral:"
        f" {variant_integral}\nBased on the changes, determine whether the variant is"
        " 'easier', 'harder', or 'equivalent' to the original. Answer with one word:"
        " easier, harder, or equivalent."
    )
    evaluation_response = run_inference(
        prompt=prompt_evaluation, model_dir=model, max_new_tokens=4048, temperature=0.3
    )
    print(f"evaluation_response: {evaluation_response}")
    if evaluation_response:
        evaluation = evaluation_response.strip().split()[0].lower()

    return {
        "original": original_integral,
        "requested_difficulty": difficulty,
        "variant": variant_integral,
        "reasoning": variant_data.get("reasoning"),
        "variant_response": None,
        "verification_passed": verification,
        "evaluation": evaluation,
        "transformations_used": variant_data.get("transformations_used", []),
        "solution": (
            solution
        ),  # Will be None if integration took too long or if CALCULATE_SYMBOLIC is False.
        "evaluations": (
            evaluations
        ),  # Will be {} if integration took too long or if CALCULATE_SYMBOLIC is False.
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def generate_variant_chunk(
    model: str, output_dir: str, integral_str: str, difficulty: str, count: int
) -> list:
    """
    Generate a chunk (up to 10) of variants in a single LLM call.
    The prompt instructs the LLM to produce `count` variants in the specified format.
    After receiving the response, each variant is parsed and then further processed.
    """
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    num_choices = random.choice([3, 5])
    chosen_transforms = random.sample(
        transformations, min(num_choices, len(transformations))
    )
    transforms_text = " and ".join(chosen_transforms)

    personas = [
        "a calculus professor who loves elegant simplifications",
        "a creative mathematician who enjoys unusual substitutions",
        "a student who prefers working with polynomials and rational functions",
        "a theoretical physicist who likes trigonometric and exponential forms",
        "an engineer who favors practical, computational approaches",
        "a number theorist fascinated by prime numbers and rational coefficients",
        "a geometry enthusiast who thinks in terms of geometric transformations",
    ]
    personas_str = ", ".join(personas)

    prompt_variant = (
        "Assume you can adopt various mathematical personas such as"
        f" {personas_str}.\n\nGiven the integral: {integral_str}\nYour task is to"
        f" generate {count} variant(s) that are {difficulty} than the original.\n\n1."
        " Analyze the original integral and identify its key characteristics.\n2."
        f" Consider the following transformation ideas: {transforms_text}. You may use"
        " them or devise your own modifications.\n3. For each variant, provide a brief"
        " reasoning from the perspective of a distinct persona and then present the"
        " variant in valid Python sympy syntax.\n\nReturn your answer in the following"
        " exact format for each variant:\n====\nVariant <number>:\nReasoning: <your"
        " explanation>\nVariant: integrate(<integrand>, x)\n====\nEnsure each variant"
        " is clearly separated by the delimiter '===='."
    )

    response_text = run_inference(
        prompt=prompt_variant, model_dir=model, max_new_tokens=2024, temperature=0.3
    )
    print(f"response_text: {response_text}")
    parsed_variants = parse_variants(response_text)

    for variant in parsed_variants:
        variant["transformations_used"] = chosen_transforms

    processed_variants = []
    for variant in parsed_variants:
        result = process_single_variant(
            model=model,
            output_dir=output_dir,
            original_integral=integral_str,
            difficulty=difficulty,
            variant_data=variant,
        )
        if result is not None:
            processed_variants.append(result)

    return processed_variants


TRANSFORMATIONS_BY_DIFFICULTY = {
    "easier": [
        "remove a complicated term",
        "simplify the denominator",
        "reduce an exponent",
        "change a function to an easier one",
        "lower a coefficient",
        "remove a factor",
        "eliminate a radical",
        "drop a subexpression",
        "simplify a trigonometric component",
        "convert a product to a simpler sum",
        "replace a complex fraction with a simpler one",
        "remove nested functions",
        "reduce the degree of a polynomial",
        "simplify composite trigonometric functions",
        "remove logarithmic terms",
        "eliminate absolute value terms",
        "reduce the number of terms in the expression",
        "replace transcendental functions with simpler algebraic ones",
        "change a function to an easier one",
    ],
    "equivalent": [
        "change a function to an easier one",
        "change a function to a different one",
        "change coefficient values slightly",
        "alter constant terms",
        "modify an exponent slightly",
        "rewrite the integrand in a different form without changing overall complexity",
        "exchange similar functions (e.g., sin to cos)",
        "adjust parameters while keeping the integral equivalent",
        "rearrange the order of terms",
        "use trigonometric identities to rewrite expression",
        "substitute equivalent exponential forms",
        "change variables while maintaining complexity",
        "distribute terms differently",
        "factor common terms differently",
        "rewrite using alternate algebraic forms",
        "swap numerator and denominator with reciprocal",
        "use alternate but equivalent radical forms",
        "rewrite using different logarithmic properties",
        "apply algebraic identities that preserve complexity",
    ],
    "harder": [
        "introduce an additional polynomial factor",
        "increase the exponent",
        "add a non-linear term",
        "include a higher degree term",
        "insert a logarithmic factor",
        "complicate the denominator",
        "introduce a composite trigonometric function",
        "add a product of functions",
        "embed an extra constant factor that makes the expression less trivial",
    ],
}


def process_integral(
    model: str,
    output_dir: str,
    integral_str: str,
    difficulties: list,
    num_variants: int = 3,
) -> list:
    """
    Generate a batch of variants for the given integral and for each difficulty.
    If more than 10 variants are requested per difficulty, the work is split into multiple LLM calls.
    A buffer multiplier is used to allow for duplicates or filtering before trimming to the requested number.
    """
    final_results = []
    seen_variants = set()
    buffer_multiplier = 2

    for difficulty in difficulties:
        total_to_request = num_variants * buffer_multiplier
        num_chunks = math.ceil(total_to_request / 10)
        difficulty_results = []

        for i in range(num_chunks):
            count = (
                10
                if (i < num_chunks - 1)
                else (total_to_request - 10 * (num_chunks - 1))
            )
            chunk_results = generate_variant_chunk(
                model=model,
                output_dir=output_dir,
                integral_str=integral_str,
                difficulty=difficulty,
                count=count,
            )

            for variant in chunk_results:
                variant_expr = variant.get("variant")
                if (
                    variant_expr
                    and variant_expr not in seen_variants
                    and not (
                        variant.get("evaluation", "") == "harder"
                        and difficulty != "harder"
                    )
                ):
                    seen_variants.add(variant_expr)
                    difficulty_results.append(variant)

        final_results.extend(difficulty_results[:num_variants])

    return final_results


def generate_variants_main(
    model_dir: str, output_file: str, output_dir: str, gpus: str = "0,1,2,3,4,5,6,7"
):

    base_integral = "integrate(1/(x**2 - x + 1), x)"
    # difficulties = ["easier", "equivalent", "harder"]
    difficulties = ["easier"]
    print("Processing integral:", base_integral)
    variants = process_integral(
        model=model_dir,
        output_dir=output_dir,
        integral_str=base_integral,
        difficulties=difficulties,
        num_variants=2,
    )

    print("GOT HEREEEEEE")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as outfile:
        json.dump(variants, outfile, indent=4)

    print("GOT HEREEEEEE 2")
    for idx, v in enumerate(variants, start=1):
        print(f"\n--- Variant {idx} ---")
        print("Requested difficulty:", v["requested_difficulty"])
        print("Transformations used:", v["transformations_used"])
        print("Variant integral:", v["variant"])
        print("Verification passed:", v["verification_passed"])
        print("LLM evaluation:", v["evaluation"])
        print("Solution (antiderivative):", v["solution"])
        print("Evaluations at random points:", v["evaluations"])


if __name__ == "__main__":
    generate_variants_main(
        model_dir="meta-llama/Llama-3.2-3B-Instruct",
        gpus="0,1,2,3,4,5,6,7",
        output_file="variants.json",
        output_dir="/home/ubuntu/o1-replication/TTRL/variant_results_test",
    )
