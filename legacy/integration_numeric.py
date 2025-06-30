"""
Code taken from the repo: github.com/tamassimonds/CustomTinyZero
In the file verl/utils/reward_score/integration_numeric.py.
Just slightly modified to add status of the rewards.
"""
import re
import random
import sympy as sp
import mpmath as mp
import signal
from enum import Enum, auto

class FormalStatus(Enum):
    OK = auto()
    NO_OK = auto()
    NO_SOL_OR_NO_GT = auto()
    NO_INP_IN_TAGS = auto()
    ERROR = auto()



# Timeout exception for integration evaluation
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Integration evaluation timed out.")

# Set the signal handler (Note: signal.alarm works on Unix-like systems)
signal.signal(signal.SIGALRM, timeout_handler)

def extract_candidate_solution(solution_str: str, method: str = 'strict') -> str:
    solution_str = solution_str.split("</instruction>")[-1] if "</instruction>" in solution_str else solution_str
    if not solution_str or not isinstance(solution_str, str):
        return None
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    candidate = None
    if method == 'strict':
        try:
            matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
            candidate = matches[-1].strip() if matches else None
        except Exception:
            return None
    else:
        candidate = solution_str.strip()

    if candidate and re.search(r'\bintegrate\b', candidate, re.IGNORECASE):
        return None

    return candidate

def preprocess_candidate_solution(solution: str) -> str:
    if not solution or not isinstance(solution, str):
        return "0"
        
    try:
        # Remove ellipsis patterns: remove any "* ..." as well as standalone "..."
        solution = re.sub(r'\*\s*\.\.\.', '', solution)
        solution = solution.replace("...", "")
        solution = solution.replace(r"\(", "").replace(r"\)", "")
        solution = solution.replace("$", "")
        solution = solution.replace("\\arctan", "atan")
        solution = solution.replace("\\arcsin", "asin")
        solution = solution.replace("\\arccos", "acos")
        solution = solution.replace("arccos", "acos")
        solution = solution.replace("arcsin", "asin")
        solution = solution.replace("arctan", "atan")
        solution = solution.replace("e^", "2.718**")
        solution = solution.replace("^", "**")
        solution = solution.replace("\\ln", "log")
        solution = re.sub(r'e\*\*([^*]+)', r'exp(\1)', solution)
        solution = re.sub(r"\+?\s*C\b", "", solution)
        return solution.strip() or "0"
    except Exception:
        return "0"

def is_close_to_ground_truth(candidate_val, ground_truth_val, tol):
    try:
        # Convert to float for numerical comparison
        candidate_float = float(candidate_val)
        ground_truth_float = float(ground_truth_val)
        
        # Use relative tolerance instead of absolute
        return abs((candidate_float - ground_truth_float) / ground_truth_float) <= tol
    except (TypeError, ValueError, ZeroDivisionError):
        return False

def compute_score(solution_str: str,
                  ground_truth: str,
                  method: str = 'strict',
                  tol: float = 1e-2,
                  score: float = 1.0,
                  format_score: float = 0.05,
                  num_tests: int = 3,
                  timeout_secs: int = 1,
                  max_response_length: int = None,
                  tokenizer = None):
    # print("--------------------------------")
    
    if not solution_str or not ground_truth:
        return 0.0, FormalStatus.NO_SOL_OR_NO_GT

    candidate = extract_candidate_solution(solution_str, method=method)
    if not candidate:
        return 0.0, FormalStatus.NO_INP_IN_TAGS

    candidate = preprocess_candidate_solution(candidate)
    ground_truth_processed = preprocess_candidate_solution(ground_truth)
    
    x, k = sp.symbols('x k')
    locals_dict = {
        'x': x,
        'k': k,
        'C': 0,
        'integrate': sp.integrate,
        'Sum': sp.Sum,   # Use Sum for symbolic summation.
        'sum': sp.Sum,   # Allow 'sum' to be an alias.
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'exp': sp.exp,
        'sqrt': sp.sqrt,
        'atan': sp.atan,
        'asin': sp.asin,
        'acos': sp.acos
    }

    if random.random() < 0.1:
        pass
        # print(f"solution_str: {solution_str}")
        # print(f"ground_truth: {ground_truth}")
        # print(f"candidate: {candidate}")
        # print(f"ground_truth_processed: {ground_truth_processed}")
    
    try:
        candidate_expr = sp.parse_expr(candidate, local_dict=locals_dict)
        
        # Detect if ground_truth represents a definite integral
        definite_match = re.search(r",\s*\(x,\s*([^,]+),\s*([^)]+)\)", ground_truth_processed)
        if definite_match:
            # Definite integral detected: extract limits
            lower_limit_str = definite_match.group(1)
            upper_limit_str = definite_match.group(2)
            lower_limit = sp.sympify(lower_limit_str, locals=locals_dict)
            upper_limit = sp.sympify(upper_limit_str, locals=locals_dict)
            
            # Extract integrand (everything before the comma)
            integrand_str = ground_truth_processed.replace('integrate(', '').split(',', 1)[0]
            # Fix summation if needed (insert dummy variable limits if missing)
            if integrand_str.strip().lower().startswith("sum("):
                content = integrand_str.strip()[len("sum("):-1]
                if ",(" not in content:
                    integrand_str = f"Sum({content}, (k, 0, 999))"
                    # Note: "oo" is recognized by sympy as infinity.
                    locals_dict['oo'] = sp.oo
            
            integrand_expr = sp.parse_expr(integrand_str, local_dict=locals_dict)
            
            # Compute the definite integral using sympy
            computed_def_int = sp.integrate(integrand_expr, (x, lower_limit, upper_limit))
            # Evaluate candidate expression and computed integral numerically
            candidate_val = sp.N(candidate_expr)
            computed_val = sp.N(computed_def_int)
            
            # print(f"Candidate value: {candidate_val}, Computed definite integral: {computed_val}")
            
            # Commented for reference.
            # final_score = (score + format_score) if is_close_to_ground_truth(candidate_val, computed_val, tol) else format_score
            if is_close_to_ground_truth(candidate_val, computed_val, tol):
                return score + format_score, FormalStatus.OK
            else:
                return format_score, FormalStatus.NO_OK
        else:
            # Indefinite integral: process as before.
            # Extract the integrand from ground_truth by removing 'integrate(' and splitting at the comma.
            integrand_str = ground_truth_processed.replace('integrate(', '').split(',')[0]
            # Fix summation if needed (insert dummy variable limits if missing)
            if integrand_str.strip().lower().startswith("sum("):
                content = integrand_str.strip()[len("sum("):-1]
                if ",(" not in content:
                    integrand_str = f"Sum({content}, (k, 0, oo))"
                    locals_dict['oo'] = sp.oo
            
            integrand_expr = sp.parse_expr(integrand_str, local_dict=locals_dict)
            
            candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
            integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
            
            # Check for functions that require a positive domain.
            if any(fn in ground_truth_processed or fn in candidate for fn in ["sqrt(", "log("]):
                domain_low, domain_high = 0.1, 10
            else:
                domain_low, domain_high = -10, 10

            is_correct = True
            successful_tests = 0
            for test_num in range(num_tests):
                a_val = random.uniform(domain_low, domain_high)
                b_val = a_val + 0.1
              
                if abs(b_val - a_val) < 1e-3:
                    continue
                    
                try:
                    signal.alarm(timeout_secs)
                    candidate_diff = candidate_func(b_val) - candidate_func(a_val)
                    definite_integral = mp.quad(integrand_func, [a_val, b_val])
                    signal.alarm(0)
                    
                    if not is_close_to_ground_truth(candidate_diff, definite_integral, tol):
                        is_correct = False
                        break
                    successful_tests += 1
                except TimeoutException as te:
                    # print(f"Test {test_num + 1}: Timeout during evaluation: {te}")
                    signal.alarm(0)
                    continue
                except Exception as e:
                    signal.alarm(0)
                    # print(f"Test {test_num + 1}: Error during evaluation: {str(e)}")
                    continue
            
            # Commented for reference.
            # final_score = (score + format_score) if (is_correct and successful_tests > 0) else format_score
            if is_correct and successful_tests > 0:
                return score + format_score, FormalStatus.OK
            else:
                return format_score, FormalStatus.NO_OK
    except Exception as e:
        print(f"Error during computation: {str(e)}")
        return 0.0, FormalStatus.ERROR

# Example usage:
if __name__ == "__main__":
    # Example 1: Integral with potential singularity
    # candidate_solution = "<answer>2*sqrt(x)</answer>"
    # ground_truth_solution = "integrate(1/sqrt(x), x)"
    # result = compute_score(candidate_solution, ground_truth_solution, tol=1e-2)
    # print("Reward:", result)
    
    # # Example 2: A definite integral
    # candidate_solution2 = "<answer>-1+exp(1)</answer>"
    # ground_truth_solution2 = "integrate(exp(x), (x, 0, 1))"
    # result2 = compute_score(candidate_solution2, ground_truth_solution2, tol=1e-2)
    # print("Reward:", result2)

    # candidate_solution3 = "<answer>-cos(x)</answer>"
    # ground_truth_solution3 = "integrate(sin(x), x)"
    # result3 = compute_score(candidate_solution3, ground_truth_solution3, tol=1e-2)
    # print("Reward:", result3)
    
    # # Example 4: Polynomial with definite integral
    # candidate_solution4 = "<answer>7</answer>"
    # ground_truth_solution4 = "integrate(3*x**2, (x, 1, 2))"
    # result4 = compute_score(candidate_solution4, ground_truth_solution4, tol=1e-2)
    # print("Reward:", result4)
    
    # # Example 5: Natural logarithm
    # candidate_solution5 = "<answer>x*log(x) - x</answer>"
    # ground_truth_solution5 = "integrate(log(x), x)"
    # result5 = compute_score(candidate_solution5, ground_truth_solution5, tol=1e-2)
    # print("Reward:", result5)
    
    # # Example 6: Complex trig function with definite integral
    # candidate_solution6 = "<answer>pi/4</answer>"
    # ground_truth_solution6 = "integrate(sin(x)**2, (x, 0, pi/2))"
    # result6 = compute_score(candidate_solution6, ground_truth_solution6, tol=1e-2)
    # print("Reward:", result6)
    
    # # Example 7: Exponential function
    # candidate_solution7 = "<answer>exp(x)/2</answer>"
    # ground_truth_solution7 = "integrate(exp(x)/2, x)"
    # result7 = compute_score(candidate_solution7, ground_truth_solution7, tol=1e-2)
    # print("Reward:", result7)

    # # Example 8: Sum in definite integral (fixed)
    # candidate_solution8 = "<answer>pi/4</answer>"
    # ground_truth_solution8 = "integrate(sum((-1)**k*x**(2*k)), (x, 0, 1))"
    # result8 = compute_score(candidate_solution8, ground_truth_solution8, tol=1e-2)
    # print("Reward:", result8)
    
    # # Example 9: Function with ellipsis in the integrand
    # candidate_solution9 = "<answer>1/2*arctan(x**2*exp(2*x))</answer>"
    # ground_truth_solution9 = "integrate(x**(1/3) * (x**(1/4) * (x**(1/5) * (x**(1/6) * ...))), x)"
    # result9 = compute_score(candidate_solution9, ground_truth_solution9, tol=1e-2)
    # print("Reward:", result9)

    candidate_solution = "integrate(1 / sec(x)**2, x)\n<instruction> Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer> wrapped in a <answer> tags. Importantly, put * between terms you want to multiply! Show your full working out before solving, don't include any constants of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags. </instruction> \n<working>\n\\[\n\\int \\frac{1}{\\sec^2(x)} dx = \\int \\cos^2(x) dx\n\\]\nTo integrate \\(\\cos^2(x)\\), we use the trigonometric identity:\n\\[\n\\cos^2(x) = \\frac{1 + \\cos(2x)}{2}\n\\]\nSo the integral becomes:\n\\[\n\\int \\cos^2(x) dx = \\int \\frac{1 + \\cos(2x)}{2} dx\n\\]\nThis can be split into two integrals:\n\\[\n\\int \\frac{1 + \\cos(2x)}{2} dx = \\frac{1}{2} \\int 1 dx + \\frac{1}{2} \\int \\cos(2x) dx\n\\]\nThe first integral is:\n\\[\n\\frac{1}{2} \\int 1 dx = \\frac{1}{2} x\n\\]\nFor the second integral, we use the substitution \\(u = 2x\\), so \\(du = 2 dx\\):\n\\[\n\\frac{1}{2} \\int \\cos(2x) dx = \\frac{1}{2} \\cdot \\frac{1}{2} \\int \\cos(u) du = \\frac{1}{4} \\sin(u) = \\frac{1}{4} \\sin(2x)\n\\]\nCombining these results:\n\\[\n\\frac{1}{2} x + \\frac{1}{4} \\sin(2x)\n\\]\nSince \\(\\sin(2x) = 2 \\sin(x) \\cos(x)\\), the expression can be written as:\n\\[\n\\frac{1}{2} x + \\frac{1}{2} \\sin(x) \\cos(x)\n\\]\nSo the final antiderivative is:\n</working> \n<answer>1/2*x + 1/2*sin(x)*cos(x)</answer><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
    ground_truth_solution = "integrate(sec(x)**(-2), x)"
    result = compute_score(candidate_solution, ground_truth_solution, tol=1e-2)
    print("Reward:", result)
