import re
import sympy as sp
def extract_integral(ground_truth: str) -> str:
    return ground_truth[10:-4]

def extract_candidate_solution(solution_str: str, method: str = 'strict') -> str:
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    """

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

    # Filter out candidates that contain the word 'integrate' (in any case)
    if candidate and re.search(r'\bintegrate\b', candidate, re.IGNORECASE):
        return None

    return candidate

def preprocess_candidate_solution(solution: str) -> str:
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

def sympy_correct_formatting(solution: str) -> bool:
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
    
    try:
        candidate = preprocess_candidate_solution(solution)
        candidate_expr = sp.parse_expr(candidate, local_dict=locals_dict)
        candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
        test_result = candidate_func(10)
    except NameError:
        return False
    except SyntaxError:
        return False
    except Exception:
        return False
    return True