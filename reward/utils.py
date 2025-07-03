import random
import re
from enum import Enum
from enum import auto

import mpmath as mp
import sympy as sp

from data.format_for_ttrl import SYMBOLS_DICT
from data.format_for_ttrl import clean_expr
from data.format_for_ttrl import timeout


class FormalStatus(Enum):
    OK = auto()
    NO_OK = auto()
    NO_INP_IN_TAGS = auto()
    NO_SYMPY_EXPR = auto()
    ERROR_EVAL_EXPR = auto()
    ERROR = auto()


def extract_candidate_solution(solution_str: str, method: str = "strict") -> str:
    solution_str = (
        solution_str.split("</instruction>")[-1]
        if "</instruction>" in solution_str
        else solution_str
    )
    if not solution_str or not isinstance(solution_str, str):
        return None

    assert method in ["strict", "flexible"], "Method must be 'strict' or 'flexible'"
    candidate = None
    if method == "strict":
        try:
            matches = re.findall(
                r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL
            )
            candidate = matches[-1].strip() if matches else None
        except Exception:
            return None
    else:
        candidate = solution_str.strip()

    if candidate and re.search(r"\bintegrate\b", candidate, re.IGNORECASE):
        return None

    return candidate


def is_close_to_ground_truth(candidate_val, ground_truth_val, tol):
    try:
        # Convert to float for numerical comparison
        candidate_float = float(candidate_val)
        ground_truth_float = float(ground_truth_val)

        # Use relative tolerance instead of absolute
        return abs((candidate_float - ground_truth_float) / ground_truth_float) <= tol
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _parse_gt_expr(ground_truth: str, gt_var: str):
    gt_var_sp = sp.symbols(gt_var)
    try:
        integrand_expr = sp.parse_expr(
            ground_truth, local_dict=SYMBOLS_DICT | {gt_var: gt_var_sp}
        )
    except Exception:
        # This should never happen, so we raise an error.
        raise ValueError("Variant is not valid")

    return integrand_expr, gt_var_sp


def compute_score_indefinite(
    candidate: str | None,
    ground_truth: str,
    gt_var: str,
    tol: float = 1e-2,
    score: float = 1.0,
    format_score: float = 0.05,
    num_tests: int = 10,
    max_timeout: float = 1,
):
    if not candidate:
        return 0.0, FormalStatus.NO_INP_IN_TAGS

    candidate = clean_expr(candidate)

    # HACK: We assume that x is the variable of integration
    try:
        _x = sp.symbols("x")
        candidate_expr = sp.parse_expr(candidate, local_dict=SYMBOLS_DICT | {"x": _x})
        candidate_func = sp.lambdify(_x, candidate_expr, "mpmath")
        positive_domain = candidate_expr.has(sp.log) or candidate_expr.has(sp.sqrt)
    except Exception:
        return format_score, FormalStatus.NO_SYMPY_EXPR

    integrand_expr, gt_var_sp = _parse_gt_expr(ground_truth, gt_var)
    integrand_func = sp.lambdify(gt_var_sp, integrand_expr, "mpmath")
    gt_positive_domain = integrand_expr.has(sp.log) or integrand_expr.has(sp.sqrt)

    if positive_domain or gt_positive_domain:
        domain_low, domain_high = 0.1, 10
    else:
        domain_low, domain_high = -10, 10

    is_correct = True
    successful_tests = 0
    for _ in range(num_tests):
        a_val = random.uniform(domain_low, domain_high)
        b_val = a_val + 0.1

        if abs(b_val - a_val) < 1e-3:
            continue

        try:
            with timeout(max_timeout):
                candidate_diff = candidate_func(b_val) - candidate_func(a_val)
                definite_integral = mp.quad(integrand_func, [a_val, b_val])

            if not is_close_to_ground_truth(candidate_diff, definite_integral, tol):
                is_correct = False
                break

            successful_tests += 1
        except TimeoutError:
            continue
        except Exception:
            continue

    if is_correct and successful_tests > 0:
        return score + format_score, FormalStatus.OK
    else:
        return format_score, FormalStatus.NO_OK


def compute_score_definite(
    candidate: str | None,
    ground_truth: str,
    lb: float,
    ub: float,
    gt_var: str,
    tol: float = 1e-2,
    score: float = 1.0,
    format_score: float = 0.05,
):

    if not candidate:
        return 0.0, FormalStatus.NO_INP_IN_TAGS

    candidate = clean_expr(candidate)

    # HACK: We assume that x is the variable of integration
    try:
        candidate_expr_scalar = sp.parse_expr(
            candidate, local_dict=SYMBOLS_DICT | {"x": sp.symbols("x")}
        )
    except Exception:
        return format_score, FormalStatus.NO_SYMPY_EXPR

    integrand_expr, gt_var_sp = _parse_gt_expr(ground_truth, gt_var)
    gt_lambda = sp.lambdify(gt_var_sp, integrand_expr, "mpmath")
    computed_def_int = mp.quad(gt_lambda, [lb, ub])

    # Evaluate candidate expression and computed integral numerically
    try:
        candidate_val = sp.N(candidate_expr_scalar)
    except Exception as e:
        print(computed_def_int, candidate, e)
        return format_score, FormalStatus.ERROR_EVAL_EXPR

    ground_truth_val = sp.N(computed_def_int)

    if is_close_to_ground_truth(candidate_val, ground_truth_val, tol):
        return score + format_score, FormalStatus.OK
    else:
        return format_score, FormalStatus.NO_OK
