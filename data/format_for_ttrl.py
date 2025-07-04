import json
import math
import os
import re
import signal
import sys
import warnings
from collections import deque
from contextlib import contextmanager
from enum import Enum
from enum import auto
from typing import Any
from typing import Deque
from typing import Dict
from typing import List
from typing import NamedTuple

import mpmath as mp
import sympy as sp
from sympy import parse_expr
from sympy import simplify

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the "
    "reasoning process in the mind and then provides the user with the answer"
)


TASK_PROMPT_INDEFINITE = (
    "Solve the following indefinite integral: integrate({integrand}, {var}). Provide"
    " ONLY your antiderivative as a valid Python sympy expression e.g "
    " <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer> wrapped in a <answer> tags."
    " Importantly, put * between terms you want to multiply! Show your full working out"
    " before solving, don't include any constants of integration. DO NOT OUTPUT IN"
    " LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags."
)

TASK_PROMPT_DEFINITE = (
    "Solve the following definite integral: integrate({integrand}, ({var}, {lb},"
    " {ub})). Provide ONLY your response as a valid sympy expression or numeric value"
    " e.g <answer>sqrt(pi)</answer> wrapped in a <answer> tags. Importantly, put *"
    " between terms you want to multiply! Show your full working out before solving,"
    " don't include any constants of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT"
    " IN SYMPY in <answer> tags."
)


SYMBOLS_DICT = {
    "k": sp.symbols("k"),
    "C": 0,
    "integrate": sp.integrate,
    "Sum": sp.Sum,  # Use Sum for symbolic summation.
    "sum": sp.Sum,  # Allow 'sum' to be an alias.
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "atan": sp.atan,
    "asin": sp.asin,
    "acos": sp.acos,
}


class Variant(NamedTuple):
    integrand: str
    var: str
    lb: float
    ub: float
    is_definite: bool
    level: int = -1


class FlatTree(NamedTuple):
    base_question: str
    tree_id: str
    variants: list[Variant]


class IntegralKind(Enum):
    """Kinds of integrals accepted by `classify_integral_call`."""

    INDEFINITE = auto()  # integrand(exp, x)
    DEFINITE_SEPARATE = auto()  # integrand(exp, x, 2, 4)
    DEFINITE_TUPLE = auto()  # integrand(exp, (x, 2, 4))
    ERROR = auto()


def _smart_split(arg_string: str) -> list[str]:
    """
    Split on top-level commas only (ignore commas inside nested parentheses).

    Example: "f(x) + g(y), (t, 0, 1)"  ->  ["f(x) + g(y)", "(t, 0, 1)"]
    """
    parts: List[str] = []
    depth = 0
    current: List[str] = []

    for ch in arg_string:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1

        if ch == "," and depth == 0:  # top-level comma
            parts.append("".join(current).strip())
            current.clear()
        else:
            current.append(ch)

    if current:
        parts.append("".join(current).strip())

    return parts


def classify_integral_call(call: str):
    """
    Classify an `integrate(...)` call string into one of the IntegralKind values.

    Parameters
    ----------
    call : str
        The literal text of the call, e.g. "integrate(f(x), x, 0, 1)".

    Returns
    -------
    IntegralKind
        The recognised kind of integral.
    """
    call = call.strip()
    if not call.startswith("integrate"):
        return [], IntegralKind.ERROR

    try:
        open_paren = call.index("(")
        close_paren = call.rindex(")")
    except ValueError:
        return [], IntegralKind.ERROR

    arg_section = call[open_paren + 1 : close_paren]
    args = _smart_split(arg_section)

    if len(args) == 2:
        maybe_tuple = args[1]
        if maybe_tuple.startswith("(") and maybe_tuple.endswith(")"):
            inner_parts = [p.strip() for p in maybe_tuple[1:-1].split(",")]
            if len(inner_parts) == 3:
                return [args[0]] + inner_parts, IntegralKind.DEFINITE_TUPLE
            else:
                return args, IntegralKind.ERROR
        if len(maybe_tuple) == 1:
            return args, IntegralKind.INDEFINITE

    if len(args) == 4:
        return args, IntegralKind.DEFINITE_SEPARATE

    return args, IntegralKind.ERROR


def to_verl_format(variant: Variant, index: int, tree_id: str):
    if variant.is_definite:
        query = TASK_PROMPT_DEFINITE.format(
            integrand=variant.integrand, var=variant.var, lb=variant.ub, ub=variant.ub
        )
    else:
        query = TASK_PROMPT_INDEFINITE.format(
            integrand=variant.integrand, var=variant.var
        )

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    data = {
        "data_source": "integration_numeric",
        "prompt": prompt,
        "ability": "integration",
        "reward_model": {"style": "rule", "ground_truth": variant.integrand},
        "extra_info": {
            "question_index": index,
            "tree_id": tree_id,
            "var": variant.var,
            "lb": variant.lb,
            "ub": variant.ub,
            "level": variant.level,
            "is_definite": variant.is_definite,
        },
    }

    return data


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.alarm(0)


def clean_expr(integrand: str):
    integrand = re.sub(r"\*\s*\.\.\.", "", integrand)
    integrand = integrand.replace("...", "")
    integrand = integrand.replace(r"\(", "").replace(r"\)", "")
    integrand = integrand.replace("$", "")
    integrand = integrand.replace("\\arctan", "atan")
    integrand = integrand.replace("\\arcsin", "asin")
    integrand = integrand.replace("\\arccos", "acos")
    integrand = integrand.replace("arccos", "acos")
    integrand = integrand.replace("arcsin", "asin")
    integrand = integrand.replace("arctan", "atan")
    integrand = integrand.replace("e^", "2.718**")
    integrand = integrand.replace("^", "**")
    integrand = integrand.replace("\\ln", "log")
    integrand = re.sub(r"e\*\*([^*]+)", r"exp(\1)", integrand)
    integrand = re.sub(r"\+?\s*C\b", "", integrand)

    return integrand.strip()


def parse_variant(problem: str, level: int = -1) -> Variant | None:
    parts, kind = classify_integral_call(problem)

    match kind:
        case IntegralKind.INDEFINITE:
            integrand = parts[0]
            integrand = clean_expr(integrand)
            var = parts[1]
            lb = -math.inf
            ub = math.inf
            if len(var) == 1 and var.isalpha():
                return Variant(integrand, var, lb, ub, False, level)
            else:
                return None

        case IntegralKind.DEFINITE_SEPARATE | IntegralKind.DEFINITE_TUPLE:
            integrand = parts[0]
            integrand = clean_expr(integrand)
            var = parts[1]
            lb = parts[2]
            ub = parts[3]
            if var.isalpha() and (len(var) == 1) and lb.isnumeric() and ub.isnumeric():
                return Variant(integrand, var, float(lb), float(ub), True, level)
            else:
                return None

        case IntegralKind.ERROR:
            return None


def variant_to_sympy(variant: Variant):
    expr_dict = SYMBOLS_DICT | {variant.var: sp.symbols(variant.var)}
    return parse_expr(variant.integrand, local_dict=expr_dict)


def is_valid_sympy(variant: Variant):
    # We make sure a variant is parsable by sympy and that
    with warnings.catch_warnings(record=True) as warnings_list:
        try:
            expr = variant_to_sympy(variant)
        except Exception:
            return False, None

        if len(warnings_list) > 1:
            return False, None

    return True, expr


def is_indefinite_integral_valid(
    integrand: sp.Expr, variable: str, max_timeout: float = 0.1
):
    """
    In the verifier we use numerical approximation, so as long as
    that works then its a valid integral.
    """
    var = sp.symbols(variable)

    if integrand.has(sp.log) or integrand.has(sp.sqrt):
        lb, ub = 0.1, 10
    else:
        lb, ub = -10, 10

    try:
        with timeout(max_timeout):
            sol = mp.quad(sp.lambdify(var, integrand, "mpmath"), [lb, ub])
            complex(sp.N(sol))
    except TimeoutError:
        return False
    except Exception:
        return False

    return True


def is_definite_integral_valid(
    integrand: sp.Expr,
    variable: str,
    lower_bound: float,
    upper_bound: float,
    max_timeout: float = 0.1,
):
    """
    Just try to solve it within a time limit.
    """
    var = sp.symbols(variable)
    try:
        with timeout(max_timeout):
            sol = mp.quad(
                sp.lambdify(var, integrand, "mpmath"), [lower_bound, upper_bound]
            )
            complex(sp.N(sol))

    except TimeoutError:
        return False
    except Exception:
        return False

    return True


def dedup_variants(
    variants: List[str],
    level: int,
    seen: set[str],
    seen_sp: set[Any],
    max_time: float = 0.01,
):
    """
    Update the seen variants handling duplicates with sympy or falling back
    to string comparison.
    """
    timeout_errors = 0
    res = []
    for variant in variants:
        variant_parsed = parse_variant(variant, level)

        if variant_parsed is None:
            continue

        valid_syntax, integrand_sp = is_valid_sympy(variant_parsed)

        if not valid_syntax:
            continue

        if variant_parsed.is_definite:
            valid_integral = is_definite_integral_valid(
                integrand_sp, variant_parsed.var, variant_parsed.lb, variant_parsed.ub
            )
        else:
            valid_integral = is_indefinite_integral_valid(
                integrand_sp, variant_parsed.var
            )

        if not valid_integral:
            continue

        skip_variant = False
        try:
            with timeout(max_time):
                integrand_sp_simpl = simplify(integrand_sp)

                if any(
                    simplify(expr_sp - integrand_sp_simpl) == 0 for expr_sp in seen_sp
                ):
                    skip_variant = True
                else:
                    # No sympy equivalent expr found for variant
                    seen_sp.add(integrand_sp_simpl)

        except KeyboardInterrupt:
            sys.exit(1)
        except TimeoutError:
            timeout_errors += 1
            skip_variant = variant_parsed.integrand in seen
        except Exception:
            skip_variant = variant_parsed.integrand in seen

        if not skip_variant:
            seen.add(variant_parsed.integrand)
            res.append(variant_parsed)

    return res, timeout_errors


def flatten_tree(data: Dict[str, Any], max_time: float = 0.01):
    """
    Applies BFS with a queue.
    """
    base_variant = parse_variant(data["base_question"], 0)
    if base_variant is None:
        raise ValueError(
            f"Could not extract integrand from base question {data['base_question']}"
        )

    try:
        seen_sp = {simplify(variant_to_sympy(base_variant))}
    except Exception:
        seen_sp = set()

    seen = {base_variant.integrand}

    num_timeout = 0
    dup_elements = 0
    elements: List[Variant] = []
    for root_node in data["tree"]:
        variants, timeouts = dedup_variants(
            root_node["variants"], root_node["level"], seen, seen_sp, max_time
        )
        elements.extend(variants)
        num_timeout += timeouts
        dup_elements += len(root_node["variants"])

        queue: Deque[Dict[str, Any]] = deque()
        queue.append(root_node)

        while len(queue) > 0:
            node = queue.popleft()
            for child in node["children"]:
                queue.append(child)

                variants, timeouts = dedup_variants(
                    child["variants"], child["level"], seen, seen_sp, max_time
                )
                elements.extend(variants)
                num_timeout += timeouts
                dup_elements += len(child["variants"])

    return elements, num_timeout, dup_elements


def flatten_trees_from_dir(trees_dir: str, max_time: float = 0.01) -> List[FlatTree]:
    res = []

    for file in os.listdir(trees_dir)[:1]:
        print(f"Processing {file}")
        file_path = os.path.join(trees_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        base_question = data["base_question"]
        # Format of files is tree_id.json
        question_id = file.split(".")[0].split("_")[1]
        variants, num_timeout, dup_elements = flatten_tree(data, max_time)

        res.append(FlatTree(base_question, question_id, variants))

        print(f"Variants: {dup_elements}")
        print(f"Sympy Timeout Variants: {num_timeout}")
        print(f"Deduplicated and valid variants: {len(variants)}")

    return res
