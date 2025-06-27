import json
import os
import signal
import sys
from collections import deque
from contextlib import contextmanager
from typing import Any
from typing import Deque
from typing import Dict
from typing import List
from typing import NamedTuple

import sympy as sp
from sympy import parse_expr
from sympy import simplify

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the "
    "reasoning process in the mind and then provides the user with the answer"
)

TASK_PROMPT = (
    "{problem}\nSolve the following integral. Provide ONLY your antiderivative as a"
    " valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+(1/3)*x**3</answer>"
    " wrapped in a <answer> tags. Importantly, put * between terms you want to"
    " multiply! Show your full working out before solving, don't include any constants"
    " of integration. DO NOT OUTPUT IN LATEX FORMAT. OUTPUT IN SYMPY in <answer> tags."
)


SYMBOLS_DICT = {
    "C": 0,
    "integrate": sp.integrate,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
}


class IntegrandExpr(NamedTuple):
    integrand: str
    var: str


class Variant(NamedTuple):
    expr: str
    level: int


class FlatTree(NamedTuple):
    base_question: str
    tree_id: str
    variants: list[Variant]


def to_verl_format(variant: Variant, index: int, tree_id: str):
    query = TASK_PROMPT.format(problem=variant.expr)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    data = {
        "data_source": "integration_numeric",
        "prompt": prompt,
        "ability": "integration",
        "reward_model": {"style": "rule", "ground_truth": variant.expr},
        "extra_info": {
            "question_index": index,
            "tree_id": tree_id,
            "level": variant.level,
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


def extract_integrand(problem: str):
    if len(problem) == 0:
        return None

    start_index = problem.find("(")
    end_index = problem.rfind(")")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        integrand_expr = problem[start_index + 1 : end_index]
        parts = integrand_expr.split(",")

        if len(parts) <= 1:
            return None

        return IntegrandExpr(integrand=parts[0], var=parts[-1])

    else:
        return None


def integrand_expr_to_sympy(integrand_expr: IntegrandExpr):
    expr_dict = SYMBOLS_DICT | {integrand_expr.var: sp.symbols(integrand_expr.var)}
    return parse_expr(integrand_expr.integrand, local_dict=expr_dict)


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
        skip_variant = False
        try:
            with timeout(max_time):
                variant_expr = extract_integrand(variant)

                if variant_expr is None:
                    raise ValueError("Could not extract integrand")

                variant_sp = simplify(integrand_expr_to_sympy(variant_expr))

                if any(simplify(expr_sp - variant_sp) == 0 for expr_sp in seen_sp):
                    skip_variant = True
                else:
                    # No sympy equivalent expr found for variant
                    seen_sp.add(variant_sp)

        except KeyboardInterrupt:
            sys.exit(1)
        except TimeoutError:
            timeout_errors += 1
            skip_variant = variant in seen
        except Exception:
            skip_variant = variant in seen

        if not skip_variant:
            seen.add(variant)
            res.append(Variant(variant, level))

    return res, timeout_errors


def flatten_tree(data: Dict[str, Any], max_time: float = 0.01):
    """
    Applies BFS with a queue.
    """
    try:
        base_expr = extract_integrand(data["base_question"])
        if base_expr is None:
            raise ValueError("Could not extract integrand from base question")

        seen_sp = {simplify(integrand_expr_to_sympy(base_expr))}
    except Exception:
        seen_sp = set()
    finally:
        seen = {data["base_question"]}

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

    for file in os.listdir(trees_dir):
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
        print(f"Deduplicated Variants: {len(variants)}")
        print(f"Sympy Timeout Variants: {num_timeout}")

    return res
