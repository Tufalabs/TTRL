import json
import sys
import os
import signal
import time
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

import sympy as sp
from sympy import parse_expr
from sympy import simplify
from sympy import sympify


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


def extract_integrand(problem: str):
    if len(problem) == 0:
        return None

    start_index = problem.find("(")
    end_index = problem.rfind(")")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        integrand_expr = problem[start_index + 1 : end_index]
        parts = integrand_expr.split(",")

        if len(parts) != 2:
            return None

        return IntegrandExpr(integrand=parts[0], var=parts[1])

    else:
        return None


def integrand_expr_to_sympy(integrand_expr: IntegrandExpr):
    expr_dict = SYMBOLS_DICT | {integrand_expr.var: sp.symbols(integrand_expr.var)}
    return parse_expr(integrand_expr.integrand, local_dict=expr_dict)


def process_train_files(trees_dir: str) -> List[Dict[str, Any]]:
    """
    Process all tree.json files in the current directory and format them into the required structure.
    """
    formatted_data = []

    # Walk through all directories
    for root, dirs, files in os.walk(trees_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            if file_path.endswith(".json"):
                try:
                    # Check if file is empty
                    if os.path.getsize(file_path) == 0:
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Process each tree structure
                    if (
                        isinstance(data, dict)
                        and "base_question" in data
                        and "tree" in data
                    ):
                        question_data = {
                            "question": data["base_question"],
                            "variants": [],
                        }

                        try:
                            base_expr = extract_integrand(data["base_question"])
                            seen_variants_sp = {
                                simplify(integrand_expr_to_sympy(base_expr))
                            }
                        except:
                            seen_variants_sp = set()

                        seen_variants = {data["base_question"]}
                        num_timeouts = [0]
                        total_variants = [0]

                        # Process all variants from the tree structure
                        def process_node(node):
                            # Add variants from current level
                            total_variants[0] = total_variants[0] + len(node.get("variants", []))
                            for variant in node.get("variants", []):
                                try:
                                    with timeout(0.01):
                                        variant_expr = extract_integrand(variant)
                                        variant_sp = simplify(
                                            integrand_expr_to_sympy(variant_expr)
                                        )

                                        for seen_sp in seen_variants_sp:
                                            if simplify(seen_sp - variant_sp) == 0:
                                                continue

                                        # No sympy equivalent expr found for variant
                                        seen_variants_sp.add(variant_sp)
                                except KeyboardInterrupt:
                                    sys.exit(1)
                                except TimeoutError:
                                    num_timeouts[0] = num_timeouts[0] + 1
                                    if variant in seen_variants:
                                        continue
                                except:
                                    if variant in seen_variants:
                                        continue

                                seen_variants.add(variant)
                                variant_data = {
                                    "variant": variant,
                                    "difficulty": f"level_{node['level']}",
                                }
                                question_data["variants"].append(variant_data)

                            # Process children recursively
                            for child in node.get("children", []):
                                process_node(child)

                        # Process the tree starting from root
                        for root_node in data["tree"]:
                            process_node(root_node)

                        if question_data[
                            "variants"
                        ]:  # Only add if there are valid variants
                            dedup_variants = len(question_data["variants"])
                            print(f"Total variants: {total_variants}")
                            print(f"Dedup variants: {dedup_variants}")
                            print(f"Num timeout: {num_timeouts[0]}/{dedup_variants}")
                            formatted_data.append(question_data)

                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            print(f"Processed {file_path}")

    return formatted_data


def main():
    # Process all train files
    formatted_data = process_train_files()

    # Create output directory if it doesn't exist
    os.makedirs("/home/ubuntu/test/TTRL/test_trees", exist_ok=True)

    # Write formatted data to output file
    output_path = os.path.join("/home/ubuntu/test/TTRL/test_trees", "ttrl.json")
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Processed data written to {output_path}")

    # Print number of questions processed
    print(f"Total number of questions processed: {len(formatted_data)}")


if __name__ == "__main__":
    main()
