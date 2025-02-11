import json, os
import argparse

"""
Analyze the greedy eval results for a run of a particular level
"""
from src.dataset import construct_kernelbench_dataset


def analyze_runs(run_names, level=1):
    dataset = construct_kernelbench_dataset(level)
    total_count = len(dataset)

    for run_name in run_names:
        eval_file_path = f"runs/{run_name}/eval_results.json"

        if not os.path.exists(eval_file_path):
            print(f"Eval file does not exist at {eval_file_path}")
            continue

        with open(eval_file_path, "r") as f:
            eval_results = json.load(f)

        # Initialize counters
        total_eval = len(eval_results)
        compiled_count = 0
        correct_count = 0

        # Count results
        for entry in eval_results.values():
            if entry["compiled"] == True:
                compiled_count += 1
            if entry["correctness"] == True:
                correct_count += 1

        # Print results
        print("-" * 128)
        print(f"Eval Summary for {run_name}")
        print("-" * 128)
        print(f"Total test cases with Eval Results: {total_eval} out of {total_count}")
        print(f"Successfully compiled: {compiled_count}")
        print(f"Functionally correct: {correct_count}")

        print(f"\nSuccess rates:")
        print(f"Compilation rate: {compiled_count/total_count*100:.1f}%")
        print(f"Correctness rate: {correct_count/total_count*100:.1f}%")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze greedy eval results for multiple runs."
    )
    parser.add_argument("run_names", nargs="+", help="List of run names to analyze.")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Level of the dataset to analyze (default: 1).",
    )

    args = parser.parse_args()
    analyze_runs(args.run_names, args.level)
