import json
import argparse
from typing import Dict, Any
import os


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare kernel execution results against baseline."
    )
    parser.add_argument("-b", "--baseline", help="Path to baseline JSON file")
    parser.add_argument(
        "-r", "--results", nargs="+", help="run names to compare against baseline"
    )
    parser.add_argument(
        "-l",
        "--level",
        choices=["level1", "level2", "level3"],
        default="level1",
        help="Level to compare (default: level1)",
    )
    parser.add_argument("-o", "--output", help="Path to save the comparison results")
    return parser.parse_args()


def process_baseline(baseline_data: Dict[str, Any], level: str) -> Dict[str, float]:
    """Process baseline data to map problem numbers to mean runtimes."""
    result = {}
    level_data = baseline_data.get(level, {})

    for problem_name, stats in level_data.items():
        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
        problem_number = problem_name.split("_")[0]
        result[problem_number] = stats.get("mean", float("inf"))

    return result


def process_result(result_data: Dict[str, Any]) -> Dict[str, float]:
    """Process result data to map problem numbers to mean runtimes."""
    processed = {}

    for problem_number, data in result_data.items():
        if data.get("compiled") and data.get("correctness"):
            runtime_stats = data.get("runtime_stats", {})
            processed[problem_number] = runtime_stats.get("mean", float("inf"))

    return processed


def calculate_speedups(
    baseline_means: Dict[str, float], result_means: Dict[str, float]
) -> Dict[str, float]:
    """Calculate speedups for each problem (baseline_time / result_time)."""
    speedups = {}

    for problem_number in baseline_means:
        if problem_number in result_means:
            baseline_time = baseline_means[problem_number]
            result_time = result_means[problem_number]
            if result_time > 0:  # Avoid division by zero
                speedups[problem_number] = baseline_time / result_time

    return speedups


def main():
    args = parse_args()

    # Load baseline data
    baseline_data = load_json(args.baseline)
    baseline_means = process_baseline(baseline_data, args.level)
    print("len(baseline_means)", len(baseline_means))

    # Process each result file
    comparison_results = {}

    for result_name in args.results:
        result_path = f"runs/{result_name}/eval_results.json"
        result_data = load_json(result_path)
        result_means = process_result(result_data)
        print("len(result_means)", len(result_means))

        # Calculate speedups
        speedups = calculate_speedups(baseline_means, result_means)

        # Calculate average speedup
        if speedups:
            avg_speedup = sum(speedups.values()) / len(speedups)
        else:
            avg_speedup = 0

        comparison_results[result_name] = {
            "average_speedup": avg_speedup,
            "problem_speedups": speedups,
        }

    # Print results
    print("\nComparison Results:")
    print(json.dumps(comparison_results, indent=2))

    # Save results if output path is specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
