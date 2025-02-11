# Standard library imports
import os
import sys
import time
import random
import json
import argparse
from dataclasses import dataclass
import multiprocessing as mp


# Third-party imports
import numpy as np
import torch
import dspy
from datasets import load_dataset

# Local imports
from signatures import CodeGenerator, CodeFixer, CodeFixerO1, CodeGeneratorO1
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.dataset import construct_kernelbench_dataset
from src.utils import (
    extract_first_code,
    maybe_multithread,
)
from scripts.eval_from_generations import cuda_single_eval_wrapper, EvalConfig, WorkArgs

MAX_ATTEMPTS = 10

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs_dir = os.path.join(REPO_TOP_DIR, "runs")


# @dataclass
# class WorkArgs:
#     problem_id: int  # logically indexed
#     sample_id: int


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--iterations", "-i", type=int, required=True, help="Number of iterations"
    # )
    parser.add_argument(
        "--level", "-l", type=int, required=True, help="level of the dataset"
    )
    parser.add_argument(
        "--llms",
        type=str,
        nargs="+",
        default=["claude-sonnet", "llama", "claude-sonnet"],
        help="List of LLM names",
    )

    parser.add_argument(
        "--partial_dataset", "-p", action="store_true", help="Use partial dataset"
    )

    parser.add_argument("--test", "-t", action="store_true", help="Use test subset")

    parser.add_argument(
        "--num_workers", "-n", type=int, default=1, help="Number of workers"
    )
    parser.add_argument(
        "--api_query_interval", "-a", type=float, default=0.0, help="API query interval"
    )
    parser.add_argument(
        "--run_name", "-r", type=str, default="run_name", help="Run name"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    parser.add_argument("--start_id", "-s", type=int, default=1, help="Start id")

    parser.add_argument(
        "--num_gpus",
        "-g",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs available",
    )

    return parser.parse_args()


with open("api_keys.json", "r") as f:
    api_keys = json.load(f)

os.environ["ANTHROPIC_API_KEY"] = api_keys["ANTHROPIC_API_KEY"][0]
os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"][1]
os.environ["TOGETHERAI_API_KEY"] = api_keys["TOGETHER_API_KEY"][0]
# os.environ["SAMBANOVA_API_KEY"] = api_keys["SAMBANOVA_API_KEY"][2]
# os.environ["DEEPSEEK_API_KEY"] = api_keys["DEEPSEEK_API_KEY"][0]

# Really really don't like this set up lol

o1_mini = dspy.LM("openai/o1-mini", max_tokens=8192, temperature=1)
o1_preview = dspy.LM("openai/o1-preview", max_tokens=8192, temperature=1)
o1_list = [o1_mini, o1_preview]

gpt4o = dspy.LM("openai/gpt-4o-mini", max_tokens=8192, temperature=0.7, cache=False)
claude_sonnet = dspy.LM(
    "claude-3-5-sonnet-20240620", max_tokens=8192, temperature=0.0, cache=False
)
claude_haiku = dspy.LM(
    "claude-3-haiku-20240307", max_tokens=4096, temperature=0.7, cache=False
)
llama = dspy.LM(
    "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    max_tokens=8192,
    temperature=0.7,
    cache=False,
)
llama_sambanova = dspy.LM(
    "sambanova/Meta-Llama-3.1-405B-Instruct", max_tokens=8192, temperature=0.7
)
qwen = dspy.LM(
    "together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=8192,
    temperature=0.7,
    cache=False,
)
deepseek = dspy.LM(
    "deepseek/deepseek-coder", max_tokens=8192, temperature=0.7, cache=False
)

r1 = dspy.LM("together_ai/deepseek-ai/DeepSeek-R1", temperature=0.7, cache=False)

llms_map = {
    # "o1-mini": o1_mini,
    # "o1-preview": o1_preview,
    "gpt4o": gpt4o,
    "claude-haiku": claude_haiku,
    "claude-sonnet": claude_sonnet,
    "llama": llama,
    # "llama-sambanova": llama_sambanova,
    "qwen": qwen,
    # "deepseek": deepseek,
    "r1": r1,
}

args = parse_args()
llm_names = args.llms
device = torch.device("cuda:0")
llms = [llms_map[llm_name] for llm_name in llm_names]


class CodeGenerationPipeline(dspy.Module):
    def __init__(self, llms):
        super().__init__()
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.llms = llms
        self.code_generator = CodeGenerator()
        self.code_generator_o1 = CodeGeneratorO1()
        self.code_fixer = CodeFixer()
        # self.code_fixer_o1 = CodeFixerO1() # I dont think there is a difference?

        # Add counters for tracking progress
        print("Initializing CodeGenerationPipeline with:")
        print(f"- Using LLMs: {llm_names}\n")
        print(f"-Number of LLMs: {len(llms)}")

    def _test_code(self, work, run_dir, dataset):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print("No GPU available")

        # Use cuda_single_eval_wrapper instead of eval_kernel_against_ref
        work_args = WorkArgs(
            problem_id=work.problem_id,
            sample_id=work.sample_id,
            device=device,
        )
        config = EvalConfig()
        config.run_name = args.run_name
        config.dataset_src = "huggingface"  # or "local", depending on your setup
        config.level = args.level
        config.gpu_arch = ["Ada"]  # or your specific GPU architecture
        config.verbose = args.verbose
        config.measure_performance = False

        eval_result = cuda_single_eval_wrapper(
            curr_work=work_args, configs=config, dataset=dataset, run_dir=run_dir
        )

        return eval_result

    def _fix_code(self, problem_prompt, previous_attempt, metadata, cur_llm):
        """Code fixing logic"""

        # if is_together:
        #     time.sleep(8)

        # if is_sambanova:
        #     time.sleep(10)

        attempts = 0
        exception = None
        while attempts < MAX_ATTEMPTS:
            try:
                # still dont think this is needed?
                if cur_llm in o1_list:
                    return self.code_fixer_o1(
                        llm=cur_llm,
                        problem=problem_prompt,
                        failed_code=previous_attempt,
                        metadata=metadata,
                    )
                else:
                    return self.code_fixer(
                        llm=cur_llm,
                        problem=problem_prompt,
                        failed_code=previous_attempt,
                        metadata=metadata,
                    )
            except Exception as e:
                if "The maximum context length of" in str(e):
                    raise e
                print(f"Attempt {attempts} failed with error: {e}")
                exception = e
                attempts += 1
                if "RateLimitError" in str(e):
                    time.sleep(random.randint(15, 45))
                else:
                    time.sleep(random.randint(1, 5))

        raise exception

    def _check_evaluation(self, work, eval_result):
        result = {"correct": False, "error_message": ""}
        if eval_result and eval_result.compiled and eval_result.correctness:
            if True:
                print(
                    f"Generated and validated sample {work.sample_id} for problem {work.problem_id}"
                )
            result["correct"] = True
            return result

        # If we get here, there was an error
        # if "other_error" in eval_result.metadata:
        #     result["error_message"] = "most likley a timeout or cuda error."
        # elif not eval_result.compiled:
        #     result["error_message"] = (
        #         f"Compilation error: {eval_result.metadata.get('compilation_error', 'Unknown compilation error')}"
        #     )
        # elif not eval_result.correctness:
        #     result["error_message"] = (
        #         f"Correctness error: {eval_result.metadata.get('correctness_issue', 'Unknown correctness error')}"
        #     )

        result["metadata"] = eval_result.metadata

        return result

    def generate_code(self, work, dataset, run_dir):
        print(f"\n{'='*50}")
        print(
            f"Starting code generation for Problem {work.problem_id}, Sample {work.sample_id}"
        )
        print(f"{'='*50}")

        # Fetch problem details
        curr_problem_row = dataset.filter(
            lambda x: x["problem_id"] == work.problem_id, desc=None
        )

        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
        problem_number = int(problem_name.split("_")[0])
        assert (
            problem_number == work.problem_id
        ), f"Problem number in filename ({problem_number}) does not match config problem_id ({work.problem_id})"

        # Construct Prompt
        custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(
            ref_arch_src
        )

        # configure generator here
        cur_llm_index = 0
        cur_llm = self.llms[cur_llm_index % len(self.llms)]
        attempts = 0
        exception = None
        while attempts < MAX_ATTEMPTS:
            try:
                if cur_llm in o1_list:
                    result_generation = self.code_generator_o1(
                        cur_llm, custom_cuda_prompt
                    )
                else:
                    result_generation = self.code_generator(cur_llm, custom_cuda_prompt)

                # returns prediction object with rationale
                # print(f"{result_generation=}")

                # Extract code from the Prediction object before passing to extract_first_code
                custom_cuda = result_generation.code
                # }")
                # print(f"{result_generation.code=}")
                custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
                print(f"{custom_cuda=}")

                if custom_cuda is None:
                    error_message = "Failed to generate valid code block"
                    print(
                        f"Custom CUDA code generation failed on attempt {attempts + 1}"
                    )
                    attempts += 1
                    continue
                break
            except Exception as e:
                print(f"❌ Generation attempt {attempts + 1} failed: {str(e)}")
                exception = e
                attempts += 1
                time.sleep(random.randint(15, 45))

        cur_llm_index += 1

        if attempts == MAX_ATTEMPTS:
            raise exception

        if cur_llm_index == len(self.llms):
            return custom_cuda

        save_kernel(work, run_dir, custom_cuda)

        # evaluate the code.
        print("\nEvaluating generated code...")
        eval_result = self._test_code(work, run_dir, dataset)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = self._check_evaluation(work, eval_result)

        if result["correct"]:
            print(
                f"✅ Code passed all tests on first attempt for problem {work.problem_id} with {cur_llm.model}!"
            )
            return custom_cuda

        # lets start fixing the code
        previous_attempt = custom_cuda
        metadata = result["metadata"]

        # Code fixing loop
        print("\nStarting code fixing iterations...")
        while not result["correct"] and cur_llm_index < len(self.llms):
            print(f"\nIndex {cur_llm_index} < {len(self.llms)}")
            print(f"metadata: {metadata}")

            cur_llm = self.llms[cur_llm_index % len(self.llms)]
            result_generation = self._fix_code(
                custom_cuda_prompt, previous_attempt, metadata, cur_llm
            )
            custom_cuda = result_generation.code
            custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
            # print(f"{custom_cuda=}")

            save_kernel(work, run_dir, custom_cuda)

            eval_result = self._test_code(work, run_dir, dataset)

            result = self._check_evaluation(work, eval_result)
            if result["correct"]:
                print(
                    f"✅ Code fixed successfully on index {cur_llm_index} for problem {work.problem_id} with {cur_llm.model}!"
                )
                return custom_cuda

            print(
                f"❌ Fix attempt for index {cur_llm_index} failed on problem {work.problem_id} with {cur_llm.model}!"
            )
            metadata = result["metadata"]
            previous_attempt = custom_cuda
            cur_llm_index += 1

        print("Finished pipeline  without success")
        return custom_cuda


def generate_sample_single(
    work: WorkArgs,
    args,
    dataset,
    run_dir: str,
) -> bool:
    # Set the device for the current work

    # Query server with constructed prompt
    custom_cuda = code_generation_pipeline.generate_code(work, dataset, run_dir)

    # No need to extract code again since generate_code() now returns the code directly
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    # Store to local file
    save_kernel(work, run_dir, custom_cuda)

    return True


def save_kernel(work, run_dir, custom_cuda):
    kernel_path = os.path.join(
        run_dir,
        f"level_{args.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)


code_generation_pipeline = CodeGenerationPipeline(llms)


def main():
    print(f"\n{'='*50}")
    print(f"Starting KernelBench Code Generation")
    print(f"Level: {args.level}")
    print(f"Dataset: {'Full' if not args.partial_dataset else 'Partial (10 problems)'}")
    print(f"Number of workers per GPU: {args.num_workers}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"{'='*50}\n")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # Load KernelBench dataset
    dataset = load_dataset("ScalingIntelligence/KernelBench")
    curr_level_dataset = dataset[f"level_{args.level}"]

    if args.partial_dataset:
        curr_level_dataset = curr_level_dataset[:10]

    num_problems_in_level = len(curr_level_dataset)
    problem_id_range = range(args.start_id, num_problems_in_level)

    run_dir = os.path.join(runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Distribute problems across GPUs
    problems_per_gpu = len(problem_id_range) // args.num_gpus + (
        1 if len(problem_id_range) % args.num_gpus else 0
    )

    # Create processes directly instead of using a pool
    processes = []
    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * problems_per_gpu + problem_id_range.start
        end_idx = min(start_idx + problems_per_gpu, problem_id_range.stop + 1)

        if start_idx >= end_idx:
            continue

        problems_to_run = []
        for problem_id in range(start_idx, end_idx):
            problems_to_run.append(
                WorkArgs(
                    problem_id=int(problem_id),
                    sample_id=0,
                    device=torch.device(f"cuda:{gpu_id}"),
                )
            )

        # Create and start a process for each GPU
        p = mp.Process(
            target=process_gpu_batch,
            args=(problems_to_run, gpu_id, args, curr_level_dataset, run_dir),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Count results from the run directory
    generated_files = len([f for f in os.listdir(run_dir) if f.endswith("_kernel.py")])
    total_problems = num_problems_in_level
    num_failed_problems = total_problems - generated_files
    print(
        f"Generated {generated_files} samples for total {total_problems} problems. "
        f"Please retry for the {num_failed_problems} failed problems."
    )


def process_gpu_batch(problems_to_run, gpu_id, args, dataset, run_dir):
    print(f"\nProcessing {len(problems_to_run)} problems on GPU {gpu_id}...")
    results = maybe_multithread(
        generate_sample_single,
        problems_to_run,
        args.num_workers,  # workers per GPU
        time_interval=args.api_query_interval,
        # extra args
        args=args,
        dataset=dataset,
        run_dir=run_dir,
    )
    return results


if __name__ == "__main__":
    main()
