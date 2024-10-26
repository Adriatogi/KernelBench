import os
from utils import read_file
'''
Construct Prompts
As basic as we can be, not to steer the LLM too much
'''

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)
    
def get_arch_definition(arch_src):
    '''
    Construct torch definition from original torch nn.Module definition 
    '''
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
def prompt_generate_custom_cuda(arc_src: str,
                                 example_arch_src: str,
                                 example_new_arch_src: str) -> str:
    prompt = f"""
    You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
    """

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ```
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional.\n
    """
    return prompt

def prompt_generate_custom_cuda_from_file(arch_path, example_ind=0):
    arch = get_arch_definition_from_file(arch_path)
    # These are strictly defined for now
    
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_{example_ind}.py")
    example_new_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_new_ex_{example_ind}.py")
    
    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(f"Example new architecture file not found: {example_new_arch_path}")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)
    

def prompt_generate_custom_cuda_from_file_save(arch_path, example_ind=0):
    generated_prompt = prompt_generate_custom_cuda_from_file(arch_path, example_ind)

    os.makedirs(os.path.join(REPO_TOP_PATH, "src/scratch"), exist_ok=True)
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/prompt.txt"), "w") as f:
        f.write(generated_prompt)

    return generated_prompt