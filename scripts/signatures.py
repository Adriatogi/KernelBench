import dspy


class CodeGeneratorSig(dspy.Signature):
    """You are a talented programmer. You will be provided with a programming problem and you will have to write a GPU kernel.
    It needs to compile and needs to be correct.
    Please format your code with ```python and end with ```"""

    problem = dspy.InputField(
        description="The programming problem that the generated code should solve."
    )
    code = dspy.OutputField(description="The code that solves the given problem.")


class CodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CodeGeneratorSig)

    def forward(self, llm, problem: str):
        with dspy.context(lm=llm):
            return self.prog(problem=problem)


class CodeGeneratorO1(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(CodeGeneratorSig)

    def forward(self, sample_problem: str, problem: str):
        return self.prog(sample_problem=sample_problem, problem=problem)


class CodeFixerSig(dspy.Signature):
    """The generated code for this programming problem failed. The metadata, which can include the error message, in in 'metadata'.
    Rewrite the code to fix the issue. Feel free to change the code as much as you want, or even completely rewrite it.
    Please format your code with ```python and end with ```"""

    problem = dspy.InputField(
        description="The programming problem that the generated code failed on."
    )
    metadata = dspy.InputField(
        description="The metadata that the generated code failed with. This can include the error message."
    )
    failed_code = dspy.InputField(
        description="The generated code that failed on the given test."
    )
    code = dspy.OutputField(description="The new code that fixes the issue.")


class CodeFixer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CodeFixerSig)

    def forward(
        self,
        llm,
        problem: str,
        failed_code: str,
        metadata: str,
    ):
        with dspy.context(lm=llm):
            return self.prog(
                problem=problem,
                failed_code=failed_code,
                metadata=metadata,
            )


class CodeFixerO1(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(CodeFixerSig)

    def forward(
        self,
        problem: str,
        failed_code: str,
        error_message: str,
    ):
        return self.prog(
            problem=problem,
            failed_code=failed_code,
            error_message=error_message,
        )


class CodeFixerBrokePublicSig(dspy.Signature):
    """While fixing the generated code to pass the extra tests, the code broke and stopped passing the public tests, which are a priority. The generated code for this programming problem failed on `failed_test`, giving `actual_output` instead of the expected `expected_output`. Rewrite the code to fix the issue, while still trying to pass the extra tests."""

    problem = dspy.InputField(
        description="The programming problem that the generated code failed on."
    )
    code_that_passed_public_tests = dspy.InputField(
        description="Older code that passed the public tests but not the extra tests. There is a slow chance that the extra tests are wrong and this code is actually correct."
    )
    extra_tests = dspy.InputField(
        description="The extra tests. The priority is to pass public tests, not extra ones. But it would be good to pass as many extra tests as possible."
    )
    failed_code = dspy.InputField(
        description="The generated code that failed on the public test."
    )
    failed_test = dspy.InputField(
        description="The public test that the generated code failed on."
    )
    expected_output = dspy.InputField(
        description="The expected output for the public test."
    )
    actual_output = dspy.InputField(
        description="The actual output for the public test, which is wrong."
    )
    python_code = dspy.OutputField(
        description="The new Python code that fixes the issue."
    )


class CodeFixerBrokePublic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CodeFixerBrokePublicSig)

    def forward(
        self,
        problem: str,
        code_that_passed_public_tests: str,
        extra_tests: str,
        failed_code: str,
        failed_test: str,
        expected_output: str,
        actual_output: str,
    ):
        return self.prog(
            problem=problem,
            code_that_passed_public_tests=code_that_passed_public_tests,
            extra_tests=extra_tests,
            failed_code=failed_code,
            failed_test=failed_test,
            expected_output=expected_output,
            actual_output=actual_output,
        )


class TestGenerator(dspy.Signature):
    """You are a talented competitive programmer. You will be provided with a programming problem and a few sample inputs and desired outputs. You will also be provided with an example of a public test. To better debug your solution, you write 3 extra tests with inputs and desired outputs. Carefully follow the format of the sample inputs and outputs. For example, make sure to add a newline at the end of each input and output. Your tests need not be particularly hard. Make sure to respect the format specified by the problem statement."""

    problem = dspy.InputField()
    public_test_input = dspy.InputField()
    public_test_output = dspy.InputField()
    input_1 = dspy.OutputField()
    output_1 = dspy.OutputField()
    input_2 = dspy.OutputField()
    output_2 = dspy.OutputField()
    input_3 = dspy.OutputField()
    output_3 = dspy.OutputField()
