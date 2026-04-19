import argparse
import copy

import new_utils
from new_utils import (
    query_llm,
    save_generated_code,
    load_dataset,
    extract_and_execute_python_code,
    eval_model_result,
    is_number_string,
)


DEFAULT_MODEL_NAME = ""


def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)

    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    save_generated_code(gurobi_code, prefix="agent")

    text = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        success, error_msg = extract_and_execute_python_code(text)
        if success:
            messages_bak.append({"role": "assistant", "content": gurobi_code})
            return True, error_msg, messages_bak

        print(f"\nAttempt {attempt + 1} failed, requesting LLM to fix code...\n")
        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({
            "role": "user",
            "content": f"代码执行时发生错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整可执行代码。",
        })

        gurobi_code = query_llm(messages, model_name)
        save_generated_code(gurobi_code, prefix="agent_fix")
        text = f"{gurobi_code}"

        print("\nReceived fixed code, preparing to execute again...\n")
        attempt += 1

    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum number of attempts ({max_attempts}), could not execute code successfully.")
    return False, None, messages_bak


def or_llm_agent(user_question, model_name=DEFAULT_MODEL_NAME, max_attempts=3):
    """
    LLM agent for Operations Research optimization problems.
    
    This function:
    1. Generates a mathematical model from the user's problem description
    2. Generates Gurobi Python code to solve the model
    3. Handles infeasible cases and code errors with retry mechanisms
    
    Args:
        user_question: Problem description in natural language
        model_name: LLM model to use (default: DEFAULT_MODEL_NAME)
        max_attempts: Maximum debugging attempts for code generation
    
    Returns:
        Tuple of (success_flag, result)
    """
    
    # Step 1: Generate mathematical model
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in Operations Research and optimization. "
                "Based on the user's optimization problem, construct a precise mathematical "
                "model (linear programming formulation) that accurately describes the original problem. "
                "Focus on providing correct mathematical expressions without excessive explanation. "
                "This model will be used to generate Gurobi code."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    math_model = query_llm(messages, model_name)
    print("【Mathematical Model】:\n", math_model)
    
    # Step 2: Generate initial Gurobi code
    messages.extend([
        {"role": "assistant", "content": math_model},
        {
            "role": "user",
            "content": (
                "Based on the mathematical model above, write complete and reliable "
                "Gurobi Python code to solve this optimization problem. "
                "The code should include: model construction, variable definition, "
                "constraint addition, objective function setup, solving, and result output. "
                "Output only the code in format: ```python\n{code}\n```"
            ),
        }
    ])
    
    is_solve_success, result, messages = generate_or_code_solver(
        messages, model_name, max_attempts
    )
    print(f"Stage result: Success={is_solve_success}, Result={result}")
    
    # Step 3: Handle infeasible solutions
    if is_solve_success:
        if not is_number_string(str(result)):
            print("!![No available solution warning]!!")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The current model yields *no feasible solution*. "
                        "Please carefully check the mathematical model and Gurobi code "
                        "for potential errors causing infeasibility. "
                        "After review, output revised Gurobi Python code. "
                        "Format: ```python\n{code}\n```"
                    ),
                }
            )
            is_solve_success, result, messages = generate_or_code_solver(
                messages, model_name, max_attempts=1
            )
    
    # Step 4: Handle code generation errors
    else:
        print("!![Max attempt debug error warning]!!")
        messages.append(
            {
                "role": "user",
                "content": (
                    "Multiple debugging attempts failed. Please carefully examine "
                    "the mathematical model for errors, then reconstruct the "
                    "Gurobi Python code. Format: ```python\n{code}\n```"
                ),
            }
        )
        is_solve_success, result, messages = generate_or_code_solver(
            messages, model_name, max_attempts=2
        )
    
    return is_solve_success, result



def gpt_code_agent_simple(user_question, model_name=DEFAULT_MODEL_NAME):
    messages = [
        {
            "role": "system",
            "content": (
                "你是运筹优化领域的专家。请根据用户提供的优化问题构建数学模型，并编写完整可靠的Python Gurobi代码求解该运筹优化问题。"
                "代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。"
                "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。注意{code}部分必须是完整可执行的Python代码，能够直接运行并得到结果。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    save_generated_code(gurobi_code, prefix="simple")

    is_solve_success, result = extract_and_execute_python_code(gurobi_code)
    print(f"Stage result: {is_solve_success}, {result}")
    return is_solve_success, result


def run_eval(use_agent=False, model_name=DEFAULT_MODEL_NAME, data_path="data/datasets/IndustryOR.json", start_i=0):
    dataset = load_dataset(data_path)

    pass_count = 0
    correct_count = 0
    error_datas = []

    for i, d in dataset.items():
        try:
            if int(i) < start_i:
                continue
        except (TypeError, ValueError):
            # If the key is non-numeric, keep existing behavior and run it.
            pass

        new_utils.CURRENT_QUESTION_ID = i
        print(f"=============== num {i} ==================")
        user_question, answer = d["question"], d["answer"]
        print(user_question)
        print("-------------")

        #自我调整
        use_agent = False

        if use_agent:
            print("Using Agent mode (two-stage: modeling + code generation)")
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        else:
            print("Using Simple mode (direct code generation)")
            is_solve_success, llm_result = gpt_code_agent_simple(user_question, model_name)

        if is_solve_success:
            print(f"Successfully executed code, optimal solution value: {llm_result}")
        else:
            print("Failed to execute code.")
        print("------------------")

        pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
        pass_count += 1 if pass_flag else 0
        correct_count += 1 if correct_flag else 0

        if not pass_flag or not correct_flag:
            error_datas.append(i)

        print(f"solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}")
        print(f"[Final] run pass: {pass_flag}, solve correct: {correct_flag}")
        print("\n")

    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run OR LLM evaluation with simplified agent script")
    parser.add_argument("--agent", default="True", help="Use agent mode (two-stage: modeling + code generation)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name for LLM")
    parser.add_argument("--data_path", type=str, default="data/datasets/IndustryOR.json", help="Dataset path")
    parser.add_argument("--start_i", type=int, default=0, help="Start evaluating from this question id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(use_agent=args.agent, model_name=args.model, data_path=args.data_path, start_i=args.start_i)

