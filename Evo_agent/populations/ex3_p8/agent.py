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
    LLM-powered agent for Operations Research optimization problems.
    
    This agent:
    1. Generates a mathematical model from a natural language problem description
    2. Produces Gurobi Python code to solve the model
    3. Iteratively debugs and improves the solution through multiple attempts
    
    Args:
        user_question (str): Natural language description of the optimization problem
        model_name (str): LLM model to use (defaults to DEFAULT_MODEL_NAME)
        max_attempts (int): Maximum number of debugging attempts (default: 3)
    
    Returns:
        tuple: (is_solve_success, result, math_model, messages)
            is_solve_success (bool): Whether solving was successful
            result: Solution output or error information
            math_model (str): Generated mathematical model
            messages (list): Complete conversation history
    """
    # Initialize conversation with system prompt
    messages = [
        {
            "role": "system",
            "content": (
                "你是运筹优化领域的专家。请根据用户提供的优化问题，用数学（线性规划）表达式构建能够准确描述原问题的数学模型。\n"
                "重点给出正确的数学模型表达式，不必过多解释。\n"
                "该模型将用于后续生成Gurobi代码，本步骤主要用于生成有效的线性规模表达式。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    # Stage 1: Generate mathematical model
    math_model = query_llm(messages, model_name)
    messages.append({"role": "assistant", "content": math_model})
    
    # Stage 2: Generate initial Gurobi code
    messages.append(
        {
            "role": "user",
            "content": (
                "根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。\n"
                "代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。\n"
                "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
            ),
        }
    )
    
    # Stage 3: Solve with iterative debugging
    total_attempts = 0
    max_total_attempts = max_attempts + 2  # Original + infeasible retry + error retry
    
    while total_attempts < max_total_attempts:
        current_max_attempts = 1 if total_attempts > 0 else max_attempts
        is_solve_success, result, new_messages = generate_or_code_solver(
            messages, model_name, max_attempts=current_max_attempts
        )
        
        # Update messages with the latest interaction
        if new_messages and len(new_messages) > len(messages):
            messages = new_messages
        
        total_attempts += 1
        
        # Check termination conditions
        if is_solve_success:
            if is_number_string(str(result)):
                # Successful solution found
                return True, result, math_model, messages
            else:
                # Infeasible model - trigger debugging
                print("!![No available solution warning]!!")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "当前模型得到*无可行解*。请仔细检查数学模型和Gurobi代码中可能导致不可行的错误。\n"
                            "检查后请重新输出Gurobi Python代码。\n"
                            "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
                        ),
                    }
                )
        else:
            # Code generation/solving error - trigger debugging
            print("!![Code generation/solving error warning]!!")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "代码生成或求解过程中出现错误。请仔细检查数学模型和Gurobi代码的语法和逻辑错误。\n"
                        "检查后请重新输出Gurobi Python代码。\n"
                        "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
                    ),
                }
            )
    
    # Final attempt with comprehensive debugging prompt
    print("!![Final debugging attempt]!!")
    messages.append(
        {
            "role": "user",
            "content": (
                "经过多次尝试仍未成功。请从以下几个方面全面检查：\n"
                "1. 数学模型是否准确反映原问题\n"
                "2. 变量定义是否正确（类型、边界）\n"
                "3. 约束条件是否完整且正确\n"
                "4. 目标函数是否正确\n"
                "5. Gurobi语法是否正确\n"
                "检查后请重新输出修正后的Gurobi Python代码。\n"
                "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
            ),
        }
    )
    
    final_success, final_result, final_messages = generate_or_code_solver(
        messages, model_name, max_attempts=2
    )
    
    return final_success, final_result, math_model, final_messages if final_messages else messages



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

