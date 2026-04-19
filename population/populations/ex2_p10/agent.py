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
    """运筹优化问题求解代理，通过LLM生成数学模型并迭代求解。
    
    Args:
        user_question: 用户提供的优化问题描述
        model_name: 使用的LLM模型名称
        max_attempts: 最大尝试次数，用于初始代码生成和求解
    
    Returns:
        (is_solve_success, result): 求解状态和结果
    """
    # 辅助函数：检查结果是否有效
    def is_good_result(res):
        """检查结果是否为有效的数值解，排除不可行情况。"""
        res_str = str(res).lower()
        # 检查是否包含不可行标记
        if "infeasible" in res_str:
            return False
        # 尝试使用外部函数检查是否为数值字符串
        try:
            return is_number_string(res_str)
        except NameError:
            # 回退检查：尝试转换为浮点数
            try:
                float(res_str)
                return True
            except ValueError:
                return False

    # 初始化消息历史
    messages = [
        {
            "role": "system",
            "content": (
                "你是运筹优化领域的专家。请根据用户提供的优化问题，构建准确的数学模型表达式。\n"
                "要求：\n"
                "1. 使用标准数学符号和线性/整数规划表达式\n"
                "2. 明确定义决策变量、目标函数和约束条件\n"
                "3. 保持表达式简洁，用于后续代码生成\n"
                "4. 无需解释，直接输出数学模型"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    # 第一阶段：生成数学模型
    try:
        math_model = query_llm(messages, model_name)
        print("【数学模型生成】:\n", math_model)
        if not math_model or len(math_model.strip()) < 50:
            print("警告：数学模型可能不完整，但将继续处理")
    except Exception as e:
        print(f"数学模型生成失败: {e}")
        return False, "数学模型生成失败"

    messages.append({"role": "assistant", "content": math_model})
    
    # 代码生成指令
    code_instruction = {
        "role": "user",
        "content": (
            "根据上述数学模型，编写完整的Gurobi Python求解代码。\n"
            "要求：\n"
            "1. 完整实现模型构建、变量定义、约束添加、目标函数设置\n"
            "2. 包含求解、结果输出和基本错误处理\n"
            "3. 确保代码可直接运行\n"
            "4. 输出格式：```python\n{code}\n```"
        )
    }
    messages.append(code_instruction)

    # 初始求解尝试
    is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts)
    print(f"【初始求解结果】: 成功={is_solve_success}, 结果={result}")

    # 检查初始结果是否有效
    if is_solve_success and is_good_result(result):
        print("【求解成功】初始结果有效")
        return True, result

    # 第二阶段：迭代调试循环
    max_debug_attempts = 2  # 最大调试尝试次数
    debug_attempts = 0
    
    while debug_attempts < max_debug_attempts:
        # 基于当前结果确定调试提示
        if is_solve_success:
            # 求解成功但结果无效（如无可行解）
            print("【检测到无可行解或无效结果，启动调试】")
            debug_message = {
                "role": "user",
                "content": (
                    "当前模型求解成功但结果无效或无可行解。请仔细检查：\n"
                    "1. 约束条件是否相互矛盾或过于严格\n"
                    "2. 变量定义和边界是否正确\n"
                    "3. 目标函数是否与问题描述一致\n"
                    "修正后重新输出完整Gurobi代码。\n"
                    "输出格式：```python\n{code}\n```"
                )
            }
        else:
            # 求解失败（代码错误或生成问题）
            print("【求解失败，启动错误检查】")
            debug_message = {
                "role": "user",
                "content": (
                    "代码生成或求解过程中出现错误。请全面检查：\n"
                    "1. 数学模型是否准确反映原问题\n"
                    "2. Gurobi代码语法和逻辑是否正确\n"
                    "3. 变量和约束定义是否完整\n"
                    "修正后重新输出完整Gurobi代码。\n"
                    "输出格式：```python\n{code}\n```"
                )
            }
        
        # 添加调试消息并再次尝试
        messages.append(debug_message)
        is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=2)
        debug_attempts += 1
        print(f"【调试尝试 {debug_attempts}】: 成功={is_solve_success}, 结果={result}")
        
        # 检查当前结果是否有效
        if is_solve_success and is_good_result(result):
            print("【调试成功】获得有效解")
            return True, result
    
    # 最终处理：多次尝试后仍未成功
    if is_solve_success:
        print("【求解完成但结果无效】最终结果可能不可行")
        return False, "模型求解成功但未获得有效数值解，可能无可行解"
    else:
        print("【求解失败】经过多次尝试未成功")
        return False, "求解失败，请检查问题描述或调整参数后重试"



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

