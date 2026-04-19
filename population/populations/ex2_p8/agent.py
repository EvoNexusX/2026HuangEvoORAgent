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


def or_llm_agent(
    user_question,
    model_name=DEFAULT_MODEL_NAME,
    max_attempts=3,
    enable_validation=True,
    temperature=0.1,
    verbose=False
):
    """运筹优化问题求解代理，通过LLM生成数学模型并求解。
    
    Args:
        user_question: 用户提供的优化问题描述
        model_name: 使用的LLM模型名称
        max_attempts: 最大尝试次数
        enable_validation: 是否启用模型验证
        temperature: LLM采样温度
        verbose: 是否输出详细过程
    
    Returns:
        (is_solve_success, result): 求解状态和结果
    """
    
    def validate_model_structure(math_model):
        """验证数学模型基本结构"""
        required_keywords = ['max', 'min', 'subject to', 's.t.', '约束', '目标']
        return any(keyword in math_model.lower() for keyword in required_keywords)
    
    def extract_variables(math_model):
        """从数学模型中提取决策变量"""
        import re
        patterns = [
            r'x_\{[^}]+\}',
            r'y_\{[^}]+\}',
            r'z_\{[^}]+\}',
            r'x\[[^\]]+\]',
            r'\b[a-z]_\{?[a-z0-9]+\}?\b'
        ]
        variables = set()
        for pattern in patterns:
            variables.update(re.findall(pattern, math_model))
        return list(variables)
    
    # 第一阶段：生成并验证数学模型
    messages = [
        {
            "role": "system",
            "content": (
                "你是运筹优化领域的专家。请根据用户提供的优化问题，构建准确的数学模型表达式。\n"
                "要求：\n"
                "1. 使用标准数学符号和线性/整数规划表达式\n"
                "2. 明确定义决策变量、目标函数和约束条件\n"
                "3. 保持表达式简洁，用于后续代码生成\n"
                "4. 无需解释，直接输出数学模型\n\n"
                "关键要求：\n"
                "1. 使用标准LP/MIP表示法\n"
                "2. 用清晰索引定义所有决策变量\n"
                "3. 声明目标函数（最大化/最小化）\n"
                "4. 列出所有约束和变量类型定义"
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name, temperature=temperature)
        if verbose:
            print("【数学模型生成】:")
            print(math_model)
            print("-" * 50)
        
        # 验证模型结构
        if enable_validation and not validate_model_structure(math_model):
            if verbose:
                print("⚠️ 模型结构验证失败，请求修订")
            
            messages.append({"role": "assistant", "content": math_model})
            messages.append({
                "role": "user",
                "content": (
                    "模型缺少必要组件。请修订以包含：\n"
                    "1. 清晰的决策变量定义\n"
                    "2. 明确的目标函数（最大化/最小化）\n"
                    "3. 完整的约束集\n"
                    "4. 变量类型声明（连续、整数、二进制）"
                )
            })
            math_model = query_llm(messages, model_name, temperature=temperature)
            
            if verbose:
                print("【修订后的数学模型】:")
                print(math_model)
                print("-" * 50)
        
        # 提取变量用于验证
        variables = extract_variables(math_model)
        if verbose and variables:
            print(f"📊 检测到变量: {variables}")
        
        if not math_model or len(math_model.strip()) < 50:
            print("警告：数学模型可能不完整")
            
    except Exception as e:
        print(f"数学模型生成失败: {e}")
        return False, "数学模型生成失败"
    
    # 添加到消息历史
    messages.append({"role": "assistant", "content": math_model})
    
    # 第二阶段：生成Gurobi代码并求解
    code_instruction = {
        "role": "user",
        "content": (
            "根据上述数学模型，编写完整的Gurobi Python求解代码。\n"
            "要求：\n"
            "1. 完整实现模型构建、变量定义、约束添加、目标函数设置\n"
            "2. 包含求解、结果输出和错误处理\n"
            "3. 确保代码可直接运行，必要时添加注释\n"
            "4. 输出格式：```python\n{code}\n```\n\n"
            "具体要求：\n"
            "1. 导入 gurobipy 为 gp\n"
            "2. 创建有意义的模型名称\n"
            "3. 正确定义所有变量\n"
            "4. 设置正确的目标方向\n"
            "5. 添加所有约束\n"
            "6. 调用 model.optimize()\n"
            "7. 检查状态并处理不可行情况\n"
            "8. 打印解决方案细节\n"
            "9. 包含错误处理"
        )
    }
    messages.append(code_instruction)
    
    # 初始代码生成和求解
    is_solve_success, result, messages = generate_or_code_solver(
        messages, model_name, max_attempts, temperature=temperature
    )
    
    if verbose:
        print(f"【初始求解结果】: 成功={is_solve_success}, 结果={result}")
    
    # 第三阶段：处理特殊情况和迭代优化
    attempt_count = 1
    
    while attempt_count <= max_attempts and not is_solve_success:
        if verbose:
            print(f"🔄 尝试 {attempt_count}/{max_attempts}")
        
        # 检查是否为无可行解
        result_str = str(result)
        if "infeasible" in result_str.lower() or (enable_validation and "不可行" in result_str):
            if verbose:
                print("【检测到无可行解】启动调试...")
            
            debug_message = {
                "role": "user",
                "content": (
                    "当前模型得到无可行解。请分析并修复：\n"
                    "1. 检查约束逻辑\n"
                    "2. 验证变量边界\n"
                    "3. 确保参数值正确\n"
                    "4. 如有需要，放松过于严格的约束\n\n"
                    "提供修订代码：```python\n{code}\n```"
                )
            }
            messages.append(debug_message)
            is_solve_success, result, messages = generate_or_code_solver(
                messages, model_name, 1, temperature=temperature
            )
        else:
            # 代码调试失败
            if verbose:
                print("【代码调试失败】启动错误检查...")
            
            error_message = {
                "role": "user",
                "content": (
                    "代码调试失败。请：\n"
                    "1. 检查数学模型是否正确\n"
                    "2. 修复语法和逻辑错误\n"
                    "3. 重新生成完整Gurobi代码\n"
                    "输出格式：```python\n{code}\n```"
                )
            }
            messages.append(error_message)
            is_solve_success, result, messages = generate_or_code_solver(
                messages, model_name, 1, temperature=temperature
            )
        
        attempt_count += 1
    
    # 最终结果处理
    if is_solve_success:
        if verbose:
            print("【求解成功】最终结果:", result)
    else:
        if verbose:
            print("【所有尝试失败】尝试应急恢复...")
        
        # 尝试简化版本
        recovery_message = {
            "role": "user",
            "content": (
                "多次尝试失败。提供最小可行代码：\n"
                "1. 求解简化版本\n"
                "2. 只包含关键约束\n"
                "3. 包含错误处理\n\n"
                "格式：```python\n[code]\n```"
            )
        }
        messages.append(recovery_message)
        is_solve_success, result, _ = generate_or_code_solver(
            messages, model_name, 1, temperature=temperature
        )
        
        if not is_solve_success:
            print("【求解失败】请检查问题描述或调整参数")
            result = "求解失败，请提供更清晰的问题描述或调整参数"
    
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

