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

def stage3_generation(problem_text, stage1_output, stage2_output):
    system_prompt = """根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。
代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。
请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户问题：\n{problem_text}\n\n问题分析与建模：\n{stage1_output}\n\nGurobi算法设计：\n{stage2_output}\n\n"}
    ]
    
    return messages

def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)

    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    save_generated_code(gurobi_code, prefix="or_agent")

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
        save_generated_code(gurobi_code, prefix='or_agent_fix')
        text = f"{gurobi_code}"

        print("\nReceived fixed code, preparing to execute again...\n")
        attempt += 1

    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum number of attempts ({max_attempts}), could not execute code successfully.")
    return False, None, messages_bak

def or_llm_agent(user_question, model_name="", max_attempts=3):
    # Stage 1: 问题分析与建模
    stage1_prompt = "Analyze this optimization problem:\n\n{question}\n\nProvide a complete mathematical model with:\n1. Decision variables (clearly define each variable and its type)\n2. Objective function (maximize/minimize)\n3. All constraints (linear, integer, binary, etc.)\n4. Variable domains and any special conditions\n\nOutput format:\nDecision variables:\nObjective:\nConstraints:\nModel type (LP, MIP, etc.):"
    stage1_messages = [
        {"role": "user", "content": stage1_prompt.format(question=user_question)}
    ]
    stage1_modeling = query_llm(stage1_messages, model_name)
    # 修复编码问题
    if stage1_modeling:
        stage1_modeling = stage1_modeling.encode('utf-8', errors='ignore').decode('utf-8')
    print("【Stage 1 - Problem Analysis & Modeling】:\n", stage1_modeling)

    # Stage 2: Gurobi算法设计
    stage2_messages = [
        {"role": "system", "content": "阶段2（start_state: 问题分析与建模完成, end_state: Gurobi算法设计完成）。\n请基于以下建模方案设计Gurobi求解算法。\n重点给出正确的Gurobi算法设计方案，不必过多解释。\n该模型将用于后续生成Gurobi代码，本步骤主要用于生成有效的设计方案。"},
        {"role": "user", "content": f"问题分析与建模\n{stage1_modeling}"}
    ]
    stage2_design = query_llm(stage2_messages, model_name)
    # 修复编码问题
    if stage2_design:
        stage2_design = stage2_design.encode('utf-8', errors='ignore').decode('utf-8')
    print("【Stage 2 - Gurobi Algorithm Design】:\n", stage2_design)

    # Stage 3: 代码生成
    messages_bak = stage3_generation(user_question, stage1_modeling, stage2_design)
    messages = copy.deepcopy(messages_bak)
    
    gurobi_code = query_llm(messages, model_name)
    # 修复编码问题
    if gurobi_code:
        gurobi_code = gurobi_code.encode('utf-8', errors='ignore').decode('utf-8')
    print("【Stage 3 - Initial Gurobi Code】:\n", gurobi_code)
    save_generated_code(gurobi_code, prefix='or_agent')
    
    text = f"{gurobi_code}"
    success, error_msg = extract_and_execute_python_code(text)
    
    if not success:
        print("代码执行失败，开始修复...")
        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": f"代码执行时发生错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整可执行代码。"})
        
        gurobi_code = query_llm(messages, model_name)
        # 修复编码问题
        if gurobi_code:
            gurobi_code = gurobi_code.encode('utf-8', errors='ignore').decode('utf-8')
        save_generated_code(gurobi_code, prefix='or_agent_fix')
        
        text = f"{gurobi_code}"
        success, result = extract_and_execute_python_code(text)
        
        if not success:
            print("修复后代码仍执行失败")
            return False, None
        else:
            return True, result
    else:
        return True, error_msg

def run_eval(use_agent=True, model_name=""):
    dataset = load_dataset()

    pass_count = 0
    correct_count = 0
    error_datas = []

    for i, d in dataset.items():
        new_utils.CURRENT_QUESTION_ID = i
        print(f"=============== num {i} ==================")
        user_question, answer = d["question"], d["answer"]
        print(user_question)
        print("-------------")

        if use_agent:
            print("Using Agent mode (three-stage: modeling + design + code generation)")
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)

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

if __name__ == "__main__":
    USE_AGENT = True
    MODEL_NAME = ""
    run_eval(use_agent=USE_AGENT, model_name=MODEL_NAME)
