import copy
import re

import new_utils
from new_utils import (
    query_llm,
    save_generated_code,
    load_dataset,
    extract_and_execute_python_code,
    eval_model_result,
    is_number_string,
)

def generate_or_code_solver(messages_bak, model_name, max_attempts, stage1_output, stage2_output, question):
    messages = copy.deepcopy(messages_bak)
    
    gurobi_code = query_llm(messages, model_name)
    try:
        print("【Python Gurobi Code】:\n", gurobi_code)
    except UnicodeEncodeError:
        print("【Python Gurobi Code】:\n", gurobi_code.encode('gbk', errors='ignore').decode('gbk'))
    
    save_generated_code(gurobi_code, prefix="agent")
    
    code_match = re.search(r'```python\n(.*?)```', gurobi_code, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()
    else:
        text = gurobi_code.strip()
    
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
        code_match = re.search(r'```python\n(.*?)```', gurobi_code, re.DOTALL)
        if code_match:
            text = code_match.group(1).strip()
        else:
            text = gurobi_code.strip()
        save_generated_code(gurobi_code, prefix=f"agent_fix_{attempt+1}")
        
        print("\nReceived fixed code, preparing to execute again...\n")
        attempt += 1
    
    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum number of attempts ({max_attempts}), could not execute code successfully.")
    return False, None, messages_bak

def or_llm_agent(user_question, model_name="", max_attempts=3):
    # Stage 1: 问题分析与建模
    stage1_prompt = f"Analyze this optimization problem:\n\n{user_question}\n\nProvide a complete mathematical model with:\n1. Decision variables (clearly define each variable and its type)\n2. Objective function (maximize/minimize)\n3. All constraints (linear, integer, binary, etc.)\n4. Variable domains and any special conditions\n\nOutput format:\nDecision variables:\nObjective:\nConstraints:\nModel type (LP, MIP, etc.):"
    
    stage1_messages = [
        {
            "role": "system",
            "content": "阶段1（start_state: 优化智能体开始搭建, end_state: 问题分析与建模完成）。\n你是运筹优化领域的专家。请根据用户提供的优化问题，分析问题的优化目标、决策变量、约束条件和目标函数等，用数学（线性规划）表达式构建能够准确描述原问题的数学模型。\n重点给出正确的数学模型表达式，不必过多解释。\n该模型将用于后续生成Gurobi算法设计方案，本步骤主要用于生成有效的线性规模表达式。",
        },
        {
            "role": "user",
            "content": stage1_prompt,
        },
    ]
    
    stage1_modeling = query_llm(stage1_messages, model_name)
    try:
        print("【Stage 1 - Problem Analysis & Modeling】:\n", stage1_modeling)
    except UnicodeEncodeError:
        print("【Stage 1 - Problem Analysis & Modeling】:\n", stage1_modeling.encode('gbk', errors='ignore').decode('gbk'))
    
    # Stage 2: Gurobi算法设计
    stage2_prompt = f"Problem: {user_question}\n\nMathematical model:\n{stage1_modeling}\n\nDesign a Gurobi implementation strategy:\n\n1. Required imports (gurobipy, etc.)\n2. How to create and set up the model\n3. Variable definition strategy (names, types, bounds)\n4. Objective function implementation\n5. Constraint addition method\n6. Model solving and result extraction\n7. Error handling considerations\n\nProvide a clear implementation plan."
    
    stage2_messages = [
        {
            "role": "system",
            "content": "阶段2（start_state: 问题分析与建模完成, end_state: Gurobi算法设计完成）。\n请基于以下建模方案设计Gurobi求解算法。\n重点给出正确的Gurobi算法设计方案，不必过多解释。\n该模型将用于后续生成Gurobi代码，本步骤主要用于生成有效的设计方案。",
        },
        {
            "role": "user",
            "content": stage2_prompt,
        },
    ]
    
    stage2_design = query_llm(stage2_messages, model_name)
    try:
        print("【Stage 2 - Gurobi Algorithm Design】:\n", stage2_design)
    except UnicodeEncodeError:
        print("【Stage 2 - Gurobi Algorithm Design】:\n", stage2_design.encode('gbk', errors='ignore').decode('gbk'))
    
    # Stage 3: 代码生成
    stage3_prompt = f"Generate Python code for this problem:\n\nProblem: {user_question}\n\nMathematical Model:\n{stage1_modeling}\n\nImplementation Plan:\n{stage2_design}\n\nRequirements:\n- Use gurobipy library\n- Include proper exception handling\n- Model must be properly named and configured\n- Print the optimal objective value if solution exists\n- Print appropriate messages for infeasible/unbounded cases\n- Code must be standalone and executable\n- Add comments for key steps\n\nGenerate ONLY the Python code within ```python ``` markers."
    
    stage3_messages = [
        {
            "role": "system",
            "content": "根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。\n代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。\n请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。",
        },
        {
            "role": "user",
            "content": stage3_prompt,
        },
    ]
    
    is_solve_success, result, messages = generate_or_code_solver(
        stage3_messages, model_name, max_attempts, stage1_modeling, stage2_design, user_question
    )
    print(f"Stage result: {is_solve_success}, {result}")
    
    if is_solve_success:
        if not is_number_string(str(result)):
            print("!![No available solution warning]!!")
            error_guidance = "当前模型得到*无可行解*。请回溯并检查阶段1建模和阶段2算法设计中可能导致不可行的点，然后重新输出修正后的Gurobi Python代码。"
            stage3_retry_prompt = f"Generate Python code for this problem:\n\nProblem: {user_question}\n\nMathematical Model:\n{stage1_modeling}\n\nImplementation Plan:\n{stage2_design}\n\n{error_guidance}\nRequirements:\n- Use gurobipy library\n- Include proper exception handling\n- Model must be properly named and configured\n- Print the optimal objective value if solution exists\n- Print appropriate messages for infeasible/unbounded cases\n- Code must be standalone and executable\n- Add comments for key steps\n\nGenerate ONLY the Python code within ```python ``` markers."
            
            stage3_retry_messages = [
                {
                    "role": "system",
                    "content": "根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。\n代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。\n请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。",
                },
                {
                    "role": "user",
                    "content": stage3_retry_prompt,
                },
            ]
            
            is_solve_success, result, messages = generate_or_code_solver(
                stage3_retry_messages, model_name, max_attempts=1, 
                stage1_output=stage1_modeling, stage2_output=stage2_design, question=user_question
            )
    else:
        print("!![Max attempt debug error warning]!!")
        error_guidance = "多次调试后代码仍报错。请仔细检查阶段1建模和阶段2算法设计是否有错误。检查后请重新构建Gurobi Python代码。"
        stage3_retry_prompt = f"Generate Python code for this problem:\n\nProblem: {user_question}\n\nMathematical Model:\n{stage1_modeling}\n\nImplementation Plan:\n{stage2_design}\n\n{error_guidance}\nRequirements:\n- Use gurobipy library\n- Include proper exception handling\n- Model must be properly named and configured\n- Print the optimal objective value if solution exists\n- Print appropriate messages for infeasible/unbounded cases\n- Code must be standalone and executable\n- Add comments for key steps\n\nGenerate ONLY the Python code within ```python ``` markers."
        
        stage3_retry_messages = [
            {
                "role": "system",
                "content": "根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。\n代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。\n请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。",
            },
            {
                "role": "user",
                "content": stage3_retry_prompt,
            },
        ]
        
        is_solve_success, result, messages = generate_or_code_solver(
            stage3_retry_messages, model_name, max_attempts=2, 
            stage1_output=stage1_modeling, stage2_output=stage2_design, question=user_question
        )
    
    return is_solve_success, result

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
        else:
            raise NotImplementedError("Non-agent mode not implemented")
        
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
