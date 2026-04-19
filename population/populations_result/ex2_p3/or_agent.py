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

def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)
    
    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    save_generated_code(grobi_code, prefix="agent")
    
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
        save_generated_code(grobi_code, prefix="agent_fix")
        text = f"{gurobi_code}"
        
        print("\nReceived fixed code, preparing to execute again...\n")
        attempt += 1
    
    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum number of attempts ({max_attempts}), could not execute code successfully.")
    return False, None, messages_bak

def or_llm_agent(user_question, model_name="", max_attempts=3):
    stage1_messages = [
        {
            "role": "system",
            "content": "阶段1（start_state: 优化智能体开始搭建, end_state: 问题分析与建模完成）。\n你是运筹优化领域的专家。请根据用户提供的优化问题，分析问题的优化目标、决策变量、约束条件和目标函数等，用数学（线性规划）表达式构建能够准确描述原问题的数学模型。\n重点给出正确的数学模型表达式，不必过多解释。该模型将用于后续生成Gurobi算法设计方案，本步骤主要用于生成有效的线性规模表达式。"
        },
        {
            "role": "user",
            "content": f"用户问题如下：\n{user_question}"
        }
    ]
    stage1_modeling = query_llm(stage1_messages, model_name)
    print("【Stage 1 - Problem Analysis & Modeling】:\n", stage1_modeling)
    
    stage2_messages = [
        {
            "role": "system",
            "content": "阶段2（start_state: 问题分析与建模完成, end_state: Gurobi算法设计完成）。\n请基于以下建模方案设计Gurobi求解算法。\n重点给出正确的Gurobi算法设计方案，不必过多解释。该模型将用于后续生成Gurobi代码，本步骤主要用于生成有效的设计方案。"
        },
        {
            "role": "user",
            "content": f"问题分析与建模\n{stage1_modeling}"
        }
    ]
    stage2_design = query_llm(stage2_messages, model_name)
    print("【Stage 2 - Gurobi Algorithm Design】:\n", stage2_design)
    
    stage3_messages = [
        {
            "role": "system",
            "content": "You are a Python programming expert specializing in Gurobi optimization. \nYour task is to generate complete, executable Python code for optimization problems."
        },
        {
            "role": "user",
            "content": f"Based on the following problem description, mathematical model, and solution plan, \ngenerate complete Python code using Gurobi.\n\nProblem Description:\n{user_question}\n\nMathematical Model:\n{stage1_modeling}\n\nSolution Plan:\n{stage2_design}\n\nRequirements:\n1. Generate complete Python code including necessary imports (gurobipy, etc.).\n2. The code must be standalone and output the optimal solution if exists, or indicate infeasibility.\n3. Include comments explaining key steps.\n4. Output ONLY the code within ```python ``` code blocks.\n\nPlease output ONLY the code, no other explanations."
        }
    ]
    
    is_solve_success, result, messages = generate_or_code_solver(stage3_messages, model_name, max_attempts)
    print(f"Stage result: {is_solve_success}, {result}")
    
    if is_solve_success:
        if not is_number_string(str(result)):
            print("!![No available solution warning]!!")
            stage3_messages.append(
                {
                    "role": "user",
                    "content": "当前模型得到*无可行解*。请回溯并检查阶段1建模和阶段2算法设计中可能导致不可行的点，然后重新输出修正后的Gurobi Python代码。请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
                }
            )
            is_solve_success, result, messages = generate_or_code_solver(stage3_messages, model_name, max_attempts=1)
    else:
        print("!![Max attempt debug error warning]!!")
        stage3_messages.append(
            {
                "role": "user",
                "content": "多次调试后代码仍报错。请仔细检查阶段1建模和阶段2算法设计是否有错误。检查后请重新构建Gurobi Python代码。请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
            }
        )
        is_solve_success, result, messages = generate_or_code_solver(stage3_messages, model_name, max_attempts=2)
    
    return is_solve_success, result

def run_eval(use_agent=True, model_name=""):
    try:
        dataset = load_dataset()
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("[Total 0] run pass: 0, solve correct: 0")
        print("[Total fails 0] error datas: []")
        return
    
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
            try:
                is_solve_success, llm_result = or_llm_agent(user_question, model_name)
            except Exception as e:
                print(f"Agent执行失败: {e}")
                is_solve_success = False
                llm_result = None
        
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
