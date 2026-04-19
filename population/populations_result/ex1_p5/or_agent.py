import copy
import re
import os
from datetime import datetime

# 导入工具函数
try:
    from new_utils import (
        query_llm,
        save_generated_code,
        load_dataset,
        extract_and_execute_python_code,
        eval_model_result,
        is_number_string,
        CURRENT_QUESTION_ID
    )
except ImportError:
    # 如果在独立环境中，定义这些函数的最小版本
    import subprocess
    import sys
    import tempfile
    import json
    from itertools import zip_longest
    
    CURRENT_QUESTION_ID = None
    
    def is_number_string(s):
        pattern = r"^[-+]?\d+(\.\d+)?$"
        return re.match(pattern, s) is not None
    
    def extract_and_execute_python_code(text_content):
        python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)
        if not python_code_blocks:
            return False, "No Python code blocks found"
        
        for code_block in python_code_blocks:
            code_block = code_block.strip()
            if not code_block:
                continue
            
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                    tmp_file.write(code_block)
                    temp_file_path = tmp_file.name
                
                result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    match = re.search(r'Best objective\s+([\d.e+-]+)', result.stdout)
                    if not match:
                        match = re.search(r'Optimal objective\s+([\d.e+-]+)', result.stdout)
                    if not match:
                        match = re.search(r'Optimal cost\s+([\d.e+-]+)', result.stdout)
                    
                    if match:
                        return True, str(float(match.group(1)))
                    return True, "None"
                else:
                    return False, result.stderr
            except Exception as e:
                return False, str(e)
            finally:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        return False, "No valid code blocks executed"
    
    def query_llm(messages, model_name="", temperature=0.2):
        import openai
        import os
        
        client = openai.OpenAI(
            api_key=os.getenv("", ""),
            base_url=os.getenv("", "")
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def save_generated_code(text_content, prefix="solve"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)
            if code_blocks:
                code = "\n\n".join(block.strip() for block in code_blocks if block.strip())
                if not code:
                    code = text_content
                extension = "py"
            else:
                code = text_content
                extension = "txt"
            
            qid_part = str(CURRENT_QUESTION_ID) if CURRENT_QUESTION_ID is not None else "unknown"
            filename = f"{prefix}_q{qid_part}_{timestamp}.{extension}"
            os.makedirs("result", exist_ok=True)
            file_path = os.path.join("result", filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception:
            pass
    
    def load_dataset():
        import json
        dataset = {}
        try:
            with open("dataset/IndustryOR_test.json", 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)
                
                if first_line.startswith('{"en_question"') or first_line.startswith('{"cn_question"'):
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                dataset_item = {
                                    'question': item.get('en_question', item.get('cn_question', '')),
                                    'answer': item.get('en_answer', item.get('cn_answer', '')),
                                    'difficulty': item.get('difficulty', 'Unknown'),
                                    'id': item.get('id', line_num - 1)
                                }
                                dataset[str(dataset_item['id'])] = dataset_item
                            except json.JSONDecodeError:
                                continue
                else:
                    dataset = json.load(f)
        except:
            dataset = {"0": {"question": "Minimize x + y subject to x >= 0, y >= 0, x + y >= 5", "answer": "5"}}
        return dataset
    
    def eval_model_result(success, result, ground_truth, err_rate=0.005):
        pass_flag = False
        correct_flag = False
        if success:
            pass_flag = True
            if is_number_string(str(result)) and ground_truth is not None:
                try:
                    result_num = float(result)
                    ground_truth_num = float(ground_truth)
                    if ground_truth_num == 0:
                        if result_num == 0:
                            correct_flag = True
                    else:
                        deviation = abs(result_num - ground_truth_num) / abs(ground_truth_num)
                        if deviation <= err_rate:
                            correct_flag = True
                except:
                    pass
            elif result == 'None':
                if ground_truth is None or ground_truth == 'None':
                    correct_flag = True
        return pass_flag, correct_flag

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

def or_llm_agent(user_question, model_name="", max_attempts=3):
    stage1_messages = [
        {
            "role": "system",
            "content": (
                "阶段1（start_state: 优化智能体开始搭建, end_state: 问题分析与建模完成）。\n"
                "你是运筹优化领域的专家。请根据用户提供的优化问题，分析问题的优化目标、决策变量、约束条件和目标函数等，用数学（线性规划）表达式构建能够准确描述原问题的数学模型。\n"
                "重点给出正确的数学模型表达式，不必过多解释。"
                "该模型将用于后续生成Gurobi算法设计方案，本步骤主要用于生成有效的线性规模表达式。"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题如下：\n{user_question}"
        },
    ]
    stage1_modeling = query_llm(stage1_messages, model_name)
    print("【Stage 1 - Problem Analysis & Modeling】:\n", stage1_modeling)
    
    stage2_messages = [
        {
            "role": "system",
            "content": (
                "阶段2（start_state: 问题分析与建模完成, end_state: Gurobi算法设计完成）。\n"
                "请基于以下建模方案设计Gurobi求解算法。\n"
                "重点给出正确的Gurobi算法设计方案，不必过多解释。"
                "该模型将用于后续生成Gurobi代码，本步骤主要用于生成有效的设计方案。"
            ),
        },
        {
            "role": "user",
            "content": f"问题分析与建模\n{stage1_modeling}"
        },
    ]
    stage2_design = query_llm(stage2_messages, model_name)
    print("【Stage 2 - Gurobi Algorithm Design】:\n", stage2_design)
    
    messages = [
        {
            "role": "system",
            "content": (
                "根据以上数学模型，使用Gurobi编写完整可靠的Python代码来求解该运筹优化问题。"
                "代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。"
                "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：\n{user_question}\n\n"
                f"问题分析与建模：\n{stage1_modeling}\n\n"
                f"Gurobi算法设计：\n{stage2_design}\n\n"
            ),
        },
    ]
    
    is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts)
    print(f"Stage result: {is_solve_success}, {result}")
    
    if is_solve_success:
        if not is_number_string(str(result)):
            print("!![No available solution warning]!!")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "当前模型得到*无可行解*。请回溯并检查阶段1建模和阶段2算法设计中可能导致不可行的点，"
                        "然后重新输出修正后的Gurobi Python代码。"
                        "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
                    ),
                }
            )
            is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=1)
    else:
        print("!![Max attempt debug error warning]!!")
        messages.append(
            {
                "role": "user",
                "content": (
                    "多次调试后代码仍报错。请仔细检查阶段1建模和阶段2算法设计是否有错误。"
                    "检查后请重新构建Gurobi Python代码。"
                    "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。"
                ),
            }
        )
        is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=2)
    
    return is_solve_success, result

def run_eval(use_agent=True, model_name=""):
    dataset = load_dataset()
    
    pass_count = 0
    correct_count = 0
    error_datas = []
    
    for i, d in dataset.items():
        global CURRENT_QUESTION_ID
        CURRENT_QUESTION_ID = i
        print(f"=============== num {i} ==================")
        user_question, answer = d["question"], d["answer"]
        print(user_question)
        print("-------------")
        
        if use_agent:
            print("Using Agent mode (three-stage: modeling + design + code generation)")
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        else:
            is_solve_success, llm_result = False, None
        
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

