import re
import subprocess
import sys
import tempfile
import os
import json
from datetime import datetime
from itertools import zip_longest

import openai
import anthropic
from dotenv import load_dotenv

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE_ROOT, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"llm_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

RESULT_DIR = os.path.join(WORKSPACE_ROOT, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
TRACE_FILE = os.path.join(RESULT_DIR, f"thought_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
CODE_SAVE_COUNTER = 0
CURRENT_QUESTION_ID = None

def is_number_string(s):
    pattern = r"^[-+]?\d+(\.\d+)?$"
    return re.match(pattern, s) is not None

def convert_to_number(s):
    try:
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    if "Model is infeasible" in output_text:
        return None

    match = re.search(r'Best objective\s+([\d.e+-]+)', output_text)
    if not match:
        match = re.search(r'Optimal objective\s+([\d.e+-]+)', output_text)

    if not match:
        match = re.search(r'Optimal cost\s+([\d.e+-]+)', output_text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None

def extract_and_execute_python_code(text_content):
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("No Python code blocks found.")
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            print("Found an empty Python code block, skipped.")
            continue

        print("Found Python code block, starting execution...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print("Python code executed successfully, output:\n")
                print(result.stdout)
                best_obj = extract_best_objective(result.stdout)
                if best_obj is not None:
                    print(f"\nOptimal solution value (Best objective): {best_obj}")
                else:
                    print("\nOptimal solution value not found")
                return True, str(best_obj)
            else:
                print("Python code execution error, error message:\n")
                print(result.stderr)
                return False, result.stderr

        except Exception as e:
            print(f"Error occurred while executing Python code block: {e}")
            return False, str(e)
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            print("-" * 30)

    return False, "No valid code blocks executed"

def eval_model_result(success, result, ground_truth, err_rate=0.005):
    pass_flag = False
    correct_flag = False
    if success:
        pass_flag = True
        if is_number_string(str(result)) and ground_truth is not None:
            result_num = convert_to_number(str(result))
            ground_truth_num = convert_to_number(str(ground_truth))
            if result_num is not None and ground_truth_num is not None:
                if ground_truth_num == 0:
                    if result_num == 0:
                        correct_flag = True
                else:
                    deviation = abs(result_num - ground_truth_num) / abs(ground_truth_num)
                    if deviation <= err_rate:
                        correct_flag = True
        elif result == 'None':
            if ground_truth is None or ground_truth == 'None':
                correct_flag = True
    return pass_flag, correct_flag

def log_llm_chat(messages, model_name, response_text):
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "messages": messages,
            "response": response_text
        }
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(record)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to write chat log: {e}")

def save_generated_code(text_content, prefix="solve"):
    try:
        global CODE_SAVE_COUNTER
        CODE_SAVE_COUNTER += 1
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
        if CURRENT_QUESTION_ID is None:
            qid_part = "unknown"
        else:
            qid_part = str(CURRENT_QUESTION_ID)
        filename = f"{prefix}_q{qid_part}_{CODE_SAVE_COUNTER:04d}_{timestamp}.{extension}"
        file_path = os.path.join(RESULT_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        print(f"Warning: failed to save generated code: {e}")

def query_llm(messages, model_name="", temperature=0.2):
    if not hasattr(query_llm, "_openai_client") or not hasattr(query_llm, "_anthropic_client"):
        load_dotenv()

        openai_api_data = dict(
            api_key=os.getenv("", ""),
            base_url=os.getenv("", "")
        )
        anthropic_api_data = dict(
            api_key=os.getenv("")
        )

        query_llm._openai_client = openai.OpenAI(
            api_key=openai_api_data["api_key"],
            base_url=openai_api_data["base_url"] if openai_api_data["base_url"] else None
        )
        query_llm._anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_data["api_key"]
        )

    openai_client = query_llm._openai_client
    anthropic_client = query_llm._anthropic_client

    if model_name.lower().startswith("claude"):
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"]

        conversation = system_message + "\n\n"
        for user_msg, asst_msg in zip_longest(user_messages, assistant_messages, fillvalue=None):
            if user_msg:
                conversation += f"Human: {user_msg}\n\n"
            if asst_msg:
                conversation += f"Assistant: {asst_msg}\n\n"

        if len(user_messages) > len(assistant_messages):
            conversation += f"Human: {user_messages[-1]}\n\n"

        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=temperature,
            messages=[{"role": "user", "content": conversation}]
        )
        response_text = response.content[0].text
        log_llm_chat(messages, model_name, response_text)
        return response_text

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    response_text = response.choices[0].message.content
    log_llm_chat(messages, model_name, response_text)
    return response_text

def load_dataset():
    dataset = {}
    data_path = r"c:\Users\Bryt\Desktop\end\new_agent\dataset\IndustryOR_test.json"

    with open(data_path, 'r', encoding='utf-8') as f:
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
                        print(f"Warning: Could not parse line {line_num}: {line}")
                        continue
        else:
            dataset = json.load(f)

    return dataset

class ORLLMAgent:
    def __init__(self, model_name="", max_retries=3):
        self.model_name = model_name
        self.max_retries = max_retries
        
    def stage1_analysis(self, problem_text):
        """阶段1：问题分析与建模"""
        system_prompt = """你是一个运筹优化专家。请对以下优化问题进行建模分析。

输出格式必须严格遵循：
### 决策变量
列出所有决策变量及其定义

### 目标函数
给出目标函数的数学表达式

### 约束条件
列出所有约束条件的数学表达式

### 数学模型
给出完整的数学模型（包括变量、目标函数、约束条件）"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请对以下优化问题进行建模：\n\n{problem_text}"}
        ]
        
        response = query_llm(messages, model_name=self.model_name)
        return response
    
    def stage2_solver_design(self, stage1_output):
        """阶段2：求解器算法设计"""
        system_prompt = """你是一个Gurobi优化专家。基于阶段1的数学模型，设计Gurobi求解方案。

输出格式必须严格遵循：
### 导入模块
import gurobipy as gp
from gurobipy import GRB

### 参数与数据
（根据问题需要定义参数和数据）

### 模型构建
1. 创建模型：m = gp.Model()
2. 定义决策变量：使用m.addVar()或m.addVars()
3. 设置目标函数：m.setObjective()
4. 添加约束：m.addConstr()或m.addConstrs()

### 求解与输出
1. 求解模型：m.optimize()
2. 输出结果：打印最优解和目标值"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"基于以下数学模型，设计Gurobi求解方案：\n\n{stage1_output}"}
        ]
        
        response = query_llm(messages, model_name=self.model_name)
        return response
    
    def stage3_code_generation(self, stage1_output, stage2_output):
        """阶段3：代码生成"""
        system_prompt = """你是一个Python程序员。基于阶段1和阶段2的输出，生成完整的可执行Python代码。

要求：
1. 代码必须完整，可以直接运行
2. 必须包含所有必要的导入
3. 必须有清晰的数据结构定义
4. 必须有完整的模型构建和求解过程
5. 必须输出最优目标值（如果存在）

输出格式：
```python
# 完整Python代码
import gurobipy as gp
from gurobipy import GRB

# ... 完整代码 ...
```"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"阶段1输出：\n{stage1_output}\n\n阶段2输出：\n{stage2_output}\n\n请生成完整的Python代码："}
        ]
        
        response = query_llm(messages, model_name=self.model_name)
        return response
    
    def solve_problem(self, problem_text):
        """完整三阶段求解流程"""
        # 阶段1：问题分析与建模
        print("=" * 50)
        print("阶段1：问题分析与建模")
        print("=" * 50)
        stage1_result = self.stage1_analysis(problem_text)
        print(stage1_result)
        
        # 阶段2：求解器算法设计
        print("\n" + "=" * 50)
        print("阶段2：求解器算法设计")
        print("=" * 50)
        stage2_result = self.stage2_solver_design(stage1_result)
        print(stage2_result)
        
        # 阶段3：代码生成与执行修复
        print("\n" + "=" * 50)
        print("阶段3：代码生成与执行修复")
        print("=" * 50)
        
        for retry in range(self.max_retries):
            print(f"\n--- 第{retry + 1}次代码生成尝试 ---")
            
            # 生成代码
            stage3_result = self.stage3_code_generation(stage1_result, stage2_result)
            print(stage3_result)
            
            # 保存生成的代码
            save_generated_code(stage3_result, prefix="or_agent")
            
            # 执行代码
            success, result = extract_and_execute_python_code(stage3_result)
            
            if success:
                print(f"代码执行成功！结果：{result}")
                return success, result
            else:
                print(f"代码执行失败，错误：{result}")
                
                if retry < self.max_retries - 1:
                    print("尝试修复重试...")
                    # 修复提示
                    repair_prompt = f"""之前的代码执行失败，错误信息如下：
{result}

请修复代码中的问题，重新生成完整的可执行Python代码。

要求：
1. 修复所有错误
2. 保持代码完整性
3. 输出格式必须是 ```python ... ```"""

                    messages = [
                        {"role": "system", "content": "你是一个Python程序员，需要修复代码错误。"},
                        {"role": "user", "content": f"原始问题：{problem_text}\n\n阶段1输出：{stage1_result}\n\n阶段2输出：{stage2_result}\n\n错误信息：{result}\n\n{repair_prompt}"}
                    ]
                    
                    stage3_result = query_llm(messages, model_name=self.model_name)
                else:
                    print("达到最大重试次数，放弃修复。")
                    return success, result
        
        return False, "所有重试都失败"

def run_eval(dataset, use_agent=True, model_name=""):
    """评测函数"""
    global CURRENT_QUESTION_ID
    
    pass_count = 0
    correct_count = 0
    error_datas = []
    
    if use_agent:
        agent = ORLLMAgent(model_name=model_name)
    
    for qid, data in dataset.items():
        print(f"\n{'='*60}")
        print(f"处理问题 ID: {qid}")
        print(f"难度: {data['difficulty']}")
        print(f"问题: {data['question'][:200]}...")
        print(f"答案: {data['answer']}")
        print(f"{'='*60}")
        
        CURRENT_QUESTION_ID = qid
        
        try:
            if use_agent:
                # 使用智能体求解
                success, result = agent.solve_problem(data['question'])
            else:
                # 直接使用答案（仅用于测试）
                success = True
                result = data['answer']
            
            # 评估结果
            pass_flag, correct_flag = eval_model_result(
                success, result, data['answer'], err_rate=0.005
            )
            
            if pass_flag:
                pass_count += 1
                print(f"[ID {qid}] 跑通: 是")
            else:
                print(f"[ID {qid}] 跑通: 否")
            
            if correct_flag:
                correct_count += 1
                print(f"[ID {qid}] 正确: 是")
            else:
                print(f"[ID {qid}] 正确: 否")
                
            if not pass_flag or not correct_flag:
                error_datas.append({
                    'id': qid,
                    'success': success,
                    'result': result,
                    'expected': data['answer'],
                    'pass_flag': pass_flag,
                    'correct_flag': correct_flag
                })
                
        except Exception as e:
            print(f"[ID {qid}] 异常: {str(e)}")
            error_datas.append({
                'id': qid,
                'error': str(e)
            })
    
    return pass_count, correct_count, error_datas

def main():
    """主函数"""
    print("开始加载数据集...")
    dataset = load_dataset()
    
    print(f"数据集加载完成，共 {len(dataset)} 个问题")
    
    print("\n开始评测...")
    pass_count, correct_count, error_datas = run_eval(
        dataset, 
        use_agent=True,
        model_name=""
    )
    
    # 输出统计结果
    print(f"\n{'='*60}")
    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")

if __name__ == "__main__":
    main()

