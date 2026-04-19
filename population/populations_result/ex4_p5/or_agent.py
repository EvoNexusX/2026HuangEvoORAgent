import copy
import re
import subprocess
import sys
import tempfile
import os
import json
import math
from datetime import datetime
from itertools import zip_longest

import openai
import anthropic
from dotenv import load_dotenv

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"llm_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

RESULT_DIR = os.path.join(WORKSPACE_ROOT, "generated")
os.makedirs(RESULT_DIR, exist_ok=True)
CODE_COUNTER = 0
CURRENT_QID = None

def verify_numeric_string(s):
    pattern = r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$"
    return re.match(pattern, s) is not None

def to_numeric(s):
    try:
        s = str(s).strip()
        if s.lower() == "none" or s == "":
            return None
        if '.' in s or 'e' in s.lower():
            val = float(s)
        else:
            val = int(s)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None

def extract_optimal_value(text_output):
    if "infeasible" in text_output.lower() or "unbounded" in text_output.lower():
        return None

    patterns = [
        r'Optimal solution found.*?objective value\s*[:=]?\s*([\d.e+-]+)',
        r'best objective\s*[:=]?\s*([\d.e+-]+)',
        r'objective value\s*[:=]?\s*([\d.e+-]+)',
        r'optimal value\s*[:=]?\s*([\d.e+-]+)',
        r'cost\s*[:=]?\s*([\d.e+-]+)',
        r'Result\s*[:=]?\s*([\d.e+-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    lines = text_output.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        if verify_numeric_string(line):
            try:
                return float(line)
            except ValueError:
                continue
    
    return None

def execute_python_extract(text):
    code_sections = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text)
    
    if not code_sections:
        lines = text.split('\n')
        code_lines = []
        code_started = False
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith('import ') or trimmed.startswith('from ') or trimmed.startswith('def ') or trimmed.startswith('class ') or trimmed.startswith('print('):
                code_started = True
            if code_started:
                code_lines.append(line)
        if code_lines:
            code_sections = ['\n'.join(code_lines)]
    
    if not code_sections:
        return False, "未找到有效代码段"

    for code in code_sections:
        code = code.strip()
        if not code:
            continue

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            env = os.environ.copy()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')
            
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=45,
                env=env,
                check=False
            )

            if result.returncode == 0:
                best_val = extract_optimal_value(result.stdout)
                if best_val is not None:
                    return True, str(best_val)
                else:
                    return True, "None"
            else:
                error_out = result.stderr if result.stderr else result.stdout
                return False, error_out[:500]

        except subprocess.TimeoutExpired:
            return False, "执行超时 (45秒)"
        except Exception as e:
            return False, str(e)
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return False, "无可用代码块执行成功"

def evaluate_result(success, output, ground_truth, tolerance=0.001):
    passed = False
    correct = False
    
    if success:
        passed = True
        
        if output == 'None' and (ground_truth is None or ground_truth == 'None'):
            correct = True
            return passed, correct
        
        if ground_truth is None or ground_truth == 'None':
            return passed, correct
        
        num_output = to_numeric(str(output))
        num_truth = to_numeric(str(ground_truth))
        
        if num_output is None or num_truth is None:
            return passed, correct
        
        if abs(num_truth) < 1e-10:
            if abs(num_output) < 1e-10:
                correct = True
        else:
            relative_error = abs(num_output - num_truth) / abs(num_truth)
            if relative_error <= tolerance:
                correct = True
    
    return passed, correct

def record_llm_interaction(messages, model_used, response):
    try:
        entry = {
            "time": datetime.now().isoformat(),
            "model": model_used,
            "messages": messages,
            "reply": response
        }
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
        existing.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def store_generated_code(content, label="agent"):
    try:
        global CODE_COUNTER
        CODE_COUNTER += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_parts = re.findall(r'```(?:python)?\s*([\s\S]*?)```', content)
        if not code_parts:
            code_parts = [content]
        
        combined = "\n\n".join(part.strip() for part in code_parts if part.strip())
        if not combined:
            combined = content
        
        ext = "py"
        if CURRENT_QID is None:
            qid = "unknown"
        else:
            qid = str(CURRENT_QID)
        fname = f"{label}_q{qid}_{CODE_COUNTER:04d}_{timestamp}.{ext}"
        full_path = os.path.join(RESULT_DIR, fname)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(combined)
    except Exception:
        pass

def call_llm_api(messages, model_selected="", temp=0.3):
    if not hasattr(call_llm_api, "_openai_client") or not hasattr(call_llm_api, "_claude_client"):
        load_dotenv()
        
        deepseek_config = dict(
            api_key=os.getenv("", ""),
            base_url=os.getenv("", "")
        )
        claude_config = dict(
            api_key=os.getenv("")
        )
        
        call_llm_api._openai_client = openai.OpenAI(
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config["base_url"] if deepseek_config["base_url"] else None
        )
        call_llm_api._claude_client = anthropic.Anthropic(
            api_key=claude_config["api_key"]
        )
    
    openai_client = call_llm_api._openai_client
    claude_client = call_llm_api._claude_client
    
    if model_selected.lower().startswith("claude"):
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        
        full_convo = sys_msg + "\n\n"
        for u_msg, a_msg in zip_longest(user_msgs, assistant_msgs, fillvalue=None):
            if u_msg:
                full_convo += f"Human: {u_msg}\n\n"
            if a_msg:
                full_convo += f"Assistant: {a_msg}\n\n"
        
        if len(user_msgs) > len(assistant_msgs):
            full_convo += f"Human: {user_msgs[-1]}\n\n"
        
        response = claude_client.messages.create(
            model=model_selected,
            max_tokens=8192,
            temperature=temp,
            messages=[{"role": "user", "content": full_convo}]
        )
        reply_text = response.content[0].text
        record_llm_interaction(messages, model_selected, reply_text)
        return reply_text
    
    response = openai_client.chat.completions.create(
        model=model_selected,
        messages=messages,
        temperature=temp
    )
    reply_text = response.choices[0].message.content
    record_llm_interaction(messages, model_selected, reply_text)
    return reply_text

def load_test_set():
    dataset = {}
    data_file = r"c:\Users\Bryt\Desktop\end\new_agent\dataset\IndustryOR_test.json"
    
    with open(data_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        if first_line.startswith('{"en_question"') or first_line.startswith('{"cn_question"'):
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        dataset_entry = {
                            'question': item.get('en_question', item.get('cn_question', '')),
                            'answer': item.get('en_answer', item.get('cn_answer', '')),
                            'difficulty': item.get('difficulty', 'Unknown'),
                            'id': item.get('id', idx - 1)
                        }
                        dataset[str(dataset_entry['id'])] = dataset_entry
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                dataset = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            dataset_entry = {
                                'question': item.get('en_question', item.get('cn_question', '')),
                                'answer': item.get('en_answer', item.get('cn_answer', '')),
                                'difficulty': item.get('difficulty', 'Unknown'),
                                'id': item.get('id', idx - 1)
                            }
                            dataset[str(dataset_entry['id'])] = dataset_entry
                        except json.JSONDecodeError:
                            continue
    
    return dataset

def build_solver_code(initial_messages, model_choice, max_retries):
    messages = copy.deepcopy(initial_messages)
    
    code_response = call_llm_api(messages, model_choice)
    store_generated_code(code_response, label="agent")
    
    full_text = f"{code_response}"
    for attempt in range(max_retries):
        success, error_info = execute_python_extract(full_text)
        if success:
            initial_messages.append({"role": "assistant", "content": code_response})
            return True, error_info, initial_messages
        
        messages.append({"role": "assistant", "content": code_response})
        messages.append({
            "role": "user",
            "content": f"执行代码时遇到错误：\n{error_info}\n请检查并修正代码，然后重新提供完整可运行的代码。",
        })
        
        code_response = call_llm_api(messages, model_choice)
        store_generated_code(code_response, label="agent_retry")
        full_text = f"{code_response}"
    
    initial_messages.append({"role": "assistant", "content": code_response})
    return False, None, initial_messages

def optimization_agent(user_problem, model_choice="", max_attempts=5):
    phase1_prompt = [
        {
            "role": "system",
            "content": (
                "第一阶段：问题分析与数学建模。\n"
                "你是一个专业的运筹学顾问。请仔细阅读下面的优化问题，识别其中的决策变量、目标函数和所有约束条件。\n"
                "用清晰的数学表达式（线性规划形式）建立精确的数学模型。确保变量定义明确，目标函数方向正确，约束完整。\n"
                "输出时重点呈现数学模型，无需额外解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"优化问题：\n{user_problem}"
            ),
        },
    ]
    modeling_result = call_llm_api(phase1_prompt, model_choice)
    
    phase2_prompt = [
        {
            "role": "system",
            "content": (
                "第二阶段：算法设计与求解策略。\n"
                "基于上述数学模型，设计使用Gurobi求解器的具体算法流程。\n"
                "说明如何初始化模型、添加变量、设置目标函数、添加约束、配置求解参数，以及如何提取结果。\n"
                "输出应侧重于算法步骤和关键考虑点，确保后续代码实现无误。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"数学模型：\n{modeling_result}"
            ),
        },
    ]
    algorithm_design = call_llm_api(phase2_prompt, model_choice)
    
    final_messages = [
        {
            "role": "system",
            "content": (
                "请根据提供的优化问题、数学模型和算法设计，编写完整的Gurobi Python求解代码。\n"
                "代码必须独立可运行，包含必要的import语句、模型构建、求解和结果输出部分。\n"
                "确保代码能处理可能的数值问题，并在最后打印或返回最优目标值。\n"
                "严格按照以下格式输出代码：```python\n<你的代码>\n```"
            ),
        },
        {
            "role": "user",
            "content": (
                f"原始问题：\n{user_problem}\n\n"
                f"数学模型：\n{modeling_result}\n\n"
                f"算法设计：\n{algorithm_design}\n\n"
            ),
        },
    ]
    
    success, result, final_messages = build_solver_code(final_messages, model_choice, max_attempts)
    
    if success and result == 'None':
        final_messages.append(
            {
                "role": "user",
                "content": (
                    "当前求解返回无可行解，请重新审视数学模型和算法设计，检查是否存在建模错误或约束过紧。\n"
                    "修正后重新生成完整的Gurobi Python代码。\n"
                    "输出格式：```python\n<修正后的代码>\n```"
                ),
            }
        )
        success, result, final_messages = build_solver_code(final_messages, model_choice, max_attempts=2)
    elif not success:
        final_messages.append(
            {
                "role": "user",
                "content": (
                    "多次尝试后代码仍无法正常运行。请彻底检查数学模型和算法设计的逻辑错误，然后重新编写代码。\n"
                    "输出格式：```python\n<重新编写的代码>\n```"
                ),
            }
        )
        success, result, final_messages = build_solver_code(final_messages, model_choice, max_attempts=3)
    
    return success, result

def run_eval(use_agent=True, model_name=""):
    dataset = load_test_set()
    
    pass_count = 0
    correct_count = 0
    error_datas = []
    
    for idx, data in dataset.items():
        global CURRENT_QID
        CURRENT_QID = idx
        print(f"=============== 问题 {idx} ==================")
        question, answer = data["question"], data["answer"]
        print(question)
        print("-------------")
        
        if use_agent:
            solved, llm_output = optimization_agent(question, model_name)
        else:
            solved, llm_output = False, None
        
        if solved:
            print(f"代码执行成功，最优解值: {llm_output}")
        else:
            print("代码执行失败。")
        print("------------------")
        
        passed, correct = evaluate_result(solved, llm_output, answer)
        pass_count += 1 if passed else 0
        correct_count += 1 if correct else 0
        
        if not passed or not correct:
            error_datas.append(idx)
        
        print(f"求解状态: {solved}, 模型输出: {llm_output}, 正确答案: {answer}")
        print(f"[当前结果] 运行通过: {passed}, 求解正确: {correct}")
        print("\n")
    
    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")

if __name__ == "__main__":
    AGENT_ENABLED = True
    SELECTED_MODEL = ""
    
    run_eval(use_agent=AGENT_ENABLED, model_name=SELECTED_MODEL)

