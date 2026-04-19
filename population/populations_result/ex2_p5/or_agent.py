import copy
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
CODE_SAVE_COUNTER = 0
CURRENT_QUESTION_ID = None

def is_number_string(s):
    pattern = r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$"
    return re.match(pattern, s) is not None

def convert_to_number(s):
    try:
        if s is None:
            return None
        s = str(s)
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    if "Model is infeasible" in output_text or "INFEASIBLE" in output_text.upper():
        return None

    patterns = [
        r'Best objective\s*[:=]?\s*([\d.eE+-]+)',
        r'Optimal objective\s*[:=]?\s*([\d.eE+-]+)',
        r'Optimal cost\s*[:=]?\s*([\d.eE+-]+)',
        r'Objective value\s*[:=]?\s*([\d.eE+-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None

def extract_and_execute_python_code(text_content):
    python_code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text_content)
    if not python_code_blocks:
        python_code_blocks = re.findall(r'```\s*([\s\S]*?)```', text_content)
    
    if not python_code_blocks:
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            continue

        print("Executing Python code...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            env = os.environ.copy()
            env['PYTHONPATH'] = WORKSPACE_ROOT
            
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=30
            )

            if result.returncode == 0:
                print("Execution successful")
                best_obj = extract_best_objective(result.stdout)
                if best_obj is not None:
                    print(f"Optimal solution value: {best_obj}")
                return True, str(best_obj)
            else:
                error_msg = f"Return code: {result.returncode}\n"
                if result.stderr:
                    error_msg += f"Stderr: {result.stderr[:500]}"
                return False, error_msg

        except subprocess.TimeoutExpired:
            return False, "Execution timeout (30s)"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return False, "No valid code blocks executed"

def eval_model_result(success, result, ground_truth, err_rate=0.01):
    pass_flag = False
    correct_flag = False
    
    if success:
        pass_flag = True
        if result is None or str(result).strip() == "":
            correct_flag = False
        elif ground_truth is None or str(ground_truth).strip() == "":
            correct_flag = (result is None or str(result) == "None")
        else:
            result_num = convert_to_number(str(result))
            ground_truth_num = convert_to_number(str(ground_truth))
            
            if result_num is not None and ground_truth_num is not None:
                if ground_truth_num == 0:
                    correct_flag = (abs(result_num) < 1e-10)
                else:
                    deviation = abs(result_num - ground_truth_num) / abs(ground_truth_num)
                    correct_flag = (deviation <= err_rate)
    
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
    except Exception:
        pass

def save_generated_code(text_content, prefix="solve"):
    try:
        global CODE_SAVE_COUNTER
        CODE_SAVE_COUNTER += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text_content)
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
    except Exception:
        pass

def query_llm(messages, model_name="", temperature=0.1):
    if not hasattr(query_llm, "_clients_initialized"):
        load_dotenv()
        openai_api_key = os.getenv("")
        openai_base_url = os.getenv("", "")
        anthropic_api_key = os.getenv("")
        
        query_llm._openai_client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        ) if openai_api_key else None
        
        query_llm._anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key
        ) if anthropic_api_key else None
        
        query_llm._clients_initialized = True

    if model_name.lower().startswith("claude"):
        if query_llm._anthropic_client is None:
            raise ValueError("Claude API key not configured")
        
        system_prompt = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            elif msg["role"] == "user":
                conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                conversation.append({"role": "assistant", "content": msg["content"]})
        
        response = query_llm._anthropic_client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=temperature,
            system=system_prompt,
            messages=conversation
        )
        response_text = response.content[0].text
    else:
        if query_llm._openai_client is None:
            raise ValueError("OpenAI API key not configured")
        
        response = query_llm._openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=8192
        )
        response_text = response.choices[0].message.content
    
    log_llm_chat(messages, model_name, response_text)
    return response_text

def load_dataset():
    # 创建dataset目录（如果不存在）
    dataset_dir = os.path.join(WORKSPACE_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 数据集文件路径
    data_path = os.path.join(dataset_dir, "IndustryOR_test.json")
    
    # 如果数据集文件不存在，创建一个简单的示例数据集
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Creating sample dataset...")
        sample_data = [
            {
                "id": 1,
                "en_question": "Minimize: 2x + 3y\nSubject to:\nx + y >= 10\nx >= 0\ny >= 0\nx, y integers",
                "cn_question": "最小化: 2x + 3y\n约束条件:\nx + y >= 10\nx >= 0\ny >= 0\nx, y为整数",
                "en_answer": "20",
                "cn_answer": "20",
                "difficulty": "Easy"
            },
            {
                "id": 2,
                "en_question": "Maximize: 3x + 4y\nSubject to:\nx + 2y <= 10\n2x + y <= 10\nx >= 0\ny >= 0",
                "cn_question": "最大化: 3x + 4y\n约束条件:\nx + 2y <= 10\n2x + y <= 10\nx >= 0\ny >= 0",
                "en_answer": "22.5",
                "cn_answer": "22.5",
                "difficulty": "Medium"
            }
        ]
        
        # 保存示例数据集
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"Sample dataset created at {data_path}")
    
    # 加载数据集
    dataset = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                dataset = data
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    dataset[str(idx)] = {
                        'question': item.get('en_question', item.get('cn_question', '')),
                        'answer': item.get('en_answer', item.get('cn_answer', '')),
                        'difficulty': item.get('difficulty', 'Unknown'),
                        'id': item.get('id', idx)
                    }
        except json.JSONDecodeError:
            f.seek(0)
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        dataset[str(line_num)] = {
                            'question': item.get('en_question', item.get('cn_question', '')),
                            'answer': item.get('en_answer', item.get('cn_answer', '')),
                            'difficulty': item.get('difficulty', 'Unknown'),
                            'id': item.get('id', line_num)
                        }
                    except json.JSONDecodeError:
                        continue
    
    return dataset

def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)
    
    for attempt in range(max_attempts):
        gurobi_code = query_llm(messages, model_name)
        print(f"【Generated Code Attempt {attempt+1}】")
        save_generated_code(gurobi_code, prefix="agent")
        
        success, error_msg = extract_and_execute_python_code(gurobi_code)
        
        if success:
            messages_bak.append({"role": "assistant", "content": gurobi_code})
            return True, error_msg, messages_bak
        
        print(f"Execution failed: {error_msg}")
        
        if attempt < max_attempts - 1:
            print(f"Requesting code fix (attempt {attempt+1}/{max_attempts})...")
            messages.append({"role": "assistant", "content": gurobi_code})
            messages.append({
                "role": "user",
                "content": f"Code execution failed with error:\n{error_msg}\n\nPlease fix the code and provide complete executable Python code. Focus on fixing the specific error."
            })
    
    messages_bak.append({"role": "assistant", "content": gurobi_code})
    return False, None, messages_bak

def or_llm_agent(user_question, model_name="", max_attempts=3):
    stage1_messages = [
        {
            "role": "system",
            "content": """You are an operations research expert. Analyze the optimization problem carefully.

Output requirements:
1. Identify decision variables with clear notation
2. Formulate objective function mathematically
3. List all constraints with mathematical expressions
4. Specify variable types (continuous, integer, binary) and bounds
5. Use standard LP/MILP notation

Keep analysis concise and precise."""
        },
        {
            "role": "user",
            "content": f"Problem:\n{user_question}"
        }
    ]
    
    stage1_modeling = query_llm(stage1_messages, model_name)
    print("【Stage 1 - Mathematical Model】")
    
    stage2_messages = [
        {
            "role": "system",
            "content": """Design Gurobi implementation strategy based on the mathematical model.

Output requirements:
1. Map mathematical variables to Gurobi variables with proper types
2. Specify how to add objective function to model
3. Detail constraint addition process
4. Mention any special settings (e.g., time limit, MIP gap)
5. Consider numerical stability and model scaling

Provide clear implementation plan."""
        },
        {
            "role": "user",
            "content": f"Mathematical Model:\n{stage1_modeling}"
        }
    ]
    
    stage2_design = query_llm(stage2_messages, model_name)
    print("【Stage 2 - Implementation Design】")
    
    final_messages = [
        {
            "role": "system",
            "content": """Generate complete, executable Gurobi Python code.

Requirements:
1. Include all necessary imports
2. Define model with appropriate name
3. Create variables with correct types and bounds
4. Set objective function (minimize/maximize)
5. Add all constraints
6. Include model.optimize() call
7. Extract and print optimal objective value
8. Handle potential errors (infeasibility, unboundedness)
9. Add brief comments for key sections

Format code inside ```python ``` blocks."""
        },
        {
            "role": "user",
            "content": f"""Problem Description:
{user_question}

Mathematical Model:
{stage1_modeling}

Implementation Design:
{stage2_design}

Generate complete Python code using Gurobi."""
        }
    ]
    
    is_solve_success, result, final_messages = generate_or_code_solver(final_messages, model_name, max_attempts)
    
    if not is_solve_success:
        print("Initial solving failed, attempting recovery...")
        recovery_messages = copy.deepcopy(final_messages)
        recovery_messages.append({
            "role": "user",
            "content": "The code failed to execute. Please review the mathematical model and implementation design for potential issues. Then generate corrected code."
        })
        
        is_solve_success, result, _ = generate_or_code_solver(recovery_messages, model_name, max_attempts=2)
    
    return is_solve_success, result

def run_eval(use_agent=True, model_name=""):
    dataset = load_dataset()
    
    pass_count = 0
    correct_count = 0
    error_datas = []
    
    for i, d in dataset.items():
        global CURRENT_QUESTION_ID
        CURRENT_QUESTION_ID = i
        
        print(f"\n{'='*50}")
        print(f"Processing question {i}")
        print(f"{'='*50}")
        
        user_question = d["question"]
        answer = d["answer"]
        
        print(f"Question: {user_question[:200]}...")
        
        if use_agent:
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        else:
            is_solve_success = False
            llm_result = None
        
        pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
        
        pass_count += 1 if pass_flag else 0
        correct_count += 1 if correct_flag else 0
        
        if not pass_flag or not correct_flag:
            error_datas.append(i)
        
        print(f"Result: {'PASS' if pass_flag else 'FAIL'}, {'CORRECT' if correct_flag else 'INCORRECT'}")
        print(f"LLM output: {llm_result}")
        print(f"Expected: {answer}")
    
    print(f"\n{'='*50}")
    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")
    
    # 确保输出两行总评
    print(f"\n[Final Summary] Total questions: {len(dataset)}, Passed: {pass_count}, Correct: {correct_count}")
    print(f"[Error Summary] Failed questions: {error_datas}")

if __name__ == "__main__":
    USE_AGENT = True
    MODEL_NAME = ""
    
    run_eval(use_agent=USE_AGENT, model_name=MODEL_NAME)

