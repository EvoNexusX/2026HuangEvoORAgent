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
    """
    Determine if a string is a numeric string, including integers and decimals.

    Args:
    s: The string to be checked.

    Returns:
    True if the string is a numeric string, otherwise False.
    """
    pattern = r"^[-+]?\d+(\.\d+)?$"
    return re.match(pattern, s) is not None


def convert_to_number(s):
    """
    Convert a string to a number (integer or float).

    Args:
        s: The string to be converted.

    Returns:
        int or float: Returns int if the string represents an integer, float if it represents a decimal.
        Returns None if conversion fails.
    """
    try:
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    """
    Extract Best objective or Optimal objective value from Gurobi output.

    Args:
        output_text: Gurobi output text

    Returns:
        float or None: Optimal solution value, returns None if not found
    """
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
    """
    从文本中提取Python代码块并执行。

    Args:
        text_content: 包含代码块的文本内容。

    Returns:
        bool: 执行成功返回True，否则返回False
        str: 失败时返回错误信息，成功时返回最优目标值
    """
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
                    # Relative deviation is undefined for zero denominator; require exact match.
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
        save_thought_chain(messages, model_name, response_text)
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


