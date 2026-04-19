import re
import subprocess
import sys
import tempfile
import os
import json
import random
from datetime import datetime
from itertools import zip_longest
import copy

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

    patterns = [
        r'Best objective\s+([\d.e+-]+)',
        r'Optimal objective\s+([\d.e+-]+)',
        r'Optimal cost\s+([\d.e+-]+)',
        r'Optimal value:?\s*([\d.e+-]+)',
        r'Objective\s*=\s*([\d.e+-]+)',
        r'最优目标\s*[:：]?\s*([\d.e+-]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    numbers = re.findall(r'[-+]?\d*\.?\d+e?[-+]?\d*', output_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None

def extract_and_execute_python_code(text_content, max_retries=1):
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("未找到Python代码块，尝试直接执行。")
        python_code_blocks = [text_content]

    best_result = None
    best_error = None

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            continue

        print("开始执行代码块...")
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                    tmp_file.write(code_block)
                    temp_file_path = tmp_file.name

                # 增加超时时间到120秒，并添加更详细的错误处理
                result = subprocess.run([sys.executable, temp_file_path], 
                                      capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    print("代码执行成功，输出：")
                    print(result.stdout)
                    best_obj = extract_best_objective(result.stdout)
                    
                    if best_obj is not None:
                        print(f"解析到最优值：{best_obj}")
                        return True, str(best_obj)
                    else:
                        print("未找到最优值，但执行成功")
                        return True, "SUCCESS_NO_OBJ"
                else:
                    error_msg = f"执行错误（返回值{result.returncode}）：{result.stderr}"
                    print(error_msg)
                    best_error = error_msg
                    
            except subprocess.TimeoutExpired as e:
                error_msg = f"执行超时（120秒）：{str(e)}"
                print(error_msg)
                best_error = error_msg
            except Exception as e:
                error_msg = f"执行异常：{str(e)}"
                print(error_msg)
                best_error = error_msg
            finally:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
                print("-" * 30)
            
            retry_count += 1

    return False, best_error if best_error else "所有代码块执行失败"

def eval_model_result(success, result, ground_truth, err_rate=0.01):
    pass_flag = False
    correct_flag = False
    
    if success:
        pass_flag = True
        if result == "SUCCESS_NO_OBJ":
            correct_flag = False
        elif ground_truth is None or ground_truth == 'None':
            if result == 'None':
                correct_flag = True
        elif is_number_string(str(result)):
            result_num = convert_to_number(str(result))
            ground_truth_num = convert_to_number(str(ground_truth))
            if result_num is not None and ground_truth_num is not None:
                if ground_truth_num == 0:
                    if abs(result_num) < 1e-10:
                        correct_flag = True
                else:
                    deviation = abs(result_num - ground_truth_num) / abs(ground_truth_num)
                    if deviation <= err_rate:
                        correct_flag = True
    
    return pass_flag, correct_flag

def log_llm_chat(messages, model_name, response_text, stage_tag=""):
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "stage": stage_tag,
            "messages": messages[-3:] if len(messages) > 3 else messages,
            "response": response_text[:2000]
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
        print(f"日志记录失败：{e}")

def save_generated_code(text_content, prefix="solve", stage=""):
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
        
        stage_part = f"_{stage}" if stage else ""
        filename = f"{prefix}_q{qid_part}_{CODE_SAVE_COUNTER:04d}{stage_part}_{timestamp}.{extension}"
        file_path = os.path.join(RESULT_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"代码已保存至：{file_path}")
    except Exception as e:
        print(f"代码保存失败：{e}")

def query_llm(messages, model_name="", temperature=0.3, stage_tag=""):
    if not hasattr(query_llm, "_openai_client") or not hasattr(query_llm, "_anthropic_client"):
        load_dotenv()

        openai_api_data = dict(
            api_key=os.getenv("", ""),
            base_url=os.getenv("", "")
        )
        anthropic_api_data = dict(
            api_key=os.getenv("", "")
        )

        query_llm._openai_client = openai.OpenAI(
            api_key=openai_api_data["api_key"],
            base_url=openai_api_data["base_url"] if openai_api_data["base_url"] else None
        )
        query_llm._anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_data["api_key"]
        ) if anthropic_api_data["api_key"] else None

    if model_name.lower().startswith("claude"):
        if not query_llm._anthropic_client:
            print("警告：Claude API密钥未配置，切换到DeepSeek")
            model_name = ""
        else:
            system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m["content"] for m in messages if m["role"] == "user"]
            assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"]

            conversation = system_message + "\n\n" if system_message else ""
            for user_msg, asst_msg in zip_longest(user_messages, assistant_messages, fillvalue=None):
                if user_msg:
                    conversation += f"Human: {user_msg}\n\n"
                if asst_msg:
                    conversation += f"Assistant: {asst_msg}\n\n"

            if len(user_messages) > len(assistant_messages):
                conversation += f"Human: {user_messages[-1]}\n\n"

            try:
                response = query_llm._anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=8192,
                    temperature=temperature,
                    messages=[{"role": "user", "content": conversation}]
                )
                response_text = response.content[0].text
                log_llm_chat(messages, model_name, response_text, stage_tag)
                return response_text
            except Exception as e:
                print(f"Claude API错误：{e}，切换到DeepSeek")
                model_name = ""

    openai_client = query_llm._openai_client
    
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        response_text = response.choices[0].message.content
        log_llm_chat(messages, model_name, response_text, stage_tag)
        return response_text
    except Exception as e:
        print(f"API调用失败：{e}")
        raise

def load_dataset():
    dataset = {}
    possible_paths = [
        r"c:\Users\Bryt\Desktop\end\new_agent\dataset\IndustryOR_test.json",
        "./dataset/IndustryOR_test.json",
        "../dataset/IndustryOR_test.json",
        "dataset/IndustryOR_test.json"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("警告：未找到数据集文件，使用空数据集")
        return {}
    
    try:
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
                            print(f"解析错误第{line_num}行：{line}")
                            continue
            else:
                dataset = json.load(f)
                
    except Exception as e:
        print(f"加载数据集失败：{e}")
    
    return dataset

def generate_code_with_repair(messages_bak, model_name, max_attempts=3):
    messages = copy.deepcopy(messages_bak)
    
    for attempt in range(max_attempts):
        try:
            gurobi_code = query_llm(messages, model_name, stage_tag="code_gen")
            print(f"\n【第{attempt+1}次生成Gurobi代码】")
            save_generated_code(gurobi_code, prefix="or_agent", stage=f"gen{attempt+1}")
            
            success, result = extract_and_execute_python_code(gurobi_code)
            
            if success:
                messages_bak.append({"role": "assistant", "content": gurobi_code})
                return True, result, messages_bak
            
            print(f"执行失败，错误：{result}")
            
            if attempt < max_attempts - 1:
                messages.append({"role": "assistant", "content": gurobi_code})
                error_hint = f"""代码执行失败，错误信息：
{result}

请分析错误原因：
1. 如果是语法错误，请修正语法
2. 如果是变量未定义，请检查变量声明
3. 如果是Gurobi语法错误，请查阅Gurobi文档
4. 如果是模型不可行或无界，请调整约束
5. 如果执行超时，请简化模型或设置求解时间限制

修复后请提供完整可执行的Python代码："""
                messages.append({"role": "user", "content": error_hint})
                print("正在请求修复...")
                
        except Exception as e:
            print(f"生成代码过程中出错：{e}")
            if attempt < max_attempts - 1:
                continue
            else:
                return False, str(e), messages_bak
    
    print(f"已达到最大尝试次数{max_attempts}，执行失败")
    return False, "max_attempts_exceeded", messages_bak

class OptimizedORAgent:
    def __init__(self, model_name="", max_repair_attempts=3):
        self.model_name = model_name
        self.max_repair_attempts = max_repair_attempts
        
    def stage1_analysis(self, problem_text):
        system_prompt = """你是一个运筹优化专家。请分析以下优化问题，提取关键信息并建立数学模型。

分析步骤：
1. 识别决策变量（variables）
2. 明确目标函数（objective function）
3. 列出所有约束条件（constraints）
4. 判断问题类型（线性规划、整数规划、混合整数规划等）

请用规范的数学表达式描述模型，包括：
- 变量定义
- 目标函数（最大化或最小化）
- 约束条件

问题："""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text}
        ]
        
        response = query_llm(messages, model_name=self.model_name, stage_tag="stage1")
        return response
    
    def stage2_design(self, stage1_output):
        system_prompt = """基于上述数学模型，设计Gurobi求解方案。

设计要点：
1. 如何初始化Gurobi模型
2. 如何定义决策变量（连续/整数/二进制）
3. 如何设置目标函数
4. 如何添加约束条件
5. 如何设置求解参数（如时间限制、最优性容差）
6. 如何提取和输出结果

请提供详细的算法设计步骤。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"数学模型：\n{stage1_output}"}
        ]
        
        response = query_llm(messages, model_name=self.model_name, stage_tag="stage2")
        return response
    
    def stage3_validation(self, problem_text, stage1_output, stage2_output):
        system_prompt = """请检查上述数学模型和算法设计是否存在潜在问题：

常见问题检查清单：
1. 变量类型是否正确（整数变量误设为连续变量）
2. 目标函数方向是否正确（最大化/最小化）
3. 约束条件是否完整（是否遗漏重要约束）
4. 约束条件是否矛盾（可能导致不可行）
5. 模型规模是否过大（可能需要简化）

请指出潜在问题并提出改进建议。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题：{problem_text}\n\n数学模型：\n{stage1_output}\n\n算法设计：\n{stage2_output}"}
        ]
        
        response = query_llm(messages, model_name=self.model_name, stage_tag="stage3")
        return response
    
    def stage4_generation(self, problem_text, stage1_output, stage2_output, stage3_output):
        system_prompt = """请基于前三个阶段的分析，编写完整的Gurobi Python代码。

代码要求：
1. 包含完整的import语句
2. 添加清晰的注释说明
3. 处理异常情况（如不可行、无界）
4. 输出最优目标值和主要决策变量值
5. 使用适当的求解参数，特别是设置时间限制避免超时
6. 确保代码能够快速求解，对于复杂问题设置合理的时间限制

请按以下格式输出代码：
```python
# 导入库
import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model()

# ... 代码内容 ...

# 设置求解时间限制
model.setParam('TimeLimit', 60)  # 设置60秒时间限制

# 求解并输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优目标值: {model.objVal}")
else:
    print(f"求解状态: {model.status}")
```

注意：只输出代码，不要额外解释。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""问题描述：
{problem_text}

数学模型：
{stage1_output}

算法设计：
{stage2_output}

验证反馈：
{stage3_output}

请生成完整代码，特别注意设置求解时间限制："""}
        ]
        
        return messages
    
    def solve_problem(self, problem_text):
        print("\n" + "="*60)
        print("阶段1：问题分析与建模")
        print("="*60)
        stage1_result = self.stage1_analysis(problem_text)
        print(stage1_result[:500] + "..." if len(stage1_result) > 500 else stage1_result)
        
        print("\n" + "="*60)
        print("阶段2：Gurobi算法设计")
        print("="*60)
        stage2_result = self.stage2_design(stage1_result)
        print(stage2_result[:500] + "..." if len(stage2_result) > 500 else stage2_result)
        
        print("\n" + "="*60)
        print("阶段3：模型验证与改进")
        print("="*60)
        stage3_result = self.stage3_validation(problem_text, stage1_result, stage2_result)
        print(stage3_result[:500] + "..." if len(stage3_result) > 500 else stage3_result)
        
        print("\n" + "="*60)
        print("阶段4：代码生成与执行修复")
        print("="*60)
        
        messages = self.stage4_generation(problem_text, stage1_result, stage2_result, stage3_result)
        
        is_solve_success, result, messages = generate_code_with_repair(
            messages, self.model_name, self.max_repair_attempts)
        
        if not is_solve_success:
            print("代码执行失败，尝试简化问题重试...")
            simplified_prompt = f"""原问题：{problem_text}

之前的尝试失败了，特别是执行超时。请生成一个更简单但能解决问题的Gurobi代码。重点关注：
1. 只实现核心的数学模型
2. 使用默认求解参数，但必须设置时间限制（TimeLimit=60）
3. 简化变量和约束的定义
4. 确保代码能在60秒内完成求解

生成可直接执行的代码："""
            
            simple_messages = [
                {"role": "system", "content": "生成简单可执行的Gurobi代码，必须设置时间限制"},
                {"role": "user", "content": simplified_prompt}
            ]
            
            is_solve_success, result, _ = generate_code_with_repair(
                simple_messages, self.model_name, max_attempts=2)
        
        return is_solve_success, result

def run_eval(use_agent=True, model_name="", sample_size=None):
    dataset = load_dataset()
    
    if not dataset:
        print("数据集为空，无法进行评估")
        print(f"[Total 0] run pass: 0, solve correct: 0")
        print(f"[Total fails 0] error datas: []")
        return 0, 0, []
    
    if sample_size and sample_size < len(dataset):
        print(f"随机采样 {sample_size} 个问题进行测试")
        keys = random.sample(list(dataset.keys()), sample_size)
        dataset = {k: dataset[k] for k in keys}
    
    pass_count = 0
    correct_count = 0
    error_datas = []
    
    if use_agent:
        agent = OptimizedORAgent(model_name=model_name)
    
    for idx, (qid, data) in enumerate(dataset.items(), 1):
        global CURRENT_QUESTION_ID
        CURRENT_QUESTION_ID = qid
        print(f"\n{'='*60}")
        print(f"问题 {idx}/{len(dataset)} | ID: {qid}")
        print(f"难度: {data['difficulty']}")
        print(f"问题摘要: {data['question'][:150]}...")
        print(f"标准答案: {data['answer']}")
        print(f"{'='*60}")
        
        try:
            if use_agent:
                is_solve_success, llm_result = agent.solve_problem(data['question'])
            else:
                is_solve_success = True
                llm_result = data['answer']
            
            pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, data['answer'])
            pass_count += 1 if pass_flag else 0
            correct_count += 1 if correct_flag else 0
            
            if not pass_flag or not correct_flag:
                error_datas.append(qid)
            
            status_symbol = "✓" if correct_flag else "✗"
            print(f"[ID {qid}] {status_symbol} 执行: {'通过' if pass_flag else '失败'}, 正确: {'是' if correct_flag else '否'}")
            print(f"得到结果: {llm_result}, 期望: {data['answer']}")
            
        except Exception as e:
            print(f"[ID {qid}] 异常: {str(e)}")
            error_datas.append(qid)
    
    print(f"\n{'='*60}")
    print("评估结果汇总")
    print(f"{'='*60}")
    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")
    
    return pass_count, correct_count, error_datas

def main():
    print("优化智能体 - 运筹问题求解器")
    print("="*60)
    
    print("加载数据集中...")
    dataset = load_dataset()
    print(f"加载完成，共 {len(dataset)} 个问题")
    
    model_choice = input("选择模型 (1=DeepSeek, 2=, 默认DeepSeek): ").strip()
    if model_choice == "2":
        model_name = ""
    else:
        model_name = ""
    
    sample_input = input("输入测试样本数量 (默认全部): ").strip()
    sample_size = int(sample_input) if sample_input.isdigit() else None
    
    print(f"\n开始评测，使用模型: {model_name}")
    run_eval(use_agent=True, model_name=model_name, sample_size=sample_size)

if __name__ == "__main__":
    main()

