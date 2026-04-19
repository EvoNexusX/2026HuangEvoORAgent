import json
import os
import re
import sys
import tempfile
import subprocess
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加当前目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入gurobipy
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("警告: gurobipy 未安装，代码将无法运行")

# 模拟 new_utils 中的关键函数
def extract_best_objective(output_text):
    """
    从求解器输出中提取目标值
    """
    if "Model is infeasible" in output_text:
        return None
    
    # 尝试匹配不同的输出格式
    patterns = [
        r'Best objective\s+([\d.eE+-]+)',
        r'Optimal objective\s+([\d.eE+-]+)',
        r'Optimal cost\s+([\d.eE+-]+)',
        r'Optimal Value:\s*([\d.eE+-]+)',
        r'Objective value:\s*([\d.eE+-]+)',
        r'最佳目标值\s*[:：]\s*([\d.eE+-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # 尝试从最后几行提取数字
    lines = output_text.strip().split('\n')
    for line in reversed(lines[-10:]):  # 检查最后10行
        line = line.strip()
        # 检查是否为纯数字（可能包含小数点和科学计数法）
        if re.match(r'^[-+]?[\d.]+(?:[eE][-+]?\d+)?$', line):
            try:
                return float(line)
            except ValueError:
                continue
    
    return None

def extract_and_execute_python_code(text_content):
    """
    提取并执行Python代码块
    """
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)
    
    if not python_code_blocks:
        # 如果没有代码块，直接执行整个文本
        python_code_blocks = [text_content]
    
    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            continue
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_block)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                # 提取最优值
                best_obj = extract_best_objective(result.stdout)
                return True, str(best_obj) if best_obj is not None else "None"
            else:
                return False, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "代码执行超时"
        except Exception as e:
            return False, str(e)
    
    return False, "没有可执行的代码块"

def eval_model_result(success, result, ground_truth, err_rate=0.1):
    """
    评估模型结果
    """
    pass_flag = False
    correct_flag = False
    
    if success:
        pass_flag = True
        
        try:
            # 解析结果和真值
            result_str = str(result).strip()
            ground_truth_str = str(ground_truth).strip() if ground_truth is not None else None
            
            # 处理None情况
            if result_str == 'None' and ground_truth_str in [None, 'None', '无解', 'infeasible']:
                correct_flag = True
                return pass_flag, correct_flag
            
            # 尝试转换为数字
            try:
                result_val = float(result_str)
            except:
                # 如果不能转换为数字，尝试从字符串提取数字
                num_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', result_str)
                if num_match:
                    result_val = float(num_match.group())
                else:
                    return pass_flag, False
            
            if ground_truth_str:
                try:
                    truth_val = float(ground_truth_str)
                except:
                    # 从真值字符串提取数字
                    truth_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', ground_truth_str)
                    if truth_match:
                        truth_val = float(truth_match.group())
                    else:
                        return pass_flag, False
                
                # 计算相对误差
                if abs(truth_val) < 1e-10:  # 真值为0或接近0
                    if abs(result_val) < 1e-10:
                        correct_flag = True
                else:
                    relative_error = abs(result_val - truth_val) / abs(truth_val)
                    if relative_error <= err_rate:
                        correct_flag = True
        
        except Exception as e:
            print(f"评估结果时出错: {e}")
    
    return pass_flag, correct_flag

def save_generated_code(text_content, prefix="solve"):
    """
    保存生成的代码
    """
    try:
        os.makedirs("result", exist_ok=True)
        
        # 提取代码块
        code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)
        if code_blocks:
            code = "\n\n".join([cb.strip() for cb in code_blocks if cb.strip()])
            ext = "py"
        else:
            code = text_content
            ext = "txt"
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{ext}"
        filepath = os.path.join("result", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"代码已保存到: {filepath}")
        
    except Exception as e:
        print(f"保存代码时出错: {e}")

def query_llm(messages, model_name="", temperature=0.2):
    """
    模拟LLM查询 - 简化版本
    """
    # 这里模拟一个简单的响应
    # 在实际使用中，应该替换为真正的LLM API调用
    system_msg = ""
    user_msg = ""
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            user_msg = msg["content"]
    
    # 返回一个模拟的响应
    response = f"""基于以下输入，我生成了Python代码：

{user_msg}

```python
import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Optimization_Problem")

# 注意: 这里是一个示例，实际代码应该根据具体问题编写
# 由于没有具体问题数据，这里只提供框架

print("请提供具体的问题数据和约束条件")
print("然后设置变量、目标函数和约束")

# 示例: 求解一个简单的LP问题
# x = model.addVar(name="x")
# y = model.addVar(name="y")
# model.setObjective(x + y, GRB.MAXIMIZE)
# model.addConstr(x + 2*y <= 10, "c1")
# model.addConstr(3*x + y <= 15, "c2")
# model.optimize()
# if model.status == GRB.OPTIMAL:
#     print(f"最优值: {model.objVal}")
# else:
#     print("未找到最优解")
```
"""
    return response

def load_dataset():
    """
    加载数据集
    """
    # 查找数据集文件
    possible_paths = [
        "dataset/IndustryOR_test.json",
        "dataset/IndustryOR_test.jsonl",
        "IndustryOR_test.json",
        "IndustryOR_test.jsonl"
    ]
    
    dataset = {}
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.endswith('.jsonl'):
                        for i, line in enumerate(f):
                            if line.strip():
                                item = json.loads(line)
                                qid = str(item.get('id', i))
                                dataset[qid] = {
                                    'question': item.get('en_question') or item.get('cn_question') or '',
                                    'answer': item.get('en_answer') or item.get('cn_answer') or '',
                                    'difficulty': item.get('difficulty', 'Unknown'),
                                    'id': qid
                                }
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                qid = str(item.get('id', i))
                                dataset[qid] = {
                                    'question': item.get('en_question') or item.get('cn_question') or '',
                                    'answer': item.get('en_answer') or item.get('cn_answer') or '',
                                    'difficulty': item.get('difficulty', 'Unknown'),
                                    'id': qid
                                }
                        elif isinstance(data, dict):
                            dataset = data
                
                print(f"从 {path} 加载了 {len(dataset)} 个问题")
                return dataset
                
            except Exception as e:
                print(f"加载数据集 {path} 时出错: {e}")
    
    # 如果没有找到文件，创建示例数据
    print("警告: 未找到数据集文件，使用示例数据")
    dataset = {
        "1": {
            "question": "最小化 2x + 3y，满足 x + y >= 10, x >= 0, y >= 0",
            "answer": "20",
            "difficulty": "Easy",
            "id": "1"
        },
        "2": {
            "question": "最大化 3x + 4y，满足 2x + y <= 10, x + 2y <= 8, x >= 0, y >= 0",
            "answer": "18",
            "difficulty": "Medium",
            "id": "2"
        }
    }
    return dataset

# 设置全局变量
CURRENT_QUESTION_ID = None

class OroptAgent:
    """运筹优化智能体"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    def stage1_problem_analysis(self, problem_text):
        """阶段1: 问题分析与建模"""
        prompt = f"""请分析以下运筹优化问题，并给出数学模型：

问题描述：
{problem_text}

请按以下格式输出：
1. 决策变量：
2. 目标函数：
3. 约束条件：
4. 数学模型（数学表达式）：

注意：只输出分析结果，不要输出代码。"""
        
        messages = [
            {"role": "system", "content": "你是一位运筹优化专家，擅长问题分析和数学建模。"},
            {"role": "user", "content": prompt}
        ]
        
        return query_llm(messages)
    
    def stage2_solver_design(self, problem_analysis):
        """阶段2: Gurobi求解器算法设计"""
        prompt = f"""基于以下数学模型，设计Gurobi求解方案：

{problem_analysis}

请按以下格式输出求解方案：
1. 模型类型（LP/MIP/QP等）：
2. 变量定义方法：
3. 目标函数设置：
4. 约束添加方法：
5. 求解参数建议：
6. 结果提取方法：

注意：只输出求解方案，不要输出完整代码。"""
        
        messages = [
            {"role": "system", "content": "你是一位Gurobi求解器专家，擅长设计优化求解方案。"},
            {"role": "user", "content": prompt}
        ]
        
        return query_llm(messages)
    
    def stage3_code_generation(self, problem_text, problem_analysis, solver_design, question_id=None):
        """阶段3: 代码生成"""
        global CURRENT_QUESTION_ID
        CURRENT_QUESTION_ID = question_id
        
        prompt = f"""请根据以下信息生成完整的Python代码：

问题描述：
{problem_text}

数学模型分析：
{problem_analysis}

求解器设计方案：
{solver_design}

要求：
1. 使用gurobipy库
2. 生成完整可运行的代码
3. 代码必须包含数据定义、模型构建、求解和结果输出
4. 输出最优目标值
5. 代码放在```python```代码块中

请生成代码："""
        
        messages = [
            {"role": "system", "content": "你是一位Python和Gurobi编程专家，生成完整可运行的优化代码。"},
            {"role": "user", "content": prompt}
        ]
        
        return query_llm(messages)
    
    def solve_with_retry(self, problem_text, question_id=None, ground_truth=None):
        """使用重试机制解决问题"""
        print(f"\n=== 处理问题 {question_id} ===")
        
        # 阶段1
        print("阶段1: 问题分析与建模...")
        problem_analysis = self.stage1_problem_analysis(problem_text)
        save_generated_code(problem_analysis, f"stage1_q{question_id}")
        
        # 阶段2
        print("阶段2: 求解器设计...")
        solver_design = self.stage2_solver_design(problem_analysis)
        save_generated_code(solver_design, f"stage2_q{question_id}")
        
        # 尝试多次生成和执行代码
        for attempt in range(self.max_retries):
            print(f"\n尝试 {attempt + 1}/{self.max_retries}")
            
            # 阶段3
            print("阶段3: 代码生成...")
            code_content = self.stage3_code_generation(
                problem_text, problem_analysis, solver_design, question_id
            )
            save_generated_code(code_content, f"stage3_q{question_id}_attempt{attempt+1}")
            
            # 执行代码
            print("执行代码...")
            success, result = extract_and_execute_python_code(code_content)
            
            if success:
                print(f"执行成功，结果: {result}")
                
                # 评估结果
                pass_flag, correct_flag = eval_model_result(
                    success, result, ground_truth
                )
                
                if pass_flag:
                    print(f"结果评估: pass=True, correct={correct_flag}")
                    return pass_flag, correct_flag, result
                else:
                    print(f"结果评估失败: pass=False")
            else:
                print(f"执行失败: {result[:200]}...")
                
                # 如果还有重试机会，基于错误重新生成
                if attempt < self.max_retries - 1:
                    print("基于错误信息重新生成代码...")
                    
                    # 修改提示以包含错误信息
                    fix_prompt = f"""之前的代码执行失败，错误信息：
{result[:500]}

请修复错误并重新生成代码。

原始问题：
{problem_text}

请生成正确的Python代码："""
                    
                    messages = [
                        {"role": "system", "content": "修复代码错误并重新生成正确的Gurobi代码。"},
                        {"role": "user", "content": fix_prompt}
                    ]
                    
                    code_content = query_llm(messages)
                else:
                    print("已达到最大重试次数")
        
        return False, False, "所有尝试均失败"
    
    def run_eval(self, dataset, use_agent=True):
        """评估数据集"""
        if not use_agent:
            print("错误: use_agent必须为True")
            return 0, 0, []
        
        pass_count = 0
        correct_count = 0
        error_datas = []
        
        print(f"\n开始评估数据集，共{len(dataset)}个问题")
        
        for qid, data in dataset.items():
            try:
                print(f"\n{'='*50}")
                print(f"处理问题 {qid}: {data.get('difficulty', 'Unknown')}难度")
                
                problem_text = data.get('question', '')
                ground_truth = data.get('answer', '')
                
                if not problem_text:
                    print(f"问题 {qid} 的问题文本为空，跳过")
                    error_datas.append(f"q{qid}_empty")
                    continue
                
                # 使用智能体求解
                pass_flag, correct_flag, result = self.solve_with_retry(
                    problem_text, question_id=qid, ground_truth=ground_truth
                )
                
                if pass_flag:
                    pass_count += 1
                if correct_flag:
                    correct_count += 1
                
                if not pass_flag:
                    error_datas.append(f"q{qid}_run_fail")
                elif not correct_flag:
                    error_datas.append(f"q{qid}_wrong_result")
                
            except Exception as e:
                print(f"处理问题 {qid} 时出错: {e}")
                error_datas.append(f"q{qid}_error_{str(e)[:50]}")
                continue
        
        return pass_count, correct_count, error_datas

def or_llm_agent(problem_text, question_id=None):
    """LLM智能体接口"""
    agent = OroptAgent()
    pass_flag, correct_flag, result = agent.solve_with_retry(problem_text, question_id)
    return pass_flag, result

def main():
    """主函数"""
    print("="*60)
    print("运筹优化智能体")
    print("="*60)
    
    # 检查Gurobi
    if not GUROBI_AVAILABLE:
        print("警告: gurobipy 未安装")
        print("请使用以下命令安装: pip install gurobipy")
        print("(需要Gurobi许可证)")
    
    # 加载数据集
    dataset = load_dataset()
    
    # 创建智能体
    agent = OroptAgent(max_retries=3)
    
    # 运行评估
    pass_count, correct_count, error_datas = agent.run_eval(dataset, use_agent=True)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")
    print("="*60)

if __name__ == "__main__":
    main()
