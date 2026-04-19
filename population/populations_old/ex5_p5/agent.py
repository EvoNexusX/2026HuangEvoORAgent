import argparse
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


DEFAULT_MODEL_NAME = ""


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


def or_llm_agent(user_question, model_name=DEFAULT_MODEL_NAME, max_attempts=3):
    """
    Robust LLM agent for operations research optimization problems.
    
    This agent orchestrates mathematical modeling and Gurobi code generation
    with intelligent retry mechanisms. It features:
    - Enhanced model validation with structured format checking
    - Context-aware retry prompts for different failure modes
    - Dedicated infeasibility analysis with constraint debugging
    - Execution history tracking to prevent infinite loops
    
    Args:
        user_question (str): The optimization problem description
        model_name (str): LLM model to use (default from DEFAULT_MODEL_NAME)
        max_attempts (int): Maximum number of generation attempts (default: 3)
        
    Returns:
        tuple: (status, result) where status is one of:
            - "solved": Successful optimization with numeric objective
            - "infeasible": Model proven infeasible
            - "error": Failure in modeling or execution
            Result is corresponding float, "INFEASIBLE", or error message.
    """
    # Stage 1: Mathematical model generation with comprehensive validation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in operations research optimization. "
                "Given an optimization problem, construct a precise mathematical model "
                "suitable for linear/integer programming implementation.\n\n"
                "Requirements:\n"
                "1. Define decision variables with clear notation (e.g., x_i, y_j)\n"
                "2. State objective function as minimize or maximize\n"
                "3. List all constraints with proper inequality/equality operators\n"
                "4. Use standard mathematical notation\n"
                "5. Output ONLY the mathematical model, no explanations or code"
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model】:\n", math_model)
        
        # Comprehensive model validation
        model_text = math_model.strip().lower()
        model_valid = (
            math_model and 
            len(math_model.strip()) >= 50 and
            any(keyword in model_text for keyword in ["min", "max"]) and
            any(marker in model_text for marker in ["subject to", "s.t.", "constraint", "st."]) and
            any(symbol in model_text for symbol in ["x_", "y_", "z_", "=", "<=", ">="])
        )
        
        if not model_valid:
            print("Warning: Mathematical model may be incomplete or malformed")
            # Add validation feedback to messages for potential revision
            messages.append({"role": "assistant", "content": math_model})
            messages.append({
                "role": "user",
                "content": (
                    "The model appears incomplete. Please ensure it includes:\n"
                    "1. Clearly defined decision variables\n"
                    "2. Objective function with minimize/maximize\n"
                    "3. Complete constraint set\n"
                    "4. Proper mathematical notation\n"
                    "Provide the corrected mathematical model."
                )
            })
            math_model = query_llm(messages, model_name)
            print("【Revised Model】:\n", math_model)
        
        messages.append({"role": "assistant", "content": math_model})
    except Exception as e:
        print(f"!![Model generation error]!!: {e}")
        return "error", f"Failed to generate mathematical model: {e}"
    
    # Stage 2: Code generation with intelligent retry management
    code_prompt = {
        "role": "user",
        "content": (
            "Generate executable Gurobi Python code implementing the above model.\n\n"
            "Code requirements:\n"
            "1. Complete import statements (gurobipy, numpy/pandas if needed)\n"
            "2. Create model with descriptive name\n"
            "3. Define variables with correct types (continuous, integer, binary)\n"
            "4. Set objective with proper sense (GRB.MINIMIZE/MAXIMIZE)\n"
            "5. Add all constraints exactly as specified\n"
            "6. Include optimization and status checking\n"
            "7. Extract and return objective value if optimal\n"
            "8. Handle infeasible/unbounded cases appropriately\n"
            "9. Ensure code is runnable with error handling\n\n"
            "Format: ```python\n{code}\n```"
        )
    }
    messages.append(code_prompt)
    
    execution_history = []  # Track attempts to detect patterns
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Generation Attempt {attempt}/{max_attempts} ===")
        
        # Generate and execute code
        code_gen_success, exec_result, new_messages = generate_or_code_solver(
            messages, model_name, max_attempts=1
        )
        
        # Store attempt for pattern detection
        attempt_record = {
            "code_gen_success": code_gen_success,
            "result_type": type(exec_result).__name__,
            "result_str": str(exec_result)[:200]
        }
        execution_history.append(attempt_record)
        
        # Detect repetitive failures using explicit conditions
        has_repetitive_failures = False
        if len(execution_history) >= 2:
            last_two = execution_history[-2:]
            same_status = (last_two[0]["code_gen_success"] == last_two[1]["code_gen_success"])
            same_result = (last_two[0]["result_str"] == last_two[1]["result_str"])
            has_repetitive_failures = (same_status and same_result)
            
        if has_repetitive_failures:
            print("!![Repetitive failure pattern detected]!!")
            break
        
        if not code_gen_success:
            print(f"!![Code generation failed]!!: {exec_result}")
            
            if attempt < max_attempts:
                # Error-specific retry prompts
                error_lower = str(exec_result).lower()
                if any(err in error_lower for err in ["syntax", "indent", "pars"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Syntax error: {exec_result}\n"
                            "Fix Python syntax issues:\n"
                            "1. Check indentation and parentheses\n"
                            "2. Verify correct use of operators\n"
                            "3. Ensure proper line endings\n"
                            "Provide corrected code in ```python\n{code}\n```"
                        )
                    }
                elif any(err in error_lower for err in ["import", "module", "no module"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Import error: {exec_result}\n"
                            "Ensure proper imports:\n"
                            "1. Add 'import gurobipy as gp'\n"
                            "2. Import numpy/pandas if used\n"
                            "3. Check module availability\n"
                            "Provide corrected code in ```python\n{code}\n```"
                        )
                    }
                elif any(err in error_lower for err in ["attribute", "method", "has no"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Attribute error: {exec_result}\n"
                            "Check Gurobi method usage:\n"
                            "1. Verify correct object.attribute references\n"
                            "2. Ensure variable/constraint naming consistency\n"
                            "3. Check method spelling and arguments\n"
                            "Provide corrected code in ```python\n{code}\n```"
                        )
                    }
                else:
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Code failed with: {exec_result}\n"
                            "Debug the following:\n"
                            "1. Variable definition and usage\n"
                            "2. Constraint formulation\n"
                            "3. Model optimization flow\n"
                            "Provide corrected code in ```python\n{code}\n```"
                        )
                    }
                
                messages = new_messages
                messages.append(retry_prompt)
                continue
            else:
                return "error", f"Code generation failed after {max_attempts} attempts: {exec_result}"
        
        # Analyze execution outcome
        if isinstance(exec_result, (int, float)):
            print(f"✓ Optimization successful: {exec_result}")
            return "solved", exec_result
        
        elif exec_result in ["INFEASIBLE", "INF_OR_UNBD"] or "infeasible" in str(exec_result).lower():
            print(f"!![Infeasible model]!!: {exec_result}")
            
            if attempt < max_attempts:
                # Detailed infeasibility analysis
                analysis_prompt = {
                    "role": "user",
                    "content": (
                        "Model is infeasible. Perform constraint analysis:\n\n"
                        "1. Identify constraint conflicts (e.g., x≤5 and x≥10)\n"
                        "2. Check variable bounds and integer requirements\n"
                        "3. Verify constraint directions (≤, ≥, =)\n"
                        "4. Look for contradictory requirements across constraints\n\n"
                        "If the mathematical model is correct, generate code that:\n"
                        "a) Computes IIS (Irreducible Inconsistent Subsystem) using model.computeIIS()\n"
                        "b) Identifies conflicting constraints\n"
                        "c) Suggests relaxations\n"
                        "Otherwise, revise the mathematical model and regenerate code.\n\n"
                        "Output format:\n"
                        "Analysis: [brief analysis]\n"
                        "```python\n{code}\n```"
                    )
                }
                messages = new_messages
                messages.append(analysis_prompt)
                continue
            else:
                return "infeasible", "INFEASIBLE"
        
        else:
            # Handle ambiguous or unexpected results
            print(f"!![Unexpected result]!!: {exec_result}")
            
            if attempt < max_attempts:
                error_prompt = {
                    "role": "user",
                    "content": (
                        f"Unexpected execution result: {exec_result}\n\n"
                        "Review the entire modeling pipeline:\n"
                        "1. Verify problem interpretation matches user intent\n"
                        "2. Check mathematical model correctness\n"
                        "3. Ensure code implements model exactly\n"
                        "4. Validate result extraction logic\n\n"
                        "Provide revised mathematical model if needed, then corrected code.\n"
                        "Format: [Model revisions or 'NO_CHANGE']\n```python\n{code}\n```"
                    )
                }
                messages = new_messages
                messages.append(error_prompt)
                continue
            else:
                return "error", f"Execution failed: {exec_result}"
    
    # Final fallback after all attempts
    print(f"!!! Maximum attempts ({max_attempts}) exhausted !!!")
    
    # Analyze history for final status
    if execution_history:
        last = execution_history[-1]
        if not last["code_gen_success"]:
            return "error", f"Code generation failed: {last['result_str']}"
        elif "infeasible" in last["result_str"].lower():
            return "infeasible", "INFEASIBLE"
    
    return "error", "Optimization process failed without clear resolution"



def gpt_code_agent_simple(user_question, model_name=DEFAULT_MODEL_NAME):
    messages = [
        {
            "role": "system",
            "content": (
                "你是运筹优化领域的专家。请根据用户提供的优化问题构建数学模型，并编写完整可靠的Python Gurobi代码求解该运筹优化问题。"
                "代码应包含必要的模型构建、变量定义、约束添加、目标函数设置，以及求解与结果输出。"
                "请按 ```python\n{code}\n``` 格式输出，不要附带代码解释。注意{code}部分必须是完整可执行的Python代码，能够直接运行并得到结果。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    save_generated_code(gurobi_code, prefix="simple")

    is_solve_success, result = extract_and_execute_python_code(gurobi_code)
    print(f"Stage result: {is_solve_success}, {result}")
    return is_solve_success, result


def run_eval(use_agent=False, model_name=DEFAULT_MODEL_NAME, data_path="data/datasets/IndustryOR.json", start_i=0):
    dataset = load_dataset(data_path)

    pass_count = 0
    correct_count = 0
    error_datas = []

    for i, d in dataset.items():
        try:
            if int(i) < start_i:
                continue
        except (TypeError, ValueError):
            # If the key is non-numeric, keep existing behavior and run it.
            pass

        new_utils.CURRENT_QUESTION_ID = i
        print(f"=============== num {i} ==================")
        user_question, answer = d["question"], d["answer"]
        print(user_question)
        print("-------------")

        #自我调整
        use_agent = False

        if use_agent:
            print("Using Agent mode (two-stage: modeling + code generation)")
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        else:
            print("Using Simple mode (direct code generation)")
            is_solve_success, llm_result = gpt_code_agent_simple(user_question, model_name)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run OR LLM evaluation with simplified agent script")
    parser.add_argument("--agent", default="True", help="Use agent mode (two-stage: modeling + code generation)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name for LLM")
    parser.add_argument("--data_path", type=str, default="data/datasets/IndustryOR.json", help="Dataset path")
    parser.add_argument("--start_i", type=int, default=0, help="Start evaluating from this question id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(use_agent=args.agent, model_name=args.model, data_path=args.data_path, start_i=args.start_i)

