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
    Enhanced LLM agent for operations research optimization problems.
    
    This agent guides the LLM through mathematical modeling and Gurobi code generation
    with iterative refinement based on execution feedback. It combines robustness from
    Parent A with clarity from Parent B, improving retry logic and infeasible handling.
    
    Args:
        user_question (str): The optimization problem description
        model_name (str): LLM model to use
        max_attempts (int): Maximum number of code generation attempts
        
    Returns:
        tuple: (success_status, final_result) where:
            - success_status: "solved", "infeasible", or "error"
            - final_result: objective value (float), "INFEASIBLE", or error message
    """
    # Stage 1: Generate and validate mathematical model
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in operations research optimization. "
                "Given an optimization problem, construct an accurate mathematical model "
                "using linear programming formulations. "
                "Provide the complete mathematical expressions including decision variables, "
                "objective function, and all constraints. "
                "Focus on correctness and completeness for Gurobi code generation. "
                "Output only the mathematical model without explanation."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model】:\n", math_model)
        
        # Enhanced model validation with clear condition naming
        has_min_or_max = any(keyword in math_model.lower() for keyword in ["min", "max"])
        has_constraint_markers = any(marker in math_model.lower() 
                                     for marker in ["subject to", "constraint", "s.t."])
        model_valid = (
            math_model and 
            len(math_model.strip()) >= 50 and
            has_min_or_max and
            has_constraint_markers
        )
        if not model_valid:
            print("Warning: Mathematical model may be incomplete or malformed")
        messages.append({"role": "assistant", "content": math_model})
    except Exception as e:
        print(f"!![Model generation error]!!: {e}")
        return "error", f"Failed to generate mathematical model: {e}"
    
    # Stage 2: Code generation with enhanced retry logic
    code_prompt = {
        "role": "user",
        "content": (
            "Based on the mathematical model above, generate complete Gurobi Python code.\n"
            "Requirements:\n"
            "1. Import necessary libraries (e.g., gurobipy)\n"
            "2. Define model name appropriately\n"
            "3. Create all decision variables with correct types and bounds\n"
            "4. Set objective function with proper sense (minimize/maximize)\n"
            "5. Add all constraints from the model\n"
            "6. Include model optimization and status checking\n"
            "7. Extract and print objective value and key variables\n"
            "8. Handle infeasible/unbounded cases with error messages\n"
            "Format: ```python\n{code}\n```"
        )
    }
    messages.append(code_prompt)
    
    execution_results = []  # Store attempt history for fallback
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")
        
        # Generate and execute code with controlled inner retries
        code_gen_success, exec_result, new_messages = generate_or_code_solver(
            messages, model_name, max_attempts=1  # Single attempt per outer iteration
        )
        
        if not code_gen_success:
            print(f"!![Code generation failed]!!: {exec_result}")
            execution_results.append(("generation_failed", exec_result))
            
            if attempt < max_attempts:
                # Context-aware retry prompt based on error type
                if "syntax" in str(exec_result).lower():
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Syntax error detected: {exec_result}\n"
                            "Please fix the syntax issues and regenerate the code.\n"
                            "Focus on:\n"
                            "1. Correct Python syntax and indentation\n"
                            "2. Proper Gurobi method calls\n"
                            "3. Balanced parentheses and brackets\n"
                            "Format: ```python\n{code}\n```"
                        )
                    }
                elif "import" in str(exec_result).lower() or "module" in str(exec_result).lower():
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Import/module error: {exec_result}\n"
                            "Please ensure all required imports are present and correct.\n"
                            "Common imports: gurobipy, math, numpy, pandas (if needed)\n"
                            "Format: ```python\n{code}\n```"
                        )
                    }
                else:
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"The previous code failed with error: {exec_result}\n"
                            "Please fix the following issues:\n"
                            "1. Check for syntax errors in the code\n"
                            "2. Verify all variables are defined before use\n"
                            "3. Ensure correct Gurobi method calls and imports\n"
                            "4. Validate constraint expressions against the mathematical model\n"
                            "Generate corrected code in ```python\n{code}\n``` format."
                        )
                    }
                messages = new_messages
                messages.append(retry_prompt)
                continue
            else:
                return "error", f"Code generation failed after {max_attempts} attempts: {exec_result}"
        
        # Analyze execution results with clear type checking
        if isinstance(exec_result, (int, float)):
            print(f"✓ Solution found: {exec_result}")
            return "solved", exec_result
        elif exec_result in ["INFEASIBLE"] or "infeasible" in str(exec_result).lower():
            print(f"!![Infeasible model detected]!!: {exec_result}")
            
            if attempt < max_attempts:
                # Enhanced infeasibility analysis with constraint prioritization
                analysis_prompt = {
                    "role": "user",
                    "content": (
                        "The model is infeasible. Perform detailed analysis:\n"
                        "1. Identify potentially conflicting constraints (e.g., x <= 5 and x >= 10)\n"
                        "2. Check variable bounds and types (continuous, integer, binary)\n"
                        "3. Verify constraint directions (<=, >=, =) and right-hand side values\n"
                        "4. Review the mathematical model for logical errors\n"
                        "Provide a revised mathematical model if necessary, then generate corrected Gurobi code.\n"
                        "Format: First output the revised model (or 'NO_CHANGE'), then code in ```python\n{code}\n```"
                    )
                }
                messages = new_messages
                messages.append(analysis_prompt)
                continue
            else:
                return "infeasible", "INFEASIBLE"
        else:
            # Handle unexpected results (e.g., runtime errors, ambiguous outputs)
            print(f"!![Unexpected execution result]!!: {exec_result}")
            execution_results.append(("error", exec_result))
            
            if attempt < max_attempts:
                # General error recovery with model validation
                error_prompt = {
                    "role": "user",
                    "content": (
                        f"Execution resulted in: {exec_result}\n"
                        "Please review the entire modeling process:\n"
                        "1. Verify the mathematical model accurately represents the problem\n"
                        "2. Ensure the code correctly implements all model components\n"
                        "3. Check for common pitfalls (e.g., off-by-one errors, unit conversions)\n"
                        "Revise the mathematical model if needed, then regenerate the Gurobi code.\n"
                        "Format: First output any model revisions (or 'NO_CHANGE'), then code in ```python\n{code}\n```"
                    )
                }
                messages = new_messages
                messages.append(error_prompt)
                continue
            else:
                return "error", f"Execution failed with unexpected result: {exec_result}"
    
    # Fallback: should only reach here if loop completes without returns
    print(f"\n!!! Maximum attempts ({max_attempts}) reached without resolution !!!")
    if execution_results:
        last_status, last_result = execution_results[-1]
        if "generation_failed" in last_status:
            return "error", f"Code generation failed: {last_result}"
        else:
            return "error", f"Execution failed: {last_result}"
    return "error", "Unknown failure in optimization process"



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

