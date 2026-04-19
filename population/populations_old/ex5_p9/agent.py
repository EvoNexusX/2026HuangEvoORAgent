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
    Advanced LLM agent for operations research optimization problems.
    
    This agent combines robust mathematical modeling with intelligent iterative refinement,
    featuring enhanced error diagnosis, systematic retry logic, and comprehensive solution
    validation while maintaining compatibility with the original interface.
    
    Args:
        user_question (str): Natural language description of optimization problem
        model_name (str): LLM model to use (defaults to DEFAULT_MODEL_NAME)
        max_attempts (int): Maximum number of refinement attempts (default: 3)
    
    Returns:
        tuple: (status, result) where:
            - status: "solved", "infeasible", or "error"
            - result: objective value (float), "INFEASIBLE", or error message
    """
    
    def validate_solution(solution):
        """Robust validation of solution format with detailed diagnostics."""
        if solution is None:
            return False, "No solution generated"
        
        # Handle string results
        if isinstance(solution, str):
            sol_lower = solution.lower()
            
            # Detect error/infeasibility indicators
            if any(indicator in sol_lower for indicator in 
                   ["infeasible", "inf_or_unbd", "error", "exception", "traceback"]):
                return False, f"Invalid solution state: {solution}"
            
            # Attempt to extract numeric value from string
            try:
                return True, float(solution)
            except ValueError:
                return False, f"Non-numeric string result: {solution}"
        
        # Handle numeric results
        if isinstance(solution, (int, float)):
            # Check for unusual numeric values
            if not (float('-inf') < solution < float('inf')):
                return False, f"Invalid numeric value: {solution}"
            return True, float(solution)
        
        return False, f"Unrecognized result type: {type(solution)}"
    
    # Phase 1: Structured mathematical modeling with enhanced prompts
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert operations research optimization specialist. "
                "Translate the problem into a precise mathematical programming formulation.\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Define ALL decision variables with clear notation and domains\n"
                "2. Formulate objective function with exact coefficients\n"
                "3. List ALL constraints with proper indices and bounds\n"
                "4. Specify variable types (continuous, integer, binary)\n"
                "5. Include any special conditions or logical constraints\n"
                "6. Use standard OR notation for clarity\n\n"
                "Output ONLY the mathematical model without additional explanation."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model】\n" + "=" * 60 + f"\n{math_model}\n" + "=" * 60)
        
        # Validate model has minimum content
        if not math_model or len(math_model.strip()) < 50:
            print("Warning: Generated mathematical model may be insufficient")
        
        messages.append({"role": "assistant", "content": math_model})
    except Exception as e:
        print(f"!![Model generation error]!!: {e}")
        return "error", f"Failed to generate mathematical model: {e}"
    
    # Phase 2: Initial code generation with comprehensive requirements
    messages.append({
        "role": "user",
        "content": (
            "Transform the mathematical model into robust Gurobi Python code.\n\n"
            "ESSENTIAL COMPONENTS:\n"
            "1. Import necessary libraries with error handling\n"
            "2. Define model with appropriate name\n"
            "3. Create all variables with correct types and bounds\n"
            "4. Set objective function with proper optimization sense\n"
            "5. Add ALL constraints efficiently\n"
            "6. Include model optimization with status checking\n"
            "7. Extract and return objective value and key variables\n"
            "8. Handle infeasible/unbounded cases explicitly\n\n"
            "CODING STANDARDS:\n"
            "- Use try-except blocks for solver errors\n"
            "- Validate solution status before extraction\n"
            "- Set appropriate numerical tolerances\n"
            "- Include comments for complex constraints\n\n"
            "Output ONLY executable code in ```python\n{code}\n``` format."
        )
    })
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}\nAttempt {attempt}/{max_attempts}\n{'='*60}")
        
        # Generate and execute code
        code_gen_success, exec_result, new_messages = generate_or_code_solver(
            messages, model_name, max_attempts=1
        )
        
        # Update conversation history
        if new_messages and len(new_messages) > len(messages):
            messages = new_messages
        
        # Handle code generation failure
        if not code_gen_success:
            print(f"!![Code generation failed]!!: {exec_result}")
            
            if attempt < max_attempts:
                retry_prompt = {
                    "role": "user",
                    "content": (
                        f"Previous code failed with error: {exec_result}\n\n"
                        "Please fix these issues:\n"
                        "1. Check syntax errors and undefined variables\n"
                        "2. Verify correct Gurobi API usage and imports\n"
                        "3. Ensure all variables are properly defined before use\n"
                        "4. Validate constraint expressions match mathematical model\n\n"
                        "Provide corrected code in ```python\n{code}\n``` format."
                    )
                }
                messages.append(retry_prompt)
            continue
        
        # Validate execution result
        is_valid, valid_result = validate_solution(exec_result)
        
        if is_valid:
            # Successful solution
            print(f"✓ Solution found: {valid_result}")
            return "solved", valid_result
        
        # Handle infeasibility
        result_str = str(exec_result).lower()
        if "infeasible" in result_str or exec_result in ["INFEASIBLE", "INF_OR_UNBD"]:
            print("!![Infeasible model detected]!!")
            
            if attempt < max_attempts:
                infeasibility_prompt = {
                    "role": "user",
                    "content": (
                        "The model is INFEASIBLE. Please analyze and fix:\n\n"
                        "POTENTIAL CAUSES TO INVESTIGATE:\n"
                        "1. Conflicting or overly restrictive constraints\n"
                        "2. Incorrect variable bounds or types\n"
                        "3. Wrong constraint directions (≤, ≥, =)\n"
                        "4. Mathematical model formulation errors\n\n"
                        "Provide a revised mathematical model if necessary, "
                        "followed by corrected Gurobi code.\n"
                        "Output format: First revised model, then code in ```python\n{code}\n``` format."
                    )
                }
                messages.append(infeasibility_prompt)
            else:
                return "infeasible", "INFEASIBLE"
            continue
        
        # Handle other validation failures
        print(f"!![Solution validation failed]!!: {valid_result}")
        
        if attempt < max_attempts:
            validation_prompt = {
                "role": "user",
                "content": (
                    f"Solution validation failed: {valid_result}\n\n"
                    "Please diagnose and correct:\n"
                    "1. Check solution extraction and formatting\n"
                    "2. Verify solver status handling\n"
                    "3. Ensure objective value is properly captured\n"
                    "4. Review model for numerical issues\n\n"
                    "Provide corrected code in ```python\n{code}\n``` format."
                )
            }
            messages.append(validation_prompt)
            continue
    
    # All attempts exhausted without success
    print(f"\n{'!'*60}\nMaximum attempts ({max_attempts}) exhausted\n{'!'*60}")
    return "error", "Optimization process failed without producing results"



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

