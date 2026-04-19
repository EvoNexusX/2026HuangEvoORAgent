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
    
    This agent combines strengths from previous versions:
    1. Mathematical modeling with structured output validation
    2. Iterative refinement with targeted debugging prompts
    3. Comprehensive solution validation and error handling
    4. Maintains original return format compatibility
    
    Args:
        user_question (str): Natural language description of optimization problem
        model_name (str): LLM model to use (defaults to DEFAULT_MODEL_NAME)
        max_attempts (int): Maximum number of debugging attempts (default: 3)
    
    Returns:
        tuple: (is_solve_success, result, math_model, messages)
            is_solve_success (bool): Whether solving was successful
            result: Solution output (float), error info, or "INFEASIBLE"
            math_model (str): Generated mathematical model
            messages (list): Complete conversation history
    """
    
    # Helper function for solution validation
    def validate_solution(solution):
        """Validate solution format and content."""
        if solution is None:
            return False, "No solution generated"
        
        # Check for special case strings
        if isinstance(solution, str):
            sol_lower = solution.lower()
            # Define error indicators as a set for clarity and performance
            ERROR_INDICATORS = {"infeasible", "error", "exception", "traceback"}
            if any(indicator in sol_lower for indicator in ERROR_INDICATORS):
                return False, f"Solution contains error indicator: {solution}"
            
            # Check if it's a number string
            try:
                float_val = float(solution)
                return True, float_val
            except ValueError:
                return False, f"Non-numeric string result: {solution}"
        
        # Check for numeric types
        if isinstance(solution, (int, float)):
            return True, solution
        
        return False, f"Unrecognized result type: {type(solution)}"
    
    # Phase 1: Generate mathematical model with enhanced prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in operations research optimization. "
                "Translate the user's problem into precise mathematical programming formulations.\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Define ALL sets, parameters, and decision variables with clear notation\n"
                "2. Formulate objective function (minimize/maximize) with exact coefficients\n"
                "3. List ALL constraints with proper indices and bounds\n"
                "4. Specify variable types (continuous, integer, binary) and domains\n"
                "5. Include any special conditions (if-then, logical constraints)\n"
                "6. Use standard OR notation (e.g., ∑ for summation, ∀ for all)\n\n"
                "Output format:\n"
                "Sets: ...\n"
                "Parameters: ...\n"
                "Variables: ...\n"
                "Objective: ...\n"
                "Constraints: ...\n"
                "Additional conditions: ..."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model】\n", "-" * 50, "\n", math_model, "\n", "-" * 50)
        
        # Validate model has reasonable content
        if not math_model or len(math_model.strip()) < 50:
            print("Warning: Mathematical model may be incomplete")
        
        messages.append({"role": "assistant", "content": math_model})
    except Exception as e:
        print(f"!![Mathematical modeling error]!!: {e}")
        return False, f"Failed to generate mathematical model: {e}", "", messages
    
    # Phase 2: Initial code generation prompt
    code_generation_messages = messages + [
        {
            "role": "user",
            "content": (
                "Transform the mathematical model into robust Gurobi Python code.\n\n"
                "ESSENTIAL COMPONENTS TO INCLUDE:\n"
                "1. Model initialization with error handling\n"
                "2. Complete variable definitions with proper types and bounds\n"
                "3. Objective function setup with correct sense\n"
                "4. ALL constraints implemented efficiently\n"
                "5. Model optimization with time limits and tolerances\n"
                "6. Solution extraction with validation checks\n"
                "7. Error handling for infeasibility/unbounded cases\n"
                "8. Clean output formatting of optimal values\n\n"
                "CODING STANDARDS:\n"
                "- Use GRB for Gurobi constants\n"
                "- Add comments for complex constraints\n"
                "- Include try-except blocks for solver errors\n"
                "- Set appropriate numerical tolerances\n"
                "- Validate solution status before extraction\n\n"
                "Output ONLY executable code in ```python\n{code}\n``` format."
            ),
        }
    ]
    
    # Phase 3: Iterative refinement with targeted debugging
    total_attempts = 0
    execution_history = []
    
    while total_attempts < max_attempts:
        total_attempts += 1
        print(f"\n=== Attempt {total_attempts}/{max_attempts} ===")
        
        # Generate and execute code
        is_solve_success, result, new_messages = generate_or_code_solver(
            code_generation_messages, model_name, max_attempts=1
        )
        
        # Update message history
        if new_messages and len(new_messages) > len(code_generation_messages):
            code_generation_messages = new_messages
        
        execution_history.append((is_solve_success, result))
        
        # Validate and process result
        if is_solve_success:
            is_valid, valid_result = validate_solution(result)
            
            if is_valid:
                # Successful numeric solution
                print(f"✓ Solution found: {valid_result}")
                return True, valid_result, math_model, code_generation_messages
            
            elif "infeasible" in str(result).lower():
                # Infeasible model - targeted debugging
                print("!![Infeasible model detected]!!")
                debug_prompt = {
                    "role": "user",
                    "content": (
                        "The model is INFEASIBLE. Please analyze and fix:\n"
                        "1. Check for conflicting or overly restrictive constraints\n"
                        "2. Verify variable bounds and types are correctly defined\n"
                        "3. Ensure constraint directions (≤, ≥, =) match the mathematical model\n"
                        "4. Review the mathematical model for logical errors\n\n"
                        "Provide revised Gurobi code addressing these issues.\n"
                        "Output code in ```python\n{code}\n``` format."
                    )
                }
                code_generation_messages.append(debug_prompt)
                continue
            else:
                # Solution validation failed
                print(f"!![Solution validation failed]!!: {valid_result}")
                debug_prompt = {
                    "role": "user",
                    "content": (
                        f"Solution validation failed: {valid_result}\n"
                        "Please diagnose and fix potential issues:\n"
                        "1. Check solution extraction code\n"
                        "2. Verify the model produces expected output format\n"
                        "3. Ensure proper handling of solver status codes\n\n"
                        "Provide corrected code in ```python\n{code}\n``` format."
                    )
                }
                code_generation_messages.append(debug_prompt)
                continue
        else:
            # Code generation/execution failed
            print(f"!![Code execution error]!!: {result}")
            debug_prompt = {
                "role": "user",
                "content": (
                    f"Code execution failed with error: {result}\n"
                    "Please fix the following:\n"
                    "1. Check for syntax errors and undefined variables\n"
                    "2. Verify correct Gurobi API usage\n"
                    "3. Ensure mathematical expressions match the model\n"
                    "4. Test constraint indices and ranges\n\n"
                    "Provide corrected code in ```python\n{code}\n``` format."
                )
            }
            code_generation_messages.append(debug_prompt)
            continue
    
    # Phase 4: Final comprehensive debugging attempt
    print("\n!![Final comprehensive debugging attempt]!!")
    
    # Analyze failure patterns from history
    failure_types = [status for status, _ in execution_history[-2:] if not status]
    if len(failure_types) >= 2:
        final_prompt = {
            "role": "user",
            "content": (
                "Multiple attempts failed. Please conduct comprehensive review:\n"
                "1. Re-examine the mathematical model for fundamental errors\n"
                "2. Check ALL variable definitions and constraints\n"
                "3. Verify objective function formulation\n"
                "4. Ensure code implements the exact mathematical model\n"
                "5. Add detailed error handling and debugging output\n\n"
                "Output fully corrected code with comprehensive comments in ```python\n{code}\n``` format."
            )
        }
    else:
        final_prompt = {
            "role": "user",
            "content": (
                "Final debugging attempt. Please:\n"
                "1. Review the entire model and code for any remaining issues\n"
                "2. Add explicit error handling for edge cases\n"
                "3. Ensure solution extraction is robust\n"
                "4. Verify numerical tolerances are appropriate\n\n"
                "Output corrected code in ```python\n{code}\n``` format."
            )
        }
    
    code_generation_messages.append(final_prompt)
    
    final_success, final_result, final_messages = generate_or_code_solver(
        code_generation_messages, model_name, max_attempts=2
    )
    
    # Update messages if new ones were generated
    if final_messages and len(final_messages) > len(code_generation_messages):
        code_generation_messages = final_messages
    
    # Process final result
    if final_success:
        is_valid, valid_result = validate_solution(final_result)
        if is_valid:
            return True, valid_result, math_model, code_generation_messages
        elif "infeasible" in str(final_result).lower():
            return False, "INFEASIBLE", math_model, code_generation_messages
    
    # Maximum attempts exhausted
    print(f"\n❌ Maximum attempts ({max_attempts}) exhausted without valid solution")
    
    # Return the most recent result with failure status
    if execution_history:
        last_success, last_result = execution_history[-1]
        return last_success, last_result, math_model, code_generation_messages
    
    return False, "Optimization failed after maximum attempts", math_model, code_generation_messages



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

