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
    Robust LLM agent for Operations Research optimization problems.
    
    This agent integrates strengths from both parent variants:
    1. Maintains full return signature (success, result, math_model, messages) for debugging
    2. Enhanced retry logic with differentiated error handling
    3. Clear status tracking with explicit state transitions
    4. Improved infeasibility analysis with constraint validation
    
    Args:
        user_question (str): Natural language optimization problem description
        model_name (str): LLM model name (defaults to DEFAULT_MODEL_NAME)
        max_attempts (int): Maximum generation/solving attempts (default: 3)
    
    Returns:
        tuple: (is_solve_success, result, math_model, messages)
            is_solve_success (bool): Whether solving was successful
            result: Objective value (float), "INFEASIBLE", or error message
            math_model (str): Generated mathematical model
            messages (list): Complete conversation history
    """
    import traceback
    
    # Initialize conversation with enhanced system prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert operations research optimization assistant. "
                "Given an optimization problem, construct an accurate mathematical model "
                "using appropriate formulations (linear, integer, or mixed-integer programming).\n\n"
                "Requirements:\n"
                "1. Clearly define all decision variables with indices and types\n"
                "2. Specify objective function (minimize/maximize) with complete expression\n"
                "3. List all constraints with proper mathematical notation\n"
                "4. Include any necessary assumptions or simplifications\n"
                "5. Output only the mathematical model without explanatory text"
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    # Stage 1: Generate and validate mathematical model
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model Generated】")
        
        # Enhanced validation with multiple checks
        model_valid = True
        validation_issues = []
        
        # Check for completeness
        if not math_model or len(math_model.strip()) < 60:
            validation_issues.append("Model appears too short or empty")
            model_valid = False
        
        # Check for essential components
        model_lower = math_model.lower()
        required_components = [
            ("objective", any(kw in model_lower for kw in ["min", "max", "objective"])),
            ("variables", any(kw in model_lower for kw in ["x[", "y[", "var", "variable"])),
            ("constraints", any(kw in model_lower for kw in ["subject to", "s.t.", "constraint", "st."]))
        ]
        
        for component, present in required_components:
            if not present:
                validation_issues.append(f"Missing {component} definition")
                model_valid = False
        
        if not model_valid:
            print(f"Model validation warnings: {validation_issues}")
        
        messages.append({"role": "assistant", "content": math_model})
        
    except Exception as e:
        error_msg = f"Model generation failed: {str(e)}"
        print(f"!![ERROR]!! {error_msg}")
        return False, error_msg, "", messages
    
    # Stage 2: Code generation with progressive refinement
    code_prompt = {
        "role": "user",
        "content": (
            "Based on the mathematical model above, generate complete Gurobi Python code.\n\n"
            "Code requirements:\n"
            "1. Import gurobipy and other necessary libraries\n"
            "2. Create model with descriptive name\n"
            "3. Define all variables with correct types (continuous, binary, integer) and bounds\n"
            "4. Set objective function with proper sense\n"
            "5. Add ALL constraints from the mathematical model\n"
            "6. Include model.optimize() and check optimization status\n"
            "7. Extract and return objective value\n"
            "8. Handle infeasible/unbounded cases with clear messages\n"
            "9. Ensure code is executable without syntax errors\n\n"
            "Format: ```python\n<your code>\n```"
        )
    }
    messages.append(code_prompt)
    
    # Main solving loop with differentiated retry strategies
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")
        
        # Generate and execute code
        try:
            code_gen_success, exec_result, new_messages = generate_or_code_solver(
                messages, model_name, max_attempts=1
            )
            
            # Update conversation history
            if new_messages and len(new_messages) > len(messages):
                messages = new_messages
            
            if not code_gen_success:
                # Code generation or execution failed
                print(f"!![Code execution failed]!!: {exec_result}")
                
                if attempt < max_attempts:
                    # Context-aware error recovery
                    error_str = str(exec_result).lower()
                    
                    if any(kw in error_str for kw in ["syntax", "indentation", "parse"]):
                        retry_prompt = {
                            "role": "user",
                            "content": (
                                f"Syntax error detected: {exec_result[:200]}\n"
                                "Please fix syntax issues and regenerate code.\n"
                                "Focus on:\n"
                                "1. Correct Python syntax and indentation\n"
                                "2. Balanced parentheses, brackets, and quotes\n"
                                "3. Proper statement endings\n\n"
                                "Output corrected code in ```python\n``` format."
                            )
                        }
                    elif any(kw in error_str for kw in ["import", "module", "no module"]):
                        retry_prompt = {
                            "role": "user",
                            "content": (
                                f"Import error: {exec_result[:200]}\n"
                                "Ensure all required imports are present and correctly spelled.\n"
                                "Common imports: gurobipy as gp, numpy as np, math\n\n"
                                "Output corrected code in ```python\n``` format."
                            )
                        }
                    else:
                        retry_prompt = {
                            "role": "user",
                            "content": (
                                f"Execution error: {exec_result[:200]}\n"
                                "Please fix the following potential issues:\n"
                                "1. Variables defined before use\n"
                                "2. Correct Gurobi method names and arguments\n"
                                "3. Constraint expressions matching variable definitions\n"
                                "4. Proper model update and optimization calls\n\n"
                                "Output corrected code in ```python\n``` format."
                            )
                        }
                    
                    messages.append(retry_prompt)
                    continue
                else:
                    return False, f"Code generation failed after {max_attempts} attempts: {exec_result}", math_model, messages
            
            # Analyze execution result
            if isinstance(exec_result, (int, float)):
                # Successful solution
                print(f"✓ Solution found: {exec_result}")
                return True, exec_result, math_model, messages
            
            elif exec_result in ["INFEASIBLE", "INF_OR_UNBD"] or "infeasible" in str(exec_result).lower():
                print(f"!![Infeasible model]!!")
                
                if attempt < max_attempts:
                    # Detailed infeasibility analysis
                    analysis_prompt = {
                        "role": "user",
                        "content": (
                            "Model is infeasible. Perform constraint analysis:\n\n"
                            "1. Check for contradictory constraints (e.g., x ≤ 5 and x ≥ 10)\n"
                            "2. Verify variable bounds and types are consistent\n"
                            "3. Ensure constraint coefficients and right-hand sides are correct\n"
                            "4. Review if any constraints from the original problem are missing\n\n"
                            "If the mathematical model needs revision, output:\n"
                            "REVISED_MODEL:\n<new model>\n\n"
                            "Then output corrected Gurobi code in ```python\n``` format.\n"
                            "If model is correct, just output corrected code."
                        )
                    }
                    messages.append(analysis_prompt)
                    continue
                else:
                    # Maintain original behavior: infeasible result with success=True
                    return True, "INFEASIBLE", math_model, messages
            
            else:
                # Unexpected result type
                print(f"!![Unexpected result]!!: {exec_result}")
                
                if attempt < max_attempts:
                    general_prompt = {
                        "role": "user",
                        "content": (
                            f"Unexpected result: {str(exec_result)[:200]}\n"
                            "Please review:\n"
                            "1. Does the mathematical model accurately represent the problem?\n"
                            "2. Are all constraints correctly implemented in code?\n"
                            "3. Is the objective function correctly defined?\n"
                            "4. Are results being extracted properly after optimization?\n\n"
                            "Output any necessary revisions followed by corrected code in ```python\n``` format."
                        )
                    }
                    messages.append(general_prompt)
                    continue
                else:
                    return False, f"Unexpected result after {max_attempts} attempts: {exec_result}", math_model, messages
        
        except Exception as e:
            print(f"!![Unexpected error in main loop]!!: {str(e)}")
            if attempt < max_attempts:
                messages.append({
                    "role": "user",
                    "content": f"Internal error occurred: {str(e)[:150]}\nPlease regenerate the code."
                })
                continue
            else:
                return False, f"Unexpected error after {max_attempts} attempts: {str(e)}", math_model, messages
    
    # Final fallback with comprehensive analysis
    print("!![Maximum attempts reached]!!")
    
    final_prompt = {
        "role": "user",
        "content": (
            "Maximum attempts reached. Perform comprehensive analysis:\n\n"
            "1. Verify mathematical model accurately captures all problem requirements\n"
            "2. Check for common modeling errors (wrong variable types, missing constraints)\n"
            "3. Ensure code correctly implements the mathematical model\n"
            "4. Validate objective function and constraint expressions\n\n"
            "Output final revised model (if needed) and corrected Gurobi code in ```python\n``` format."
        )
    }
    messages.append(final_prompt)
    
    try:
        final_success, final_result, final_messages = generate_or_code_solver(
            messages, model_name, max_attempts=2
        )
        
        if final_messages and len(final_messages) > len(messages):
            messages = final_messages
        
        return final_success, final_result, math_model, messages
    except Exception as e:
        return False, f"Final attempt error: {str(e)}", math_model, messages



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

