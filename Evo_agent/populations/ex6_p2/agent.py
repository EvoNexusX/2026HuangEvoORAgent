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
    with iterative refinement based on execution feedback. It combines robust validation,
    adaptive retry logic, and detailed infeasibility analysis for improved reliability.
    
    Args:
        user_question (str): The optimization problem description
        model_name (str): LLM model to use
        max_attempts (int): Maximum number of code generation attempts
        
    Returns:
        tuple: (success_status, final_result) where:
            - success_status: "solved", "infeasible", or "error"
            - final_result: objective value (float), "INFEASIBLE", or error message
    """
    # Stage 1: Generate and validate mathematical model with structured feedback
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in operations research optimization. "
                "Given an optimization problem, construct an accurate mathematical model "
                "using appropriate formulations (linear, integer, mixed-integer). "
                "Provide the complete mathematical expressions including:\n"
                "1. Decision variables with domains\n"
                "2. Objective function with sense (minimize/maximize)\n"
                "3. All constraints with proper notation\n"
                "Focus on correctness and completeness for Gurobi code generation. "
                "Output only the mathematical model without explanation."
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    try:
        math_model = query_llm(messages, model_name)
        print("【Mathematical Model】:\n", math_model)
        
        # Enhanced validation with diagnostic feedback
        model_text = math_model.strip()
        validation_checks = {
            "non_empty": bool(model_text),
            "minimum_length": len(model_text) >= 50,
            "has_objective": any(keyword in model_text.lower() 
                                for keyword in ["min", "max", "minimize", "maximize"]),
            "has_constraints": any(marker in model_text.lower() 
                                  for marker in ["subject to", "constraint", "s.t.", "such that"]),
            "has_variables": any(var_indicator in model_text 
                                for var_indicator in ["x", "y", "z", "=", "<=", ">="])
        }
        
        # Only warn for validation failures, don't return early
        failed = [k for k, v in validation_checks.items() if not v]
        if failed:
            print(f"Warning: Model validation failed for: {failed}")
        
        messages.append({"role": "assistant", "content": math_model})
    except Exception as e:
        print(f"!![Model generation error]!!: {e}")
        return "error", f"Failed to generate mathematical model: {e}"
    
    # Stage 2: Initial code generation prompt with explicit error handling
    code_prompt = {
        "role": "user",
        "content": (
            "Based on the mathematical model above, generate complete Gurobi Python code.\n"
            "Critical Requirements:\n"
            "1. Import gurobipy as 'gp' (standard convention)\n"
            "2. Create model with gp.Model()\n"
            "3. Define all variables with appropriate types (continuous, integer, binary) and bounds\n"
            "4. Set objective with correct sense using model.setObjective()\n"
            "5. Add constraints using model.addConstr() or model.addConstrs()\n"
            "6. Include model.optimize() and comprehensive status checking\n"
            "7. Extract objective value ONLY if status == GRB.OPTIMAL\n"
            "8. Handle infeasible/unbounded cases with clear error messages\n"
            "9. Include try-except blocks for runtime errors\n"
            "Format: ```python\n{code}\n```"
        )
    }
    messages.append(code_prompt)
    
    execution_history = []  # Track all attempts for fallback analysis
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")
        
        # Generate and execute code with single inner attempt
        code_gen_success, exec_result, new_messages = generate_or_code_solver(
            messages, model_name, max_attempts=1
        )
        execution_history.append((code_gen_success, exec_result))
        
        if not code_gen_success:
            print(f"!![Code generation failed]!!: {exec_result}")
            
            if attempt < max_attempts:
                # Adaptive error categorization for targeted feedback
                error_str = str(exec_result).lower()
                if any(e in error_str for e in ["syntax", "indentation", "parsing"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Syntax error: {exec_result}\n"
                            "Fix these specific issues:\n"
                            "1. Check Python syntax and indentation\n"
                            "2. Ensure balanced parentheses, brackets, and quotes\n"
                            "3. Verify proper line continuation if needed\n"
                            "Generate corrected code in ```python\n{code}\n``` format."
                        )
                    }
                elif any(e in error_str for e in ["import", "module", "no module", "gurobi"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Import error: {exec_result}\n"
                            "Required imports: 'import gurobipy as gp'\n"
                            "Optional but recommended: 'from gurobipy import GRB'\n"
                            "Ensure no typos in module names.\n"
                            "Format: ```python\n{code}\n```"
                        )
                    }
                elif any(e in error_str for e in ["attribute", "method", "function", "has no"]):
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Attribute/method error: {exec_result}\n"
                            "Common issues:\n"
                            "1. Use 'gp.Model()' not 'gurobipy.Model()'\n"
                            "2. Variable methods: model.addVar(), model.addVars()\n"
                            "3. Constraint methods: model.addConstr(), model.addConstrs()\n"
                            "4. Check Gurobi method names and signatures\n"
                            "Format: ```python\n{code}\n```"
                        )
                    }
                else:
                    retry_prompt = {
                        "role": "user",
                        "content": (
                            f"Code execution failed: {exec_result}\n"
                            "Systematic debugging steps:\n"
                            "1. Verify variable definitions match mathematical model\n"
                            "2. Check constraint indices and summation bounds\n"
                            "3. Ensure objective function uses correct variables\n"
                            "4. Add print statements to debug variable values if needed\n"
                            "Format: ```python\n{code}\n```"
                        )
                    }
                
                messages = new_messages
                messages.append(retry_prompt)
                continue
            else:
                return "error", f"Code generation failed after {max_attempts} attempts: {exec_result}"
        
        # Analyze execution results with precise classification
        if isinstance(exec_result, (int, float)):
            print(f"✓ Optimal solution found: {exec_result}")
            return "solved", exec_result
        elif exec_result in ["INFEASIBLE", "INF_OR_UNBD"] or "infeasible" in str(exec_result).lower():
            print(f"!![Infeasible model]!!: {exec_result}")
            
            if attempt < max_attempts:
                # Diagnostic infeasibility analysis with conflict identification
                analysis_prompt = {
                    "role": "user",
                    "content": (
                        "Model is infeasible. Perform conflict analysis:\n"
                        "1. Identify minimal conflicting constraint set (e.g., x ≤ 5 AND x ≥ 10)\n"
                        "2. Check variable type mismatches (continuous vs integer)\n"
                        "3. Verify constraint coefficients and right-hand sides\n"
                        "4. Consider relaxing bounds or constraints for feasibility\n"
                        "Output in this EXACT format:\n"
                        "ANALYSIS: [brief diagnosis]\n"
                        "REVISED_MODEL: [updated mathematical model or 'NO_CHANGE']\n"
                        "CODE: ```python\n{code}\n```"
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
                # Comprehensive model-code reconciliation
                error_prompt = {
                    "role": "user",
                    "content": (
                        f"Unexpected execution result: {exec_result}\n"
                        "Full model-code reconciliation required:\n"
                        "1. Compare each mathematical constraint with code implementation\n"
                        "2. Verify objective function translation\n"
                        "3. Check for off-by-one errors in indices\n"
                        "4. Validate data types and numerical precision\n"
                        "Output in this EXACT format:\n"
                        "ISSUES: [identified discrepancies]\n"
                        "REVISIONS: [mathematical model corrections or 'NO_CHANGE']\n"
                        "CODE: ```python\n{code}\n```"
                    )
                }
                messages = new_messages
                messages.append(error_prompt)
                continue
            else:
                return "error", f"Execution failed with unexpected result: {exec_result}"
    
    # Fallback logic using execution history
    print(f"\n!!! Maximum attempts ({max_attempts}) exhausted !!!")
    if execution_history:
        last_success, last_result = execution_history[-1]
        if not last_success:
            return "error", f"Persistent code generation failure: {last_result}"
        elif "infeasible" in str(last_result).lower():
            return "infeasible", "INFEASIBLE"
        else:
            return "error", f"Final execution failed: {last_result}"
    
    return "error", "Maximum attempts reached without successful execution"



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

