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
    
    This agent:
    1. Generates mathematical models from natural language problems
    2. Produces executable Gurobi code
    3. Validates solutions through multiple debugging iterations
    4. Handles edge cases like infeasibility and numerical issues
    
    Args:
        user_question: Natural language description of optimization problem
        model_name: LLM model to use (default from config)
        max_attempts: Maximum debugging iterations for code generation
    
    Returns:
        Tuple of (success_flag, result_object)
    """
    
    # Phase 1: Mathematical Modeling
    modeling_messages = [
        {
            "role": "system",
            "content": (
                "As an operations research expert, translate the user's optimization problem "
                "into precise mathematical programming formulations.\n\n"
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
    
    math_model = query_llm(modeling_messages, model_name)
    print("【Mathematical Model】\n", "-" * 50, "\n", math_model, "\n", "-" * 50)
    
    # Phase 2: Code Generation with Enhanced Context
    code_generation_messages = modeling_messages + [
        {"role": "assistant", "content": math_model},
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
    
    is_solve_success, result, messages = generate_or_code_solver(
        code_generation_messages, 
        model_name, 
        max_attempts,
        validation_mode="strict"
    )
    
    # Phase 3: Intelligent Debugging Cycles
    attempt_count = 0
    max_debug_cycles = 2
    
    while attempt_count < max_debug_cycles and not (is_solve_success and is_valid_solution(result)):
        attempt_count += 1
        print(f"\n🔧 Debugging Cycle {attempt_count}/{max_debug_cycles}")
        
        if is_solve_success:
            # Solution found but validation failed
            debug_prompt = analyze_solution_issue(result)
            print(f"Solution issue detected: {debug_prompt}")
        else:
            # Code execution failed
            debug_prompt = (
                "The Gurobi code contains errors. Diagnose and fix:\n"
                "1. Check variable/constraint naming conflicts\n"
                "2. Verify mathematical expression implementation\n"
                "3. Validate index ranges in loops\n"
                "4. Ensure proper Gurobi API usage\n"
                "5. Test with smaller instances if possible\n\n"
                "Provide corrected code in ```python\n{code}\n``` format."
            )
        
        messages.append({
            "role": "user",
            "content": debug_prompt
        })
        
        is_solve_success, result, messages = generate_or_code_solver(
            messages,
            model_name,
            max_attempts=1,
            validation_mode="debug"
        )
    
    # Phase 4: Final Validation and Reporting
    if is_solve_success and is_valid_solution(result):
        print("\n✅ SUCCESS: Valid optimization solution obtained")
        result = format_solution_output(result, include_statistics=True)
    elif is_solve_success:
        print("\n⚠️ WARNING: Solution found but validation issues remain")
        result = format_solution_output(result, include_warnings=True)
    else:
        print("\n❌ FAILURE: Maximum debugging cycles exhausted")
        result = {
            "status": "failed",
            "error": "Unable to generate executable solution",
            "debug_info": extract_debug_info(messages[-3:]),
            "partial_model": math_model
        }
    
    return is_solve_success, result

# Helper functions referenced in the main function
def is_valid_solution(solution_obj):
    """Validate solution feasibility and numerical stability."""
    if not solution_obj or not hasattr(solution_obj, 'status'):
        return False
    
    valid_statuses = ['OPTIMAL', 'FEASIBLE', 'SUBOPTIMAL']
    return (
        solution_obj.status in valid_statuses and
        hasattr(solution_obj, 'obj_val') and
        abs(solution_obj.obj_val) < 1e15 and  # Reasonable bound
        solution_obj.runtime < 3600  # Less than 1 hour
    )

def analyze_solution_issue(solution_obj):
    """Diagnose common solution problems."""
    if not hasattr(solution_obj, 'status'):
        return "Solution object missing status attribute"
    
    if solution_obj.status == 'INFEASIBLE':
        return (
            "Model is infeasible. Please:\n"
            "1. Check constraint logic and bounds\n"
            "2. Verify demand/capacity consistency\n"
            "3. Relax overly restrictive constraints\n"
            "4. Add feasibility analysis (IIS) if available\n\n"
            "Provide revised code with feasibility improvements."
        )
    elif solution_obj.status == 'UNBOUNDED':
        return (
            "Model is unbounded. Please:\n"
            "1. Add bounds to unbounded variables\n"
            "2. Check objective function coefficients\n"
            "3. Verify minimization vs maximization\n\n"
            "Provide bounded formulation."
        )
    elif solution_obj.status == 'INF_OR_UNBD':
        return "Model is infeasible or unbounded. Review constraint/variable definitions."
    elif hasattr(solution_obj, 'obj_val') and abs(solution_obj.obj_val) > 1e10:
        return "Solution exhibits numerical instability. Check scaling and large coefficients."
    else:
        return "Unknown solution issue. Review model formulation."

def format_solution_output(solution_obj, include_statistics=False, include_warnings=False):
    """Format solution for clear presentation."""
    output = {
        "status": getattr(solution_obj, 'status', 'UNKNOWN'),
        "objective_value": getattr(solution_obj, 'obj_val', None),
        "solution_time": getattr(solution_obj, 'runtime', None),
        "gap": getattr(solution_obj, 'mipgap', None)
    }
    
    if include_statistics:
        output["statistics"] = {
            "variables": getattr(solution_obj, 'num_vars', 0),
            "constraints": getattr(solution_obj, 'num_constrs', 0),
            "iterations": getattr(solution_obj, 'iter_count', 0)
        }
    
    if include_warnings:
        output["warnings"] = extract_warnings(solution_obj)
    
    # Extract variable values for key decisions
    if hasattr(solution_obj, 'getVars'):
        output["decision_variables"] = extract_key_decisions(solution_obj)
    
    return output

def extract_debug_info(message_history):
    """Extract debugging insights from conversation history."""
    debug_info = []
    for msg in message_history[-3:]:  # Last 3 exchanges
        if msg["role"] == "user" and "error" in msg["content"].lower():
            debug_info.append(msg["content"][:200])  # First 200 chars
    return debug_info if debug_info else ["No specific debug info available"]

def extract_key_decisions(solution_obj):
    """Extract non-zero or important variable values."""
    key_vars = {}
    try:
        for v in solution_obj.getVars():
            if abs(v.x) > 1e-6:  # Non-zero values
                key_vars[v.varName] = round(v.x, 4)
    except:
        pass
    return key_vars



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

