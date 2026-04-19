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


def or_llm_agent(
    user_question,
    model_name=DEFAULT_MODEL_NAME,
    max_attempts=3,
    enable_validation=True,
    temperature=0.1,
    verbose=False
):
    """
    Enhanced LLM agent for OR optimization problems with iterative refinement.
    
    Args:
        user_question: Problem description in natural language
        model_name: LLM model to use
        max_attempts: Maximum number of code generation attempts
        enable_validation: Whether to validate model feasibility
        temperature: LLM sampling temperature (lower for more deterministic output)
        verbose: Whether to print detailed progress information
    
    Returns:
        Tuple of (success_flag, result_dict)
    """
    
    def validate_model_structure(math_model):
        """Validate basic mathematical model structure."""
        required_keywords = ['max', 'min', 'subject to', 's.t.', 'constraint']
        return any(keyword in math_model.lower() for keyword in required_keywords)
    
    def extract_variables(math_model):
        """Extract decision variables from mathematical model."""
        import re
        # Look for common variable patterns
        patterns = [
            r'x_\{[^}]+\}',  # x_{i,j}
            r'y_\{[^}]+\}',
            r'z_\{[^}]+\}',
            r'x\[[^\]]+\]',  # x[i,j]
            r'\b[a-z]_\{?[a-z0-9]+\}?\b'  # general variables
        ]
        variables = set()
        for pattern in patterns:
            variables.update(re.findall(pattern, math_model))
        return list(variables)
    
    # Stage 1: Mathematical Model Generation with validation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an operations research expert. Generate a precise mathematical model "
                "for the optimization problem.\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Use standard LP/MIP notation\n"
                "2. Define all decision variables with clear indices\n"
                "3. State objective function (max/min)\n"
                "4. List all constraints with proper domain restrictions\n"
                "5. Include all problem-specific parameters\n\n"
                "Output format:\n"
                "1. Decision variables\n"
                "2. Objective function\n"
                "3. Constraints\n"
                "4. Variable domains"
            ),
        },
        {"role": "user", "content": user_question},
    ]
    
    # Initial model generation
    math_model = query_llm(messages, model_name, temperature=temperature)
    
    if verbose:
        print("【Initial Mathematical Model】")
        print(math_model)
        print("-" * 50)
    
    # Validate model structure
    if enable_validation and not validate_model_structure(math_model):
        if verbose:
            print("⚠️ Model structure validation failed - requesting revision")
        
        messages.append({"role": "assistant", "content": math_model})
        messages.append({
            "role": "user",
            "content": (
                "The model lacks essential components. Please revise to include:\n"
                "1. Clear decision variable definitions\n"
                "2. Explicit objective function (maximize/minimize)\n"
                "3. Complete constraint set\n"
                "4. Variable type declarations (continuous, integer, binary)"
            )
        })
        math_model = query_llm(messages, model_name, temperature=temperature)
        
        if verbose:
            print("【Revised Mathematical Model】")
            print(math_model)
            print("-" * 50)
    
    # Extract variables for validation
    variables = extract_variables(math_model)
    if verbose and variables:
        print(f"📊 Detected variables: {variables}")
    
    # Stage 2: Code Generation with iterative refinement
    messages.append({"role": "assistant", "content": math_model})
    messages.append({
        "role": "user",
        "content": (
            "Generate Gurobi Python code with these SPECIFIC requirements:\n"
            "1. Import gurobipy as gp\n"
            "2. Create model with meaningful name\n"
            "3. Define ALL variables from mathematical model\n"
            "4. Set objective with correct sense\n"
            "5. Add ALL constraints\n"
            "6. Call model.optimize()\n"
            "7. Check status and handle infeasibility\n"
            "8. Print solution details including variable values\n"
            "9. Include error handling\n\n"
            "Format: ```python\n[code]\n```"
        )
    })
    
    # Code generation with multiple attempts
    attempt_results = []
    for attempt in range(max_attempts):
        if verbose:
            print(f"🔄 Code generation attempt {attempt + 1}/{max_attempts}")
        
        success, result, updated_messages = generate_or_code_solver(
            messages, 
            model_name, 
            max_attempts=1,
            temperature=temperature
        )
        
        if verbose:
            status = "✅ Success" if success else "❌ Failed"
            print(f"   Attempt result: {status}")
        
        attempt_results.append((success, result))
        messages = updated_messages
        
        if success:
            # Check if result is feasible
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'infeasible' and enable_validation:
                    if verbose:
                        print("⚠️ Model is infeasible - attempting repair")
                    
                    messages.append({
                        "role": "user",
                        "content": (
                            "The model is infeasible. Please analyze and fix:\n"
                            "1. Check constraint logic\n"
                            "2. Verify variable bounds\n"
                            "3. Ensure parameter values are correct\n"
                            "4. Relax overly restrictive constraints if needed\n\n"
                            "Provide revised code: ```python\n[code]\n```"
                        )
                    })
                    continue  # Try again with revised model
                else:
                    break  # Success with feasible solution
            else:
                break  # Success with some result
    
    # Final status evaluation
    final_success = any(success for success, _ in attempt_results)
    final_result = attempt_results[-1][1] if attempt_results else None
    
    if not final_success:
        if verbose:
            print("🚨 All attempts failed - attempting emergency recovery")
        
        messages.append({
            "role": "user",
            "content": (
                "Multiple attempts failed. Provide minimal working code that:\n"
                "1. Solves a simplified version\n"
                "2. Includes only essential constraints\n"
                "3. Has proper error handling\n\n"
                "Format: ```python\n[code]\n```"
            )
        })
        
        final_success, final_result, _ = generate_or_code_solver(
            messages, 
            model_name, 
            max_attempts=1,
            temperature=temperature
        )
    
    # Prepare comprehensive result
    result_dict = {
        'success': final_success,
        'mathematical_model': math_model,
        'solution': final_result,
        'attempts': len(attempt_results),
        'variables_identified': variables,
        'feasible': (
            isinstance(final_result, dict) and 
            final_result.get('status') == 'optimal'
        ) if final_result else None
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"🎯 Final Status: {'SUCCESS' if final_success else 'FAILED'}")
        if isinstance(final_result, dict):
            for key, value in final_result.items():
                print(f"   {key}: {value}")
    
    return final_success, result_dict



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

