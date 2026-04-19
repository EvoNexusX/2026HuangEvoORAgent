import argparse
import os
import re
import subprocess
import sys
from typing import Dict, Optional, Tuple

from new_utils import extract_best_objective, eval_model_result, load_dataset


RESULT_FILE_PATTERN = re.compile(
    r"^(?P<prefix>\w+)_q(?P<qid>[^_]+)_(?P<count>\d+)_(?P<ts>\d{8}_\d{6})\.py$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated result scripts against dataset answers")
    parser.add_argument("--data_path", type=str, default="data/datasets/NL4OPT_with_optimal_solution.json", help="Dataset path")
    parser.add_argument("--result_dir", type=str, default="result", help="Directory containing generated .py files")
    parser.add_argument("--prefix", type=str, default="agent", help="File prefix to match, e.g. simple/agent")
    parser.add_argument("--start_i", type=int, default=0, help="Only evaluate question id >= start_i")
    parser.add_argument("--end_i", type=int, default=None, help="Only evaluate question id <= end_i")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout (seconds) per result script")
    return parser.parse_args()


def _is_in_range(qid_str: str, start_i: int, end_i: Optional[int]) -> bool:
    try:
        qid = int(qid_str)
    except (TypeError, ValueError):
        return True
    if qid < start_i:
        return False
    if end_i is not None and qid > end_i:
        return False
    return True


def find_latest_result_files(result_dir: str, prefix: str) -> Dict[str, str]:
    """Return latest result file path for each qid by timestamp then count."""
    latest: Dict[str, Tuple[str, int, str]] = {}

    for name in os.listdir(result_dir):
        match = RESULT_FILE_PATTERN.match(name)
        if not match:
            continue
        if match.group("prefix") != prefix:
            continue

        qid = match.group("qid")
        count = int(match.group("count"))
        ts = match.group("ts")
        path = os.path.join(result_dir, name)

        prev = latest.get(qid)
        if prev is None:
            latest[qid] = (ts, count, path)
            continue

        prev_ts, prev_count, _ = prev
        if ts > prev_ts or (ts == prev_ts and count > prev_count):
            latest[qid] = (ts, count, path)

    return {qid: item[2] for qid, item in latest.items()}


def run_result_script(script_path: str, timeout_sec: int) -> Tuple[bool, str]:
    """Execute generated script and return (success, objective_text_or_error)."""
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout_sec}s"
    except Exception as exc:
        return False, str(exc)

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"Script exited with code {proc.returncode}"
        return False, err

    best_obj = extract_best_objective(proc.stdout)
    return True, str(best_obj)


def evaluate_one_item(qid: str, answer, script_path: Optional[str], timeout_sec: int) -> Tuple[bool, bool]:
    print(f"=============== num {qid} ==================")

    if not script_path:
        print("No matching result script found.")
        print("------------------")
        print("[Final] run pass: False, solve correct: False")
        print()
        return False, False

    print(f"Using script: {script_path}")
    is_solve_success, llm_result = run_result_script(script_path, timeout_sec)
    if is_solve_success:
        print(f"Successfully executed code, optimal solution value: {llm_result}")
    else:
        print(f"Failed to execute code: {llm_result}")
    print("------------------")

    pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
    print(f"solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}")
    print(f"[Final] run pass: {pass_flag}, solve correct: {correct_flag}")
    print()
    return pass_flag, correct_flag


def main():
    args = parse_args()

    dataset = load_dataset(args.data_path)
    latest_files = find_latest_result_files(args.result_dir, args.prefix)

    pass_count = 0
    correct_count = 0
    error_datas = []

    eval_ids = []
    for i in dataset.keys():
        if _is_in_range(str(i), args.start_i, args.end_i):
            eval_ids.append(str(i))

    for i in eval_ids:
        d = dataset[i]
        answer = d.get("answer")
        script_path = latest_files.get(i)

        pass_flag, correct_flag = evaluate_one_item(i, answer, script_path, args.timeout)
        pass_count += 1 if pass_flag else 0
        correct_count += 1 if correct_flag else 0

        if not pass_flag or not correct_flag:
            error_datas.append(i)

    print(f"[Total {len(eval_ids)}] run pass: {pass_count}, solve correct: {correct_count}")
    print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")


if __name__ == "__main__":
    main()
