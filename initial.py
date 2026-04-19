import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Callable, Tuple

from new_utils import query_llm


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(WORKSPACE_DIR, "prompt")
INITIALIZE_FILE = os.path.join(PROMPT_DIR, "initialize.txt")
KNOWLEDGES_FILE = os.path.join(PROMPT_DIR, "knowledges.txt")
CHANGE_FILE = os.path.join(PROMPT_DIR, "change.txt")
CHAIN_PROMPT_FILE = os.path.join(PROMPT_DIR, "chain.txt")
NEW_UTILS_FILE = os.path.join(WORKSPACE_DIR, "new_utils.py")
TOOL_FILE = os.path.join(PROMPT_DIR, "tool.txt")
POPULATIONS_DIR = os.path.join(WORKSPACE_DIR, "populations")
AGENT_FILENAME = "or_agent.py"
LLM_RAW_FILENAME = "llm_raw_response.txt"
EVAL_STDOUT_FILENAME = "eval_stdout.txt"
EVAL_STDERR_FILENAME = "eval_stderr.txt"
EVAL_RESULT_FILENAME = "eval_result.json"
RUN_TIMEOUT_SECONDS = 3600

TOTAL_PATTERN = re.compile(r"\[Total\s+(\d+)\]\s+run\s+pass:\s*(\d+),\s*solve\s+correct:\s*(\d+)")
FAIL_PATTERN = re.compile(r"\[Total\s+fails\s+(\d+)\]\s+error\s+datas:\s*(\[[^\n]*\])")


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _render_template(template: str, **kwargs: str) -> str:
    return template.format_map(_SafeDict(kwargs))


def _extract_python_code(text: str) -> Tuple[str, bool]:
    content = text.strip()
    if not content:
        return "", False

    outer = _strip_outer_fence(content)
    if outer is not None:
        return outer, True

    blocks = _collect_fenced_blocks(content.splitlines())
    if blocks:
        return max(blocks, key=len), True

    return content, False


def _strip_outer_fence(content: str) -> str | None:
    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return None


def _collect_fenced_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    in_fence = False
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not in_fence and stripped.startswith("```"):
            in_fence = True
            current = []
            continue
        if in_fence and stripped == "```":
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            in_fence = False
            current = []
            continue
        if in_fence:
            current.append(line)

    return blocks


def _extract_json_text(text: str) -> str:
    content, _ = _extract_python_code(text)
    content = content.strip()
    if not content:
        return ""

    if content.startswith("[") and content.endswith("]"):
        return content

    start = content.find("[")
    if start == -1:
        return content

    end = _find_json_array_end(content, start)
    if end != -1:
        return content[start : end + 1]

    return content


def _find_json_array_end(content: str, start: int) -> int:
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(content)):
        depth, in_string, escape = _update_json_parse_state(content[idx], depth, in_string, escape)
        if depth == 0 and not in_string:
            return idx

    return -1


def _update_json_parse_state(
    char: str,
    depth: int,
    in_string: bool,
    escape: bool,
) -> tuple[int, bool, bool]:
    if in_string:
        if escape:
            return depth, True, False
        if char == "\\":
            return depth, True, True
        if char == '"':
            return depth, False, False
        return depth, True, False

    if char == '"':
        return depth, True, False
    if char == "[":
        return depth + 1, False, False
    if char == "]":
        return depth - 1, False, False
    return depth, False, False


def _validate_chain_items(items: list[dict]) -> None:
    required_fields = ["phase", "type", "action", "start_state", "end_state", "key"]
    previous_end_state = None

    for index, item in enumerate(items, start=1):
        for field in required_fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第{index}个动作缺少有效字段: {field}")

        if previous_end_state is not None and item["start_state"].strip() != previous_end_state:
            raise ValueError(f"第{index}个动作的 start_state 与前一步 end_state 不连续")

        previous_end_state = item["end_state"].strip()


def _build_init_messages(
    initialize_template: str,
    tool_doc: str,
    new_utils_source: str,
    variant_index: int,
    total_variants: int,
) -> list[dict[str, str]]:
    user_prompt = _render_template(
        initialize_template,
        tool_doc=tool_doc,
        new_utils_source=new_utils_source,
    )
    user_prompt += (
        "\n\n"
        f"本次生成编号：{variant_index}/{total_variants}。"
        "请生成可运行的完整优化智能体脚本。"
        "必须包含训练集评测入口（run_eval）并在运行结束时严格输出两行：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")"
    )
    return [
        {
            "role": "system",
            "content": "你是资深Python与运筹优化智能体工程师。仅输出完整Python代码。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _build_knowledge_messages(
    change_template: str,
    knowledges_text: str,
    tool_union: str,
    new_utils_source: str,
    variant_index: int,
    total_variants: int,
) -> list[dict[str, str]]:
    prompt = _render_template(
        change_template,
        knowledges=knowledges_text,
        variant_index=str(variant_index),
        total_variants=str(total_variants),
        tool_doc=tool_union,
        new_utils_source=new_utils_source,
    )

    # Ensure this mode always sees the new constraint even if change.txt is old format.
    prompt += (
        "\n\n[附加强制约束]\n"
        "new_utils.py 与 tool.txt 共同组成工具调用规范，必须在代码中落实。\n"
        "请输出完整可运行 Python 文件，不要解释。\n"
        "必须包含训练集评测入口（run_eval）并在运行结束时严格输出两行：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")"
    )

    return [
        {
            "role": "system",
            "content": "你是资深Python与运筹优化智能体工程师。严格按输入知识生成完整代码。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _build_chain_messages(
    chain_template: str,
    agent_code: str,
    tool_union: str,
    variant_index: int,
    total_variants: int,
) -> list[dict[str, str]]:
    prompt = _render_template(
        chain_template,
        agent_code=agent_code,
        tool_doc=tool_union,
        variant_index=str(variant_index),
        total_variants=str(total_variants),
    )
    return [
        {
            "role": "system",
            "content": "你是优化智能体工作流架构师。输出必须是可逆、严格、可执行映射的 JSON。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _generate_chain_json(
    chain_template: str,
    agent_code: str,
    tool_union: str,
    model_name: str,
    variant_index: int,
    total_variants: int,
    max_attempts: int = 3,
) -> tuple[str, str]:
    messages = _build_chain_messages(
        chain_template=chain_template,
        agent_code=agent_code,
        tool_union=tool_union,
        variant_index=variant_index,
        total_variants=total_variants,
    )
    last_response = ""
    last_error = ""

    for _ in range(max_attempts):
        last_response = query_llm(messages, model_name=model_name)
        json_text = _extract_json_text(last_response)

        try:
            parsed = json.loads(json_text)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("输出不是非空 JSON 数组")
            _validate_chain_items(parsed)
            return json.dumps(parsed, ensure_ascii=False, indent=2), last_response
        except ValueError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你输出的结果无法作为严格可逆的链路 JSON 使用。"
                        f"错误：{last_error}。"
                        "请重新输出唯一的 JSON 数组结果，确保每个动作都包含 "
                        "phase、type、action、start_state、end_state、key 六个字段，"
                        "且状态严格连续，不要输出解释。"
                    ),
                }
            )

    raise RuntimeError(f"链路 JSON 生成失败: {last_error or '未知错误'}")


def _parse_eval_metrics(stdout_text: str) -> tuple[bool, dict]:
    total_match = TOTAL_PATTERN.search(stdout_text)
    fail_match = FAIL_PATTERN.search(stdout_text)
    if not total_match or not fail_match:
        return False, {"reason": "missing_required_summary_lines"}

    return True, {
        "total": int(total_match.group(1)),
        "run_pass": int(total_match.group(2)),
        "solve_correct": int(total_match.group(3)),
        "total_fails": int(fail_match.group(1)),
        "error_datas": fail_match.group(2),
    }


def _run_agent_eval(pop_dir: str, timeout_sec: int = RUN_TIMEOUT_SECONDS) -> dict:
    proc = subprocess.run(
        [sys.executable, AGENT_FILENAME],
        cwd=pop_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    ok_format, metrics = _parse_eval_metrics(stdout_text)

    return {
        "returncode": proc.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "format_ok": ok_format,
        "metrics": metrics,
        "passed": proc.returncode == 0 and ok_format,
    }


def _persist_eval_result(pop_dir: str, eval_result: dict) -> None:
    _write_text(os.path.join(pop_dir, EVAL_STDOUT_FILENAME), eval_result.get("stdout", ""))
    _write_text(os.path.join(pop_dir, EVAL_STDERR_FILENAME), eval_result.get("stderr", ""))
    _write_text(
        os.path.join(pop_dir, EVAL_RESULT_FILENAME),
        json.dumps(
            {
                "passed": eval_result.get("passed", False),
                "returncode": eval_result.get("returncode", -1),
                "format_ok": eval_result.get("format_ok", False),
                "metrics": eval_result.get("metrics", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _build_fix_prompt(error_info: dict) -> str:
    if error_info.get("returncode", 1) != 0:
        return (
            "你生成的 or_agent.py 执行失败。\n"
            f"returncode={error_info.get('returncode')}\n"
            f"stderr:\n{error_info.get('stderr', '')}\n"
            "请修复并重新输出完整可运行 Python 代码，不要解释。"
        )

    return (
        "你生成的 or_agent.py 虽然执行结束，但输出格式不符合要求。\n"
        "必须在评测结束时严格输出两行：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")\n"
        f"当前stdout末尾:\n{(error_info.get('stdout') or '')[-2000:]}\n"
        "请修复并重新输出完整可运行 Python 代码，不要解释。"
    )


def _generate_and_validate_agent(
    pop_dir: str,
    messages: list[dict[str, str]],
    model_name: str,
    knowledges_text: str,
    max_self_fix_attempts: int = 3,
) -> tuple[str, str, dict]:
    working_messages = list(messages)
    last_code = ""
    last_response = ""
    last_eval = {
        "passed": False,
        "returncode": -1,
        "stdout": "",
        "stderr": "",
        "format_ok": False,
        "metrics": {},
    }

    for _ in range(max_self_fix_attempts):
        last_response = query_llm(working_messages, model_name=model_name)
        last_code, _ = _extract_python_code(last_response)
        _write_text(os.path.join(pop_dir, AGENT_FILENAME), last_code)
        _write_text(os.path.join(pop_dir, LLM_RAW_FILENAME), last_response)

        try:
            last_eval = _run_agent_eval(pop_dir)
        except subprocess.TimeoutExpired:
            last_eval = {
                "passed": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "执行超时",
                "format_ok": False,
                "metrics": {},
            }

        _persist_eval_result(pop_dir, last_eval)
        if last_eval.get("passed"):
            return last_code, last_response, last_eval

        working_messages.append({"role": "assistant", "content": last_response})
        working_messages.append({"role": "user", "content": _build_fix_prompt(last_eval)})

    # 自纠失败后调用知识库纠错。
    knowledge_messages = [
        {
            "role": "system",
            "content": "你是Python调试专家。请基于知识库修复代码并返回完整代码。",
        },
        {
            "role": "user",
            "content": (
                "请使用下面知识库内容修复当前 or_agent.py。\n"
                "修复目标：可运行 + 输出格式严格符合要求。\n"
                "知识库：\n"
                f"{knowledges_text}\n\n"
                "当前代码：\n"
                f"{last_code}\n\n"
                "最近一次报错/格式问题信息：\n"
                f"{_build_fix_prompt(last_eval)}\n\n"
                "请只输出完整 Python 代码，不要解释。"
            ),
        },
    ]
    last_response = query_llm(knowledge_messages, model_name=model_name)
    last_code, _ = _extract_python_code(last_response)
    _write_text(os.path.join(pop_dir, AGENT_FILENAME), last_code)
    _write_text(os.path.join(pop_dir, LLM_RAW_FILENAME), last_response)
    try:
        last_eval = _run_agent_eval(pop_dir)
    except subprocess.TimeoutExpired:
        last_eval = {
            "passed": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "执行超时",
            "format_ok": False,
            "metrics": {},
        }
    _persist_eval_result(pop_dir, last_eval)
    return last_code, last_response, last_eval


def _generate_mode_variants(
    generation_id: int,
    mode_prefix: str,
    total_variants: int,
    start_index: int,
    population_total: int,
    model_name: str,
    chain_template: str,
    tool_union: str,
    knowledges_text: str,
    message_builder: Callable[[int, int], list[dict[str, str]]],
) -> None:
    os.makedirs(POPULATIONS_DIR, exist_ok=True)

    for i in range(1, total_variants + 1):
        population_index = start_index + i - 1
        target_dir = os.path.join(POPULATIONS_DIR, f"ex{generation_id}_p{population_index}")
        os.makedirs(target_dir, exist_ok=True)

        messages = message_builder(population_index, population_total)
        code, _, eval_result = _generate_and_validate_agent(
            pop_dir=target_dir,
            messages=messages,
            model_name=model_name,
            knowledges_text=knowledges_text,
        )

        chain_json, chain_response = _generate_chain_json(
            chain_template=chain_template,
            agent_code=code,
            tool_union=tool_union,
            model_name=model_name,
            variant_index=population_index,
            total_variants=population_total,
        )

        _write_text(os.path.join(target_dir, "agent_chain.json"), chain_json)
        _write_text(os.path.join(target_dir, "chain_raw_response.txt"), chain_response)

        if os.path.exists(NEW_UTILS_FILE):
            shutil.copy2(NEW_UTILS_FILE, os.path.join(target_dir, "new_utils.py"))
        if os.path.exists(TOOL_FILE):
            shutil.copy2(TOOL_FILE, os.path.join(target_dir, "tool.txt"))

        print(
            f"[done] {mode_prefix} p{population_index}/{population_total} -> {target_dir} | "
            "saved: or_agent.py, agent_chain.json"
            f" | test_passed={eval_result.get('passed')}"
            f" | solve_correct={eval_result.get('metrics', {}).get('solve_correct', 0)}"
        )


def generate_all(num_each: int = 3, model_name: str = "") -> None:
    if num_each <= 0:
        raise ValueError("num 必须为正整数")

    initialize_template = _read_text(INITIALIZE_FILE)
    change_template = _read_text(CHANGE_FILE)
    chain_template = _read_text(CHAIN_PROMPT_FILE)
    tool_doc = _read_text(TOOL_FILE)
    new_utils_source = _read_text(NEW_UTILS_FILE)

    tool_union = (
        "[tool.txt]\n"
        + tool_doc
        + "\n\n[new_utils.py源码]\n"
        + new_utils_source
    )

    knowledges_text = _read_text(KNOWLEDGES_FILE)

    init_count = num_each // 2
    knowledge_count = num_each - init_count

    _generate_mode_variants(
        generation_id=1,
        mode_prefix="initialize_mode",
        total_variants=init_count,
        start_index=1,
        population_total=num_each,
        model_name=model_name,
        chain_template=chain_template,
        tool_union=tool_union,
        knowledges_text=knowledges_text,
        message_builder=lambda i, n: _build_init_messages(
            initialize_template=initialize_template,
            tool_doc=tool_doc,
            new_utils_source=new_utils_source,
            variant_index=i,
            total_variants=n,
        ),
    )

    _generate_mode_variants(
        generation_id=1,
        mode_prefix="knowledge_mode",
        total_variants=knowledge_count,
        start_index=init_count + 1,
        population_total=num_each,
        model_name=model_name,
        chain_template=chain_template,
        tool_union=tool_union,
        knowledges_text=knowledges_text,
        message_builder=lambda i, n: _build_knowledge_messages(
            change_template=change_template,
            knowledges_text=knowledges_text,
            tool_union=tool_union,
            new_utils_source=new_utils_source,
            variant_index=i,
            total_variants=n,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "双通道生成优化智能体并自动转chain，输出目录为 populations/exX_pX。"
        )
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=6,
        help="总生成数量，默认 6（前一半 initialize，后一半 knowledge）",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="", help="使用的模型名"
    )
    args = parser.parse_args()

    generate_all(num_each=args.num, model_name=args.model)


if __name__ == "__main__":
    main()

