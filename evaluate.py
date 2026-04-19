import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from typing import Any, Optional, Tuple

from new_utils import query_llm


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
POPULATIONS_DIR = os.path.join(WORKSPACE_DIR, "populations")
PROMPT_DIR = os.path.join(WORKSPACE_DIR, "prompt")
KNOWLEDGES_FILE = os.path.join(PROMPT_DIR, "knowledges.txt")
CHAIN_PROMPT_FILE = os.path.join(PROMPT_DIR, "chain.txt")
TOOL_FILENAME = "tool.txt"
NEW_UTILS_FILENAME = "new_utils.py"
TOOL_FILE = os.path.join(PROMPT_DIR, TOOL_FILENAME)
NEW_UTILS_FILE = os.path.join(WORKSPACE_DIR, NEW_UTILS_FILENAME)

AGENT_CHAIN_FILENAME = "agent_chain.json"
AGENT_CODE_FILENAME = "or_agent.py"
CHAIN_RAW_FILENAME = "chain_raw_response.txt"
LLM_RAW_FILENAME = "llm_raw_response.txt"
EVAL_STDOUT_FILENAME = "eval_stdout.txt"
EVAL_STDERR_FILENAME = "eval_stderr.txt"
EVAL_RESULT_FILENAME = "eval_result.json"
RUN_TIMEOUT_SECONDS = 3600

REQUIRED_CHAIN_FIELDS = ["phase", "type", "action", "start_state", "end_state", "key"]
VALID_TYPES = {"code", "prompt", "tool"}
TOTAL_PATTERN = re.compile(r"\[Total\s+(\d+)\]\s+run\s+pass:\s*(\d+),\s*solve\s+correct:\s*(\d+)")
FAIL_PATTERN = re.compile(r"\[Total\s+fails\s+(\d+)\]\s+error\s+datas:\s*(\[[^\n]*\])")


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _print_progress(stage: str, current: int, total: int, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"[进度] {stage}: {current}/{total}{suffix}")


def _render_template(template: str, **kwargs: str) -> str:
    return template.format_map(_SafeDict(kwargs))


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_python_code(text: str) -> Tuple[str, bool]:
    # 兼容三种常见返回：纯代码、整段围栏代码、多段围栏代码。
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
    # LLM 可能混入解释文本，这里尽量提取首个完整 JSON 数组。
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
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return idx

    return -1


def _validate_chain_items(items: list[dict[str, Any]]) -> None:
    # 统一校验：字段完整、类型合法、状态连续。
    if not isinstance(items, list) or not items:
        raise ValueError("链路必须是非空 JSON 数组")

    previous_end_state: Optional[str] = None
    for index, item in enumerate(items, start=1):
        _validate_chain_item(item, index)
        previous_end_state = _validate_chain_transition(previous_end_state, item, index)


def _validate_chain_item(item: Any, index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"第{index}个动作不是 JSON 对象")

    for field in REQUIRED_CHAIN_FIELDS:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"第{index}个动作缺少有效字段: {field}")

    if item["type"].strip() not in VALID_TYPES:
        raise ValueError(f"第{index}个动作 type 非法: {item['type']}")


def _validate_chain_transition(previous_end_state: Optional[str], item: dict[str, Any], index: int) -> str:
    if previous_end_state is not None and item["start_state"].strip() != previous_end_state:
        raise ValueError(f"第{index}个动作的 start_state 与前一步 end_state 不连续")
    return item["end_state"].strip()


def _list_generation_dirs(generation: int) -> list[tuple[int, str]]:
    # 仅匹配 ex{generation}_pX，避免误读其它目录。
    pattern = re.compile(rf"^ex{generation}_p(\d+)$")
    result: list[tuple[int, str]] = []

    if not os.path.isdir(POPULATIONS_DIR):
        return result

    for name in os.listdir(POPULATIONS_DIR):
        m = pattern.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        result.append((idx, os.path.join(POPULATIONS_DIR, name)))

    result.sort(key=lambda x: x[0])
    return result


def _scan_population_indices() -> dict[int, list[int]]:
    # 扫描 populations 下的 exG_pI，返回每代已有的个体编号列表。
    pattern = re.compile(r"^ex(\d+)_p(\d+)$")
    generation_to_indices: dict[int, list[int]] = {}

    if not os.path.isdir(POPULATIONS_DIR):
        return generation_to_indices

    for name in os.listdir(POPULATIONS_DIR):
        m = pattern.match(name)
        if not m:
            continue
        generation = int(m.group(1))
        index = int(m.group(2))
        generation_to_indices.setdefault(generation, []).append(index)

    for generation in generation_to_indices:
        generation_to_indices[generation].sort()
    return generation_to_indices


def _is_generation_complete(generation: int, expected_count: int) -> bool:
    # 完整判定：存在且恰好是 1..expected_count 的连续编号。
    if expected_count <= 0:
        return False
    dirs = _list_generation_dirs(generation)
    if len(dirs) != expected_count:
        return False
    actual_indices = [idx for idx, _ in dirs]
    return actual_indices == list(range(1, expected_count + 1))


def _remove_generation_dirs(generation: int) -> None:
    # 删除某一代全部 ex{generation}_pX 目录，用于从 p1 重新生成。
    for _, pop_dir in _list_generation_dirs(generation):
        shutil.rmtree(pop_dir, ignore_errors=True)


def _remove_generations_from(start_generation: int, max_generation: int) -> None:
    # 删除 start_generation..max_generation 的所有个体目录，避免脏数据影响续跑。
    for generation in range(start_generation, max_generation + 1):
        _remove_generation_dirs(generation)


def _plan_resume_start(total_offspring: Optional[int], max_generation: int = 5) -> Optional[tuple[int, int]]:
    # 自动识别从哪一轮继续：若某目标代不完整，则从该代 p1 重跑。
    generation_map = _scan_population_indices()
    if 1 not in generation_map or not generation_map[1]:
        raise RuntimeError("未检测到 ex1 父代种群，无法继续进化")

    for src_gen in range(1, max_generation):
        dst_gen = src_gen + 1
        if total_offspring is not None:
            expected_count = total_offspring
        else:
            expected_count = len(_list_generation_dirs(src_gen))

        if expected_count <= 0:
            raise RuntimeError(f"ex{src_gen} 无可用父代个体，无法生成 ex{dst_gen}")

        if not _list_generation_dirs(dst_gen):
            print(f"[恢复] 检测到 ex{dst_gen} 不存在，将清理 ex{dst_gen}..ex{max_generation} 并从 p1 重新进化")
            _remove_generations_from(dst_gen, max_generation)
            return src_gen, dst_gen

        if not _is_generation_complete(dst_gen, expected_count):
            print(
                f"[恢复] 检测到 ex{dst_gen} 不完整，将清理 ex{dst_gen}..ex{max_generation} 并从 p1 重新进化"
            )
            _remove_generations_from(dst_gen, max_generation)
            return src_gen, dst_gen

    return None


def _build_population_chain(generation: int) -> list[dict[str, Any]]:
    # 从上一代个体提取 chain + or_agent，形成进化输入池。
    population_chain: list[dict[str, Any]] = []

    for idx, pdir in _list_generation_dirs(generation):
        chain_path = os.path.join(pdir, AGENT_CHAIN_FILENAME)
        code_path = os.path.join(pdir, AGENT_CODE_FILENAME)
        if not (os.path.exists(chain_path) and os.path.exists(code_path)):
            continue

        with open(chain_path, "r", encoding="utf-8") as f:
            chain = json.load(f)
        _validate_chain_items(chain)

        code = _read_text(code_path)
        population_chain.append(
            {
                "index": idx,
                "name": f"ex{generation}_p{idx}",
                "dir": pdir,
                "chain": chain,
                "or_agent": code,
            }
        )

    if not population_chain:
        raise RuntimeError(f"未找到可用个体：populations/ex{generation}_pX")

    snapshot_path = os.path.join(POPULATIONS_DIR, f"ex{generation}_population_chain.json")
    # 保存快照便于复现实验和排查问题。
    _write_json(snapshot_path, population_chain)
    print(f"[快照] population_chain 已保存 -> {snapshot_path}")
    return population_chain


def _parse_eval_metrics(stdout_text: str) -> tuple[bool, dict[str, Any]]:
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


def _run_agent_eval(pop_dir: str, timeout_sec: int = RUN_TIMEOUT_SECONDS) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, AGENT_CODE_FILENAME],
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
        "passed": proc.returncode == 0 and ok_format,
        "returncode": proc.returncode,
        "format_ok": ok_format,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "metrics": metrics,
    }


def _persist_eval_result(pop_dir: str, eval_result: dict[str, Any]) -> None:
    _write_text(os.path.join(pop_dir, EVAL_STDOUT_FILENAME), eval_result.get("stdout", ""))
    _write_text(os.path.join(pop_dir, EVAL_STDERR_FILENAME), eval_result.get("stderr", ""))
    _write_json(
        os.path.join(pop_dir, EVAL_RESULT_FILENAME),
        {
            "passed": eval_result.get("passed", False),
            "returncode": eval_result.get("returncode", -1),
            "format_ok": eval_result.get("format_ok", False),
            "metrics": eval_result.get("metrics", {}),
        },
    )


def _build_fix_prompt(eval_result: dict[str, Any]) -> str:
    if eval_result.get("returncode", 1) != 0:
        return (
            "or_agent.py 执行失败，请修复。\n"
            f"returncode={eval_result.get('returncode')}\n"
            f"stderr:\n{eval_result.get('stderr', '')}\n"
            "请返回完整可运行 Python 代码，不要解释。"
        )

    return (
        "or_agent.py 输出格式错误，请修复。\n"
        "必须在评测结束时严格输出两行：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")\n"
        f"stdout片段:\n{(eval_result.get('stdout') or '')[-2000:]}\n"
        "请返回完整可运行 Python 代码，不要解释。"
    )


def _fix_code_with_retries(
    pop_dir: str,
    model_name: str,
    knowledges_text: str,
    max_self_fix_attempts: int = 3,
) -> dict[str, Any]:
    code = _read_text(os.path.join(pop_dir, AGENT_CODE_FILENAME))
    raw_response = _read_text(os.path.join(pop_dir, LLM_RAW_FILENAME)) if os.path.exists(os.path.join(pop_dir, LLM_RAW_FILENAME)) else code

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "你是 Python 调试专家。请修复代码并返回完整可运行代码。",
        },
        {
            "role": "user",
            "content": (
                "以下是当前 or_agent.py，请先尝试自我纠错。\n"
                "代码必须包含 run_eval，并输出两行总评格式。\n\n"
                f"{code}"
            ),
        },
    ]

    last_eval: dict[str, Any] = {
        "passed": False,
        "returncode": -1,
        "format_ok": False,
        "stdout": "",
        "stderr": "",
        "metrics": {},
    }

    for _ in range(max_self_fix_attempts):
        try:
            last_eval = _run_agent_eval(pop_dir)
        except subprocess.TimeoutExpired:
            last_eval = {
                "passed": False,
                "returncode": -1,
                "format_ok": False,
                "stdout": "",
                "stderr": "执行超时",
                "metrics": {},
            }

        _persist_eval_result(pop_dir, last_eval)
        if last_eval.get("passed"):
            return last_eval

        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": _build_fix_prompt(last_eval)})
        raw_response = query_llm(messages, model_name=model_name)
        code, _ = _extract_python_code(raw_response)
        _write_text(os.path.join(pop_dir, AGENT_CODE_FILENAME), code)
        _write_text(os.path.join(pop_dir, LLM_RAW_FILENAME), raw_response)

    # 3次自纠失败后，使用知识库兜底纠错。
    knowledge_messages = [
        {
            "role": "system",
            "content": "你是 Python 调试专家。必须结合知识库进行修复。",
        },
        {
            "role": "user",
            "content": (
                "请基于 knowledges.txt 修复当前代码。\n"
                "修复目标：可运行且严格输出总评两行。\n\n"
                f"[knowledges]\n{knowledges_text}\n\n"
                f"[code]\n{code}\n\n"
                f"[error]\n{_build_fix_prompt(last_eval)}\n"
                "请仅输出完整 Python 代码。"
            ),
        },
    ]
    raw_response = query_llm(knowledge_messages, model_name=model_name)
    code, _ = _extract_python_code(raw_response)
    _write_text(os.path.join(pop_dir, AGENT_CODE_FILENAME), code)
    _write_text(os.path.join(pop_dir, LLM_RAW_FILENAME), raw_response)

    try:
        last_eval = _run_agent_eval(pop_dir)
    except subprocess.TimeoutExpired:
        last_eval = {
            "passed": False,
            "returncode": -1,
            "format_ok": False,
            "stdout": "",
            "stderr": "执行超时",
            "metrics": {},
        }
    _persist_eval_result(pop_dir, last_eval)
    return last_eval


def _score_from_eval(eval_result: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
    solve_correct = int(metrics.get("solve_correct", 0) or 0)
    run_pass = int(metrics.get("run_pass", 0) or 0)
    total_fails = int(metrics.get("total_fails", 10**9) or 10**9)
    passed = 1 if eval_result.get("passed") else 0
    return passed, solve_correct, run_pass, -total_fails


def _load_eval_result_from_disk(pop_dir: str) -> dict[str, Any]:
    # 从已有 eval_result.json 直接读取，无需重新测试。
    eval_path = os.path.join(pop_dir, EVAL_RESULT_FILENAME)
    if not os.path.exists(eval_path):
        return {
            "passed": False,
            "returncode": -1,
            "format_ok": False,
            "metrics": {},
        }
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[警告] 读取 {eval_path} 失败: {e}")
        return {
            "passed": False,
            "returncode": -1,
            "format_ok": False,
            "metrics": {},
        }


def _rank_parents_from_existing_eval(
    population_chain: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # 直接从已有 eval_result.json 读取分数并排序，不重新测试。
    for parent in population_chain:
        eval_result = _load_eval_result_from_disk(parent["dir"])
        parent["eval_result"] = eval_result
        parent["score"] = _score_from_eval(eval_result)

    population_chain.sort(key=lambda x: x.get("score", (0, 0, 0, -10**9)), reverse=True)

    ranking = []
    for rank, p in enumerate(population_chain, start=1):
        m = p.get("eval_result", {}).get("metrics", {})
        ranking.append(
            {
                "rank": rank,
                "name": p["name"],
                "passed": p.get("eval_result", {}).get("passed", False),
                "solve_correct": int(m.get("solve_correct", 0) or 0),
                "run_pass": int(m.get("run_pass", 0) or 0),
                "total_fails": int(m.get("total_fails", 0) or 0),
            }
        )
    _write_json(os.path.join(POPULATIONS_DIR, "parent_ranking.json"), ranking)
    return population_chain


def _build_tool_union() -> str:
    # chain 抽取与代码生成都使用统一的工具上下文。
    tool_doc = _read_text(TOOL_FILE)
    new_utils_source = _read_text(NEW_UTILS_FILE)
    return "[tool.txt]\n" + tool_doc + "\n\n[new_utils.py源码]\n" + new_utils_source


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


def _generate_chain_from_code(
    agent_code: str,
    model_name: str,
    chain_template: str,
    tool_union: str,
    variant_index: int,
    total_variants: int,
    max_attempts: int = 3,
) -> tuple[list[dict[str, str]], str]:
    # 用 chain 提示词把代码反向抽取为可校验的状态-动作链。
    messages = _build_chain_messages(
        chain_template=chain_template,
        agent_code=agent_code,
        tool_union=tool_union,
        variant_index=variant_index,
        total_variants=total_variants,
    )

    last_response = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"[LLM] 代码反向抽取链路，第 {attempt}/{max_attempts} 次")
        last_response = query_llm(messages, model_name=model_name)
        json_text = _extract_json_text(last_response)

        try:
            parsed = json.loads(json_text)
            _validate_chain_items(parsed)
            return parsed, last_response
        except Exception as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你输出的链路 JSON 无法使用。"
                        f"错误：{last_error}。"
                        "请仅输出合法 JSON 数组，保持状态连续，不要解释。"
                    ),
                }
            )

    raise RuntimeError(f"生成 chain 失败: {last_error or '未知错误'}")


def _generate_crossover_chain(
    population_chain: list[dict[str, Any]],
    model_name: str,
    offspring_index: int,
    total_offspring: int,
) -> tuple[list[dict[str, str]], str]:
    # 交叉算子：让 LLM 在全体父代链路上做“状态合并 + 新路径重组”。
    prompt = (
        "你将基于同一代所有个体的 chain 进行交叉。\n"
        "要求：\n"
        "1. 先合并语义相同或相似的状态节点。\n"
        "2. 在合并后的状态网络中选择一条新的可执行路径。\n"
        "3. 输出新的 agent_chain.json（JSON 数组）。\n"
        "4. 每个元素必须包含 phase,type,action,start_state,end_state,key。\n"
        "5. 状态必须严格连续。\n"
        "6. type 只能是 code/prompt/tool。\n"
        "7. 只输出 JSON 数组，不要解释。\n\n"
        f"目标后代编号：{offspring_index}/{total_offspring}\n\n"
        "上一代 population_chain 如下：\n"
        f"{json.dumps(population_chain, ensure_ascii=False, indent=2)}"
    )
    messages = [
        {"role": "system", "content": "你是优化智能体进化算子。严格输出 JSON。"},
        {"role": "user", "content": prompt},
    ]

    last_response = ""
    last_error = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"[LLM] 交叉生成链路，第 {attempt}/{max_attempts} 次")
        last_response = query_llm(messages, model_name=model_name)
        json_text = _extract_json_text(last_response)
        try:
            parsed = json.loads(json_text)
            _validate_chain_items(parsed)
            return parsed, last_response
        except Exception as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "输出不满足要求。"
                        f"错误：{last_error}。"
                        "请重新输出唯一合法 JSON 数组。"
                    ),
                }
            )

    raise RuntimeError(f"交叉生成 chain 失败: {last_error or '未知错误'}")


def _generate_code_from_chain_and_knowledge(
    chain_items: list[dict[str, str]],
    knowledges_text: str,
    model_name: str,
) -> tuple[str, str]:
    # 交叉产物先得到新 chain，再基于知识库生成代码。
    prompt = (
        "请根据给定的 agent_chain.json 与 knowledges.txt 生成完整可运行的 or_agent.py。\n"
        "要求：\n"
        "1. 工作流与 chain 保持一致。\n"
        "2. 只输出完整 Python 代码，不要解释，不要代码围栏。\n"
        "3. 代码应可直接保存为 or_agent.py。\n"
        "4. 必须包含训练集评测入口（run_eval），并严格输出：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")\n\n"
        "[agent_chain.json]\n"
        f"{json.dumps(chain_items, ensure_ascii=False, indent=2)}\n\n"
        "[knowledges.txt]\n"
        f"{knowledges_text}"
    )
    messages = [
        {"role": "system", "content": "你是运筹优化智能体代码生成器。"},
        {"role": "user", "content": prompt},
    ]
    raw = query_llm(messages, model_name=model_name)
    code, _ = _extract_python_code(raw)
    if not code.strip():
        raise RuntimeError("根据 chain+knowledge 生成代码失败：空输出")
    return code, raw


def _mutate_code_with_knowledge(
    parent_code: str,
    knowledges_text: str,
    model_name: str,
) -> tuple[str, str]:
    # 知识学习变异：父代代码 + 知识库，强调吸收框架经验。
    prompt = (
        "请对上一代个体代码进行知识学习变异。\n"
        "要求：\n"
        "1. 学习并吸收 knowledges.txt 的框架与经验。\n"
        "2. 保持运筹优化智能体可运行。\n"
        "3. 输出完整新代码，不要解释，不要代码围栏。\n"
        "4. 不要原样照抄输入代码。\n"
        "5. 必须包含训练集评测入口（run_eval），并严格输出：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")\n\n"
        "[parent or_agent.py]\n"
        f"{parent_code}\n\n"
        "[knowledges.txt]\n"
        f"{knowledges_text}"
    )
    messages = [
        {"role": "system", "content": "你是优化智能体代码变异器。"},
        {"role": "user", "content": prompt},
    ]
    raw = query_llm(messages, model_name=model_name)
    code, _ = _extract_python_code(raw)
    if not code.strip():
        raise RuntimeError("知识学习变异失败：空输出")
    return code, raw


def _mutate_code_direct(parent_code: str, model_name: str) -> tuple[str, str]:
    # 直接变异：仅基于父代代码进行结构/提示策略扰动。
    prompt = (
        "请对上一代个体代码做直接变异，生成新的优化智能体代码。\n"
        "要求：\n"
        "1. 保持核心功能可运行。\n"
        "2. 可对提示词、流程细节、重试策略做改变。\n"
        "3. 输出完整新代码，不要解释，不要代码围栏。\n"
        "4. 不要原样照抄输入代码。\n"
        "5. 必须包含训练集评测入口（run_eval），并严格输出：\n"
        "print(f\"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}\")\n"
        "print(f\"[Total fails {len(error_datas)}] error datas: {error_datas}\")\n\n"
        "[parent or_agent.py]\n"
        f"{parent_code}"
    )
    messages = [
        {"role": "system", "content": "你是优化智能体代码变异器。"},
        {"role": "user", "content": prompt},
    ]
    raw = query_llm(messages, model_name=model_name)
    code, _ = _extract_python_code(raw)
    if not code.strip():
        raise RuntimeError("直接变异失败：空输出")
    return code, raw


def _save_offspring(
    dst_generation: int,
    offspring_index: int,
    strategy: str,
    parent_name: str,
    chain_items: list[dict[str, str]],
    chain_raw: str,
    code: str,
    code_raw: str,
) -> None:
    # 通用落盘：写代码、链路、原始响应与元信息。
    pop_dir = os.path.join(POPULATIONS_DIR, f"ex{dst_generation}_p{offspring_index}")
    os.makedirs(pop_dir, exist_ok=True)

    _write_json(os.path.join(pop_dir, AGENT_CHAIN_FILENAME), chain_items)
    _write_text(os.path.join(pop_dir, CHAIN_RAW_FILENAME), chain_raw)
    _write_text(os.path.join(pop_dir, AGENT_CODE_FILENAME), code)
    _write_text(os.path.join(pop_dir, LLM_RAW_FILENAME), code_raw)

    meta = {
        "strategy": strategy,
        "parent": parent_name,
        "generation": dst_generation,
        "population_index": offspring_index,
    }
    _write_json(os.path.join(pop_dir, "ea_meta.json"), meta)

    if os.path.exists(NEW_UTILS_FILE):
        shutil.copy2(NEW_UTILS_FILE, os.path.join(pop_dir, NEW_UTILS_FILENAME))
    if os.path.exists(TOOL_FILE):
        shutil.copy2(TOOL_FILE, os.path.join(pop_dir, TOOL_FILENAME))


def _save_elite_offspring(
    dst_generation: int,
    offspring_index: int,
    parent: dict[str, Any],
) -> None:
    # 精英保留：完整复制父代目录（含 log/result/eval 等），不做交叉或变异。
    pop_dir = os.path.join(POPULATIONS_DIR, f"ex{dst_generation}_p{offspring_index}")
    if os.path.exists(pop_dir):
        shutil.rmtree(pop_dir)
    shutil.copytree(parent["dir"], pop_dir)

    meta = {
        "strategy": "elite_keep",
        "parent": parent["name"],
        "generation": dst_generation,
        "population_index": offspring_index,
    }
    _write_json(os.path.join(pop_dir, "ea_meta.json"), meta)

    if os.path.exists(NEW_UTILS_FILE):
        shutil.copy2(NEW_UTILS_FILE, os.path.join(pop_dir, NEW_UTILS_FILENAME))
    if os.path.exists(TOOL_FILE):
        shutil.copy2(TOOL_FILE, os.path.join(pop_dir, TOOL_FILENAME))


def _validate_offspring_and_refresh_chain(
    pop_dir: str,
    model_name: str,
    knowledges_text: str,
    chain_template: str,
    tool_union: str,
    offspring_index: int,
    total_offspring: int,
) -> dict[str, Any]:
    eval_result = _fix_code_with_retries(
        pop_dir=pop_dir,
        model_name=model_name,
        knowledges_text=knowledges_text,
    )

    final_code = _read_text(os.path.join(pop_dir, AGENT_CODE_FILENAME))
    chain_items, chain_raw = _generate_chain_from_code(
        agent_code=final_code,
        model_name=model_name,
        chain_template=chain_template,
        tool_union=tool_union,
        variant_index=offspring_index,
        total_variants=total_offspring,
    )
    _write_json(os.path.join(pop_dir, AGENT_CHAIN_FILENAME), chain_items)
    _write_text(os.path.join(pop_dir, CHAIN_RAW_FILENAME), chain_raw)
    return eval_result


def evolve_generation(
    src_generation: int = 1,
    dst_generation: int = 2,
    total_offspring: Optional[int] = None,
    elite_rate: float = 0.2,
    crossover_rate: float = 0.5,
    learning_rate: float = 0.5,
    model_name: str = "",
    seed: int = 42,
) -> None:
    # 进化主流程：精英保留 -> 交叉 -> 变异（学习/直接）。
    if not (0 <= elite_rate <= 1):
        raise ValueError("elite_rate 必须在 [0, 1] 范围内")
    if not (0 <= crossover_rate <= 1):
        raise ValueError("crossover_rate 必须在 [0, 1] 范围内")
    if not (0 <= learning_rate <= 1):
        raise ValueError("learning_rate 必须在 [0, 1] 范围内")

    random.seed(seed)
    print(
        f"[开始] 进化运行 src=ex{src_generation} -> dst=ex{dst_generation} | "
        f"模型={model_name} | 随机种子={seed}"
    )

    population_chain = _build_population_chain(src_generation)
    parent_count = len(population_chain)
    if total_offspring is None:
        total_offspring = parent_count
    if total_offspring <= 0:
        raise ValueError("total_offspring 必须为正整数")

    chain_template = _read_text(CHAIN_PROMPT_FILE)
    knowledges_text = _read_text(KNOWLEDGES_FILE)
    tool_union = _build_tool_union()

    # 直接从已有 eval_result.json 读取父代分数并排序，不重新测试。
    population_chain = _rank_parents_from_existing_eval(
        population_chain=population_chain,
    )

    # 按要求统一“取整”（向下取整）分配名额。
    elite_count = int(total_offspring * elite_rate)
    elite_count = max(0, min(elite_count, parent_count, total_offspring))

    # 在精英保留后的剩余个体中继续分配交叉与变异。
    remainder_after_elite = total_offspring - elite_count
    crossover_count = int(remainder_after_elite * crossover_rate)
    crossover_count = max(0, min(crossover_count, remainder_after_elite))

    mutation_count = remainder_after_elite - crossover_count
    # 变异内部再按学习率拆分：知识学习变异 + 直接变异。
    learn_count = int(mutation_count * learning_rate)
    learn_count = max(0, min(learn_count, mutation_count))
    direct_count = mutation_count - learn_count

    print(
        "[计划] "
        f"父代={parent_count}, 子代={total_offspring}, "
        f"精英保留={elite_count}, 交叉={crossover_count}, 变异={mutation_count}, "
        f"知识学习变异={learn_count}, 直接变异={direct_count}"
    )

    offspring_index = 1

    # 0) Elite keep: top-k by evaluation score (solve_correct 优先)
    # 精英保留只需复制，不进行测试。
    print(f"[阶段] 精英保留 开始 | 数量={elite_count}")
    elites = population_chain[:elite_count]
    for elite_i, parent in enumerate(elites, start=1):
        _print_progress("精英保留", elite_i, elite_count, f"目标=ex{dst_generation}_p{offspring_index}")
        _save_elite_offspring(
            dst_generation=dst_generation,
            offspring_index=offspring_index,
            parent=parent,
        )
        print(f"[完成] ex{dst_generation}_p{offspring_index} <- 精英保留 ({parent['name']})")
        offspring_index += 1
    print("[阶段] 精英保留 结束")

    # 1) Crossover: all parent chains -> merge similar states -> new path -> chain -> code
    print(f"[阶段] 交叉 开始 | 数量={crossover_count}")
    for cross_i in range(1, crossover_count + 1):
        _print_progress("交叉", cross_i, crossover_count, f"目标=ex{dst_generation}_p{offspring_index}")
        chain_items, chain_raw = _generate_crossover_chain(
            population_chain=population_chain,
            model_name=model_name,
            offspring_index=offspring_index,
            total_offspring=total_offspring,
        )
        code, code_raw = _generate_code_from_chain_and_knowledge(
            chain_items=chain_items,
            knowledges_text=knowledges_text,
            model_name=model_name,
        )
        _save_offspring(
            dst_generation=dst_generation,
            offspring_index=offspring_index,
            strategy="crossover",
            parent_name="ALL",
            chain_items=chain_items,
            chain_raw=chain_raw,
            code=code,
            code_raw=code_raw,
        )
        cross_eval = _validate_offspring_and_refresh_chain(
            pop_dir=os.path.join(POPULATIONS_DIR, f"ex{dst_generation}_p{offspring_index}"),
            model_name=model_name,
            knowledges_text=knowledges_text,
            chain_template=chain_template,
            tool_union=tool_union,
            offspring_index=offspring_index,
            total_offspring=total_offspring,
        )
        print(f"[完成] ex{dst_generation}_p{offspring_index} <- 交叉")
        print(
            f"[评测] ex{dst_generation}_p{offspring_index} passed={cross_eval.get('passed')} "
            f"solve_correct={cross_eval.get('metrics', {}).get('solve_correct', 0)}"
        )
        offspring_index += 1
    print("[阶段] 交叉 结束")

    # 2) Mutation with knowledge learning
    print(f"[阶段] 知识学习变异 开始 | 数量={learn_count}")
    for learn_i in range(1, learn_count + 1):
        _print_progress("知识学习变异", learn_i, learn_count, f"目标=ex{dst_generation}_p{offspring_index}")
        parent = random.choice(population_chain)
        code, code_raw = _mutate_code_with_knowledge(
            parent_code=parent["or_agent"],
            knowledges_text=knowledges_text,
            model_name=model_name,
        )
        chain_items, chain_raw = _generate_chain_from_code(
            agent_code=code,
            model_name=model_name,
            chain_template=chain_template,
            tool_union=tool_union,
            variant_index=offspring_index,
            total_variants=total_offspring,
        )
        _save_offspring(
            dst_generation=dst_generation,
            offspring_index=offspring_index,
            strategy="mutation_knowledge",
            parent_name=parent["name"],
            chain_items=chain_items,
            chain_raw=chain_raw,
            code=code,
            code_raw=code_raw,
        )
        learn_eval = _validate_offspring_and_refresh_chain(
            pop_dir=os.path.join(POPULATIONS_DIR, f"ex{dst_generation}_p{offspring_index}"),
            model_name=model_name,
            knowledges_text=knowledges_text,
            chain_template=chain_template,
            tool_union=tool_union,
            offspring_index=offspring_index,
            total_offspring=total_offspring,
        )
        print(f"[完成] ex{dst_generation}_p{offspring_index} <- 知识学习变异 ({parent['name']})")
        print(
            f"[评测] ex{dst_generation}_p{offspring_index} passed={learn_eval.get('passed')} "
            f"solve_correct={learn_eval.get('metrics', {}).get('solve_correct', 0)}"
        )
        offspring_index += 1
    print("[阶段] 知识学习变异 结束")

    # 3) Direct mutation
    print(f"[阶段] 直接变异 开始 | 数量={direct_count}")
    for direct_i in range(1, direct_count + 1):
        _print_progress("直接变异", direct_i, direct_count, f"目标=ex{dst_generation}_p{offspring_index}")
        parent = random.choice(population_chain)
        code, code_raw = _mutate_code_direct(
            parent_code=parent["or_agent"],
            model_name=model_name,
        )
        chain_items, chain_raw = _generate_chain_from_code(
            agent_code=code,
            model_name=model_name,
            chain_template=chain_template,
            tool_union=tool_union,
            variant_index=offspring_index,
            total_variants=total_offspring,
        )
        _save_offspring(
            dst_generation=dst_generation,
            offspring_index=offspring_index,
            strategy="mutation_direct",
            parent_name=parent["name"],
            chain_items=chain_items,
            chain_raw=chain_raw,
            code=code,
            code_raw=code_raw,
        )
        direct_eval = _validate_offspring_and_refresh_chain(
            pop_dir=os.path.join(POPULATIONS_DIR, f"ex{dst_generation}_p{offspring_index}"),
            model_name=model_name,
            knowledges_text=knowledges_text,
            chain_template=chain_template,
            tool_union=tool_union,
            offspring_index=offspring_index,
            total_offspring=total_offspring,
        )
        print(f"[完成] ex{dst_generation}_p{offspring_index} <- 直接变异 ({parent['name']})")
        print(
            f"[评测] ex{dst_generation}_p{offspring_index} passed={direct_eval.get('passed')} "
            f"solve_correct={direct_eval.get('metrics', {}).get('solve_correct', 0)}"
        )
        offspring_index += 1
    print("[阶段] 直接变异 结束")

    print(f"[结束] 第 {dst_generation} 代生成完成 | 总个体数={total_offspring}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5轮进化循环：ex1 -> ex2 -> ... -> ex5")
    parser.add_argument("-n", "--num", type=int, default=0, help="每代个体数量，默认0表示与前代等量")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["from-scratch", "resume"],
        default="from-scratch",
        help="进化模式：from-scratch 从 ex1 全量重跑；resume 自动识别中断代并从该代 p1 重跑",
    )
    parser.add_argument("--elite-rate", type=float, default=0.2, help="精英保留比例，默认 0.2")
    parser.add_argument("--cross-rate", type=float, default=0.5, help="交叉比例，默认 0.5")
    parser.add_argument("--learn-rate", type=float, default=0.5, help="变异中知识学习比例，默认 0.5")
    parser.add_argument("-m", "--model", type=str, default="", help="使用的模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    total_offspring = None if args.num <= 0 else args.num
    #自定义运行模式为 resume，方便调试时快速迭代。
    args.mode = "resume"
    max_generation = 5
    if args.mode == "from-scratch":
        # 从第一代开始重跑：先删除 ex2..ex5，再依次重建。
        for generation in range(2, max_generation + 1):
            _remove_generation_dirs(generation)
        start_src_gen = 1
        print("[模式] from-scratch：已清理 ex2..ex5，将从 ex1 开始完整进化")
    else:
        resume_pair = _plan_resume_start(total_offspring=total_offspring, max_generation=max_generation)
        if resume_pair is None:
            print("[模式] resume：已检测到 ex2..ex5 均完整，无需继续进化")
            return
        start_src_gen, _ = resume_pair
        print(f"[模式] resume：将从 ex{start_src_gen} 开始继续进化")

    # 逐轮进化：ex(start_src_gen) -> ... -> ex5
    for generation in range(start_src_gen, max_generation):
        src_gen = generation
        dst_gen = generation + 1
        print(f"\n\n========== 开始进化第 {generation} 轮: ex{src_gen} -> ex{dst_gen} ==========")
        evolve_generation(
            src_generation=src_gen,
            dst_generation=dst_gen,
            total_offspring=total_offspring,
            elite_rate=args.elite_rate,
            crossover_rate=args.cross_rate,
            learning_rate=args.learn_rate,
            model_name=args.model,
            seed=args.seed,
        )
        print(f"========== 完成进化第 {generation} 轮 ==========")


if __name__ == "__main__":
    main()

