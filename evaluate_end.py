import argparse
import json
import math
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
ARCHITECTURE_GRAPH_FILENAME = "architecture_graph.json"
DEFAULT_ARCH_LEARNING_RATE = 0.5
DEFAULT_EXPLORATION_GAMMA = 0.5
DEFAULT_PRUNING_THRESHOLD = 0.1
DEFAULT_FORGETTING_HORIZON = 3
SPARSITY_EPSILON = 1e-9

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


def _fitness_from_eval(eval_result: dict[str, Any]) -> float:
    """把评测结果映射到论文中的 F(a)，范围尽量稳定在 [0, 1]。"""
    metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
    total = int(metrics.get("total", 0) or 0)
    solve_correct = int(metrics.get("solve_correct", 0) or 0)
    run_pass = int(metrics.get("run_pass", 0) or 0)
    if total <= 0:
        return 0.0
    solve_acc = solve_correct / total
    run_acc = run_pass / total
    format_bonus = 0.05 if eval_result.get("passed") else 0.0
    return max(0.0, min(1.0, 0.85 * solve_acc + 0.10 * run_acc + format_bonus))


def _edge_key_from_chain_item(item: dict[str, Any]) -> str:
    parts = [
        item.get("phase", ""),
        item.get("type", ""),
        item.get("action", ""),
        item.get("start_state", ""),
        item.get("end_state", ""),
        item.get("key", ""),
    ]
    return "||".join(str(part).strip() for part in parts)


def _edge_key_from_edge(edge: dict[str, Any]) -> str:
    parts = [
        edge.get("phase", ""),
        edge.get("type", ""),
        edge.get("action", ""),
        edge.get("from_state", edge.get("from", "")),
        edge.get("to_state", edge.get("to", "")),
        edge.get("key", ""),
    ]
    return "||".join(str(part).strip() for part in parts)


def _architecture_graph_path(generation: int) -> str:
    return os.path.join(POPULATIONS_DIR, f"ex{generation}_{ARCHITECTURE_GRAPH_FILENAME}")


def _edge_type_from_chain_type(chain_type: str) -> str:
    value = chain_type.strip()
    if value == "tool":
        return "tool"
    if value == "prompt":
        return "reason"
    return "work"


def _phase_from_edge_phase(phase: str) -> str:
    value = phase.strip()
    if value:
        return value
    return "unknown"


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if abs(max_value - min_value) <= 1e-12:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _refresh_node_scores(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    incident: dict[str, list[float]] = {node["id"]: [] for node in nodes}
    incident_edges: dict[str, list[str]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        w_fit = float(edge.get("w_fit", 0.0) or 0.0)
        edge_id = edge.get("id", "")
        for node_id in (edge.get("from"), edge.get("to")):
            if node_id in incident:
                incident[node_id].append(w_fit)
                incident_edges[node_id].append(edge_id)

    for node in nodes:
        node_id = node["id"]
        node["w"] = _safe_mean(incident.get(node_id, []), 0.0)
        node["w_fit"] = node["w"]
        node["incident_edges"] = incident_edges.get(node_id, [])


def _rebuild_arch_adj(graph: dict[str, Any]) -> None:
    adj: dict[str, list[tuple[int, str]]] = {node["id"]: [] for node in graph.get("nodes", [])}
    for edge_index, edge in enumerate(graph.get("edges", [])):
        adj.setdefault(edge["from"], []).append((edge_index, edge["to"]))
    graph["adj"] = adj


def _build_architecture_graph_from_population(
    population_chain: list[dict[str, Any]],
    generation: int,
) -> dict[str, Any]:
    state_to_nid: dict[str, str] = {}
    edge_by_key: dict[str, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def _get_or_create_node(state: str, phase: str) -> str:
        state_text = state.strip()
        if state_text not in state_to_nid:
            node_id = f"N{len(nodes)}"
            state_to_nid[state_text] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "state": state_text,
                    "phase": _phase_from_edge_phase(phase),
                    "w": 0.0,
                    "w_fit": 0.0,
                    "weak_count": 0,
                    "incident_edges": [],
                }
            )
        return state_to_nid[state_text]

    for parent in population_chain:
        fitness = float(parent.get("fitness", 0.0) or 0.0)
        parent_name = parent.get("name", "")
        chain = parent.get("chain", [])
        for item in chain:
            if not isinstance(item, dict):
                continue
            phase = item.get("phase", "")
            from_nid = _get_or_create_node(item.get("start_state", ""), phase)
            to_nid = _get_or_create_node(item.get("end_state", ""), phase)
            edge_key = _edge_key_from_chain_item(item)
            if edge_key not in edge_by_key:
                edge = {
                    "id": f"E{len(edges)}",
                    "from": from_nid,
                    "to": to_nid,
                    "from_state": item.get("start_state", "").strip(),
                    "to_state": item.get("end_state", "").strip(),
                    "phase": phase,
                    "type": item.get("type", ""),
                    "rho": _edge_type_from_chain_type(item.get("type", "")),
                    "action": item.get("action", ""),
                    "key": item.get("key", ""),
                    "edge_key": edge_key,
                    "w_fit": fitness,
                    "count": 0,
                    "w_sparse": 0.0,
                    "weak_count": 0,
                    "traversed_by": [],
                }
                edge_by_key[edge_key] = edge
                edges.append(edge)
            edge = edge_by_key[edge_key]
            edge["traversed_by"].append(parent_name)
            edge["count"] = int(edge.get("count", 0) or 0) + 1

    for edge in edges:
        edge["w_sparse"] = 1.0 / (math.log(2.0 + int(edge.get("count", 0) or 0)) + SPARSITY_EPSILON)

    graph = {
        "generation": generation,
        "nodes": nodes,
        "edges": edges,
        "state_to_nid": state_to_nid,
        "update_params": {
            "alpha": DEFAULT_ARCH_LEARNING_RATE,
            "tau": DEFAULT_PRUNING_THRESHOLD,
            "sigma": DEFAULT_FORGETTING_HORIZON,
            "epsilon": SPARSITY_EPSILON,
        },
    }
    _refresh_node_scores(graph)
    _rebuild_arch_adj(graph)
    return graph


def _load_architecture_graph(generation: int) -> Optional[dict[str, Any]]:
    path = _architecture_graph_path(generation)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        _rebuild_arch_adj(graph)
        return graph
    except Exception as exc:
        print(f"[警告] 读取架构图失败，将重建: {path} | {exc}")
        return None


def _save_architecture_graph(graph: dict[str, Any], generation: int) -> None:
    os.makedirs(POPULATIONS_DIR, exist_ok=True)
    graph_to_save = {k: v for k, v in graph.items() if k != "adj"}
    graph_to_save["generation"] = generation
    _write_json(_architecture_graph_path(generation), graph_to_save)


def _ensure_architecture_graph(
    population_chain: list[dict[str, Any]],
    generation: int,
) -> dict[str, Any]:
    graph = _load_architecture_graph(generation)
    if graph is not None:
        return graph
    graph = _build_architecture_graph_from_population(population_chain, generation)
    _save_architecture_graph(graph, generation)
    print(f"[快照] 初始架构图已保存 -> {_architecture_graph_path(generation)}")
    return graph


def _update_architecture_graph(
    current_graph: dict[str, Any],
    evaluated_population: list[dict[str, Any]],
    dst_generation: int,
    alpha: float = DEFAULT_ARCH_LEARNING_RATE,
    tau: float = DEFAULT_PRUNING_THRESHOLD,
    sigma: int = DEFAULT_FORGETTING_HORIZON,
) -> dict[str, Any]:
    """按论文 Eq.(2)-(4) 更新每条边与每个点的参数，并落盘新一代架构图。"""
    graph = json.loads(json.dumps({k: v for k, v in current_graph.items() if k != "adj"}, ensure_ascii=False))
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    state_to_nid = {
        node.get("state", "").strip(): node["id"]
        for node in graph["nodes"]
        if node.get("id") and node.get("state", "").strip()
    }
    edge_by_key = {edge.get("edge_key") or _edge_key_from_edge(edge): edge for edge in graph["edges"]}

    def _next_node_id() -> str:
        return f"N{len(graph['nodes'])}"

    def _next_edge_id() -> str:
        return f"E{len(graph['edges'])}"

    def _get_or_create_node(state: str, phase: str) -> str:
        state_text = state.strip()
        if state_text in state_to_nid:
            return state_to_nid[state_text]
        node_id = _next_node_id()
        state_to_nid[state_text] = node_id
        graph["nodes"].append(
            {
                "id": node_id,
                "state": state_text,
                "phase": _phase_from_edge_phase(phase),
                "w": 0.0,
                "w_fit": 0.0,
                "weak_count": 0,
                "incident_edges": [],
            }
        )
        return node_id

    traversed_fitness: dict[str, list[float]] = {key: [] for key in edge_by_key}
    traversed_by: dict[str, list[str]] = {key: [] for key in edge_by_key}
    for parent in evaluated_population:
        fitness = float(parent.get("fitness", 0.0) or 0.0)
        parent_name = parent.get("name", "")
        for item in parent.get("chain", []):
            if not isinstance(item, dict):
                continue
            edge_key = _edge_key_from_chain_item(item)
            if edge_key not in edge_by_key:
                phase = item.get("phase", "")
                from_nid = _get_or_create_node(item.get("start_state", ""), phase)
                to_nid = _get_or_create_node(item.get("end_state", ""), phase)
                edge = {
                    "id": _next_edge_id(),
                    "from": from_nid,
                    "to": to_nid,
                    "from_state": item.get("start_state", "").strip(),
                    "to_state": item.get("end_state", "").strip(),
                    "phase": phase,
                    "type": item.get("type", ""),
                    "rho": _edge_type_from_chain_type(item.get("type", "")),
                    "action": item.get("action", ""),
                    "key": item.get("key", ""),
                    "edge_key": edge_key,
                    "w_fit": 0.0,
                    "count": 0,
                    "w_sparse": 0.0,
                    "weak_count": 0,
                    "traversed_by": [],
                }
                edge_by_key[edge_key] = edge
                graph["edges"].append(edge)
                traversed_fitness[edge_key] = []
                traversed_by[edge_key] = []
            traversed_fitness.setdefault(edge_key, []).append(fitness)
            traversed_by.setdefault(edge_key, []).append(parent_name)

    for edge in graph["edges"]:
        edge_key = edge.get("edge_key") or _edge_key_from_edge(edge)
        edge["edge_key"] = edge_key
        fitness_values = traversed_fitness.get(edge_key, [])
        if fitness_values:
            old_w_fit = float(edge.get("w_fit", 0.0) or 0.0)
            avg_fitness = _safe_mean(fitness_values, 0.0)
            edge["w_fit"] = old_w_fit + alpha * (avg_fitness - old_w_fit)
            edge["count"] = int(edge.get("count", 0) or 0) + len(fitness_values)
            edge["traversed_by"] = traversed_by.get(edge_key, [])
        else:
            edge["traversed_by"] = []
        edge["w_sparse"] = 1.0 / (math.log(2.0 + int(edge.get("count", 0) or 0)) + SPARSITY_EPSILON)
        edge["weak_count"] = int(edge.get("weak_count", 0) or 0) + 1 if float(edge.get("w_fit", 0.0) or 0.0) < tau else 0

    _refresh_node_scores(graph)
    kept_node_ids = {
        node["id"]
        for node in graph["nodes"]
        if int(node.get("weak_count", 0) or 0) < sigma
    }
    for node in graph["nodes"]:
        node["weak_count"] = int(node.get("weak_count", 0) or 0) + 1 if float(node.get("w", 0.0) or 0.0) < tau else 0
    kept_node_ids = {
        node["id"]
        for node in graph["nodes"]
        if int(node.get("weak_count", 0) or 0) < sigma
    }
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if int(edge.get("weak_count", 0) or 0) < sigma
        and edge.get("from") in kept_node_ids
        and edge.get("to") in kept_node_ids
    ]
    used_node_ids = {edge.get("from") for edge in graph["edges"]} | {edge.get("to") for edge in graph["edges"]}
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] in used_node_ids or node["id"] in kept_node_ids]
    graph["state_to_nid"] = {node["state"]: node["id"] for node in graph["nodes"]}
    graph["update_params"] = {"alpha": alpha, "tau": tau, "sigma": sigma, "epsilon": SPARSITY_EPSILON}
    _refresh_node_scores(graph)
    _rebuild_arch_adj(graph)
    _save_architecture_graph(graph, dst_generation)
    print(f"[快照] 架构图参数已更新 -> {_architecture_graph_path(dst_generation)}")
    return graph


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
        parent["fitness"] = _fitness_from_eval(eval_result)

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
                "fitness": round(float(p.get("fitness", 0.0) or 0.0), 6),
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


# ─── 图搜索交叉算子 ───────────────────────────────────────────────

def _build_numbered_graph(
    population_chain: list[dict[str, Any]],
    generation: int,
    architecture_graph: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """汇总父代所有 agent_chain，构建带编号的 AOE 图。"""
    state_to_nid: dict[str, str] = {}
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, Any]] = []
    arch_edge_by_key = {
        edge.get("edge_key") or _edge_key_from_edge(edge): edge
        for edge in (architecture_graph or {}).get("edges", [])
    }

    def _get_or_create_node(state: str) -> str:
        s = state.strip()
        if s not in state_to_nid:
            nid = f"N{len(state_to_nid)}"
            state_to_nid[s] = nid
            nodes.append({"id": nid, "state": s})
        return state_to_nid[s]

    for parent in population_chain:
        chain = parent.get("chain", [])
        for item in chain:
            if not isinstance(item, dict):
                continue
            from_nid = _get_or_create_node(item.get("start_state", ""))
            to_nid = _get_or_create_node(item.get("end_state", ""))
            eid = f"E{len(edges)}"
            edge_key = _edge_key_from_chain_item(item)
            arch_edge = arch_edge_by_key.get(edge_key, {})
            edges.append({
                "id": eid,
                "from": from_nid,
                "to": to_nid,
                "from_state": item.get("start_state", "").strip(),
                "to_state": item.get("end_state", "").strip(),
                "phase": item.get("phase", ""),
                "type": item.get("type", ""),
                "rho": _edge_type_from_chain_type(item.get("type", "")),
                "action": item.get("action", ""),
                "key": item.get("key", ""),
                "edge_key": edge_key,
                "w_fit": float(arch_edge.get("w_fit", parent.get("fitness", 0.0)) or 0.0),
                "count": int(arch_edge.get("count", 1) or 1),
                "w_sparse": float(arch_edge.get("w_sparse", 1.0) or 1.0),
                "weak_count": int(arch_edge.get("weak_count", 0) or 0),
            })

    adj: dict[str, list[tuple[int, str]]] = {n["id"]: [] for n in nodes}
    for ei, edge in enumerate(edges):
        adj.setdefault(edge["from"], []).append((ei, edge["to"]))

    snapshot = {
        "generation": generation,
        "nodes": nodes,
        "edges": [{k: v for k, v in e.items() if k != "key"} for e in edges],
    }
    snapshot_path = os.path.join(POPULATIONS_DIR, f"ex{generation}_numbered_graph.json")
    _write_json(snapshot_path, snapshot)
    print(f"[快照] 编号图已保存 -> {snapshot_path}")

    return {"nodes": nodes, "edges": edges, "state_to_nid": state_to_nid, "adj": adj}


def _ask_llm_merge_nodes(
    numbered_graph: dict[str, Any],
    model_name: str,
    max_attempts: int = 3,
) -> list[list[int]]:
    """让 LLM 判断 AOE 图中哪些节点语义相同/相似，返回合并组别。"""
    node_list = [{"编号": i, "ID": n["id"], "状态": n["state"]} for i, n in enumerate(numbered_graph["nodes"])]
    edge_list = [{"ID": e["id"], "从": e["from"], "到": e["to"], "阶段": e["phase"], "类型": e["type"], "动作": e["action"]} for e in numbered_graph["edges"]]

    prompt = (
        "你是一个工作流图分析专家。下面是一个优化智能体的 AOE 工作流图。\n\n"
        "## 节点列表\n"
        f"{json.dumps(node_list, ensure_ascii=False, indent=2)}\n\n"
        "## 边列表\n"
        f"{json.dumps(edge_list, ensure_ascii=False, indent=2)}\n\n"
        "## 任务\n"
        "请分析上述节点中，哪些节点在语义上相同或高度相似，应该合并为一个节点。\n\n"
        "判断标准：\n"
        "1. 状态描述含义相同或几乎相同。\n"
        "2. 合并后不会破坏工作流的逻辑连续性。\n\n"
        "## 输出格式\n"
        "请**仅**输出一个 JSON 对象：\n"
        '{"merge_groups": [[0, 3], [1, 5, 7], ...]}\n\n'
        "每个子数组是需要合并的节点编号（对应节点列表中的\"编号\"字段）。\n"
        "不需要合并的节点不出现。每个节点最多出现在一个组中。"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是图分析与语义匹配专家。严格输出指定格式的JSON。"},
        {"role": "user", "content": prompt},
    ]
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"[LLM] 节点相似度判断，第 {attempt}/{max_attempts} 次")
        raw = query_llm(messages, model_name=model_name)
        json_text = _extract_json_text(raw)
        try:
            result = json.loads(json_text)
            groups = result.get("merge_groups", [])
            if not isinstance(groups, list):
                raise ValueError("merge_groups 必须是数组")
            for g in groups:
                if not isinstance(g, list) or not all(isinstance(x, int) for x in g):
                    raise ValueError(f"merge_groups 元素必须是整数数组: {g}")
            return groups
        except Exception as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"输出格式错误：{last_error}。请仅输出JSON对象。"})
    print(f"[警告] LLM 节点合并判断失败: {last_error}，将不进行合并")
    return []


def _apply_node_merges(
    numbered_graph: dict[str, Any],
    merge_groups: list[list[int]],
) -> dict[str, Any]:
    """应用 LLM 返回的合并组，生成合并后的图。"""
    orig_nodes = numbered_graph["nodes"]
    orig_edges = numbered_graph["edges"]

    idx_to_group: dict[int, int] = {}
    for gi, group in enumerate(merge_groups):
        for idx in group:
            if 0 <= idx < len(orig_nodes):
                idx_to_group[idx] = gi

    merged_nodes: list[dict[str, Any]] = []
    idx_to_mid: dict[int, str] = {}

    for gi, group in enumerate(merge_groups):
        mid = f"M{gi}"
        member_states = [orig_nodes[i]["state"] for i in group if 0 <= i < len(orig_nodes)]
        merged_nodes.append({
            "id": mid, "canonical_state": member_states[0] if member_states else "",
            "member_indices": group, "member_states": member_states,
        })
        for idx in group:
            if 0 <= idx < len(orig_nodes):
                idx_to_mid[idx] = mid

    for i, node in enumerate(orig_nodes):
        if i not in idx_to_group:
            mid = f"M{len(merged_nodes)}"
            merged_nodes.append({
                "id": mid, "canonical_state": node["state"],
                "member_indices": [i], "member_states": [node["state"]],
            })
            idx_to_mid[i] = mid

    nid_to_mid: dict[str, str] = {}
    for i, node in enumerate(orig_nodes):
        nid_to_mid[node["id"]] = idx_to_mid.get(i, f"M{i}")

    merged_edges: list[dict[str, Any]] = []
    edge_by_dedup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for edge in orig_edges:
        new_from = nid_to_mid.get(edge["from"], edge["from"])
        new_to = nid_to_mid.get(edge["to"], edge["to"])
        if new_from == new_to:
            continue
        dedup_key = (new_from, new_to, edge.get("phase", ""), edge.get("type", ""), edge.get("action", ""))
        if dedup_key in edge_by_dedup:
            kept = edge_by_dedup[dedup_key]
            old_count = int(kept.get("count", 0) or 0)
            add_count = int(edge.get("count", 1) or 1)
            total_count = old_count + add_count
            kept["w_fit"] = (
                float(kept.get("w_fit", 0.0) or 0.0) * old_count
                + float(edge.get("w_fit", 0.0) or 0.0) * add_count
            ) / max(total_count, 1)
            kept["count"] = total_count
            kept["w_sparse"] = 1.0 / (math.log(2.0 + total_count) + SPARSITY_EPSILON)
            continue
        merged_edge = {**edge, "from": new_from, "to": new_to}
        edge_by_dedup[dedup_key] = merged_edge
        merged_edges.append(merged_edge)

    for ei, edge in enumerate(merged_edges):
        edge["id"] = f"E{ei}"

    merged_adj: dict[str, list[tuple[int, str]]] = {m["id"]: [] for m in merged_nodes}
    for ei, edge in enumerate(merged_edges):
        merged_adj.setdefault(edge["from"], []).append((ei, edge["to"]))

    print(f"[合并] 节点 {len(orig_nodes)}->{len(merged_nodes)}, 边 {len(orig_edges)}->{len(merged_edges)}")
    return {
        "merged_nodes": merged_nodes, "merged_edges": merged_edges,
        "adj": merged_adj, "idx_to_mid": idx_to_mid, "nid_to_mid": nid_to_mid,
    }


def _collect_existing_path_signatures(
    population_chain: list[dict[str, Any]],
    nid_to_mid: dict[str, str],
    state_to_nid: dict[str, str],
) -> set[tuple[str, ...]]:
    """收集所有父代链在合并图中的路径签名。"""
    signatures: set[tuple[str, ...]] = set()

    for parent in population_chain:
        chain = parent.get("chain", [])
        if not chain:
            continue
        # 签名1：原始状态名序列
        states: list[str] = []
        for item in chain:
            if not isinstance(item, dict):
                continue
            start = item.get("start_state", "").strip()
            end = item.get("end_state", "").strip()
            if not states:
                states.append(start)
            states.append(end)
        if len(states) >= 2:
            signatures.add(tuple(states))

        # 签名2：合并后节点ID序列
        mids: list[str] = []
        for item in chain:
            if not isinstance(item, dict):
                continue
            start_nid = state_to_nid.get(item.get("start_state", "").strip(), "")
            end_nid = state_to_nid.get(item.get("end_state", "").strip(), "")
            start_mid = nid_to_mid.get(start_nid, start_nid)
            end_mid = nid_to_mid.get(end_nid, end_nid)
            if not mids:
                mids.append(start_mid)
            mids.append(end_mid)
        if len(mids) >= 2:
            signatures.add(tuple(mids))

    return signatures


def _attach_edge_selection_scores(
    edges: list[dict[str, Any]],
    gamma: float,
) -> None:
    fit_values = [float(edge.get("w_fit", 0.0) or 0.0) for edge in edges]
    sparse_values = [float(edge.get("w_sparse", 0.0) or 0.0) for edge in edges]
    fit_norm = _minmax_normalize(fit_values)
    sparse_norm = _minmax_normalize(sparse_values)
    for edge, normalized_fit, normalized_sparse in zip(edges, fit_norm, sparse_norm):
        edge["phi"] = gamma * normalized_fit + (1.0 - gamma) * normalized_sparse


def _weighted_neighbor_order(
    neighbors: list[tuple[int, str]],
    edges: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    weighted: list[tuple[float, tuple[int, str]]] = []
    for neighbor in neighbors:
        edge_index, _ = neighbor
        weight = max(float(edges[edge_index].get("phi", 0.0) or 0.0), 1e-6)
        weighted.append((random.random() ** (1.0 / weight), neighbor))
    weighted.sort(key=lambda item: item[0], reverse=True)
    return [neighbor for _, neighbor in weighted]


def _find_new_paths(
    merged_graph: dict[str, Any],
    existing_signatures: set[tuple[str, ...]],
    max_paths: int,
    gamma: float = DEFAULT_EXPLORATION_GAMMA,
    max_search_steps: int = 50000,
) -> list[list[str]]:
    """在合并后的 AOE 图上用 DFS 搜索与已有路径不同的新路径。"""
    adj = merged_graph["adj"]
    edges = merged_graph["merged_edges"]
    merged_nodes = merged_graph["merged_nodes"]
    _attach_edge_selection_scores(edges, gamma)

    in_degree: dict[str, int] = {n["id"]: 0 for n in merged_nodes}
    for edge in edges:
        in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1

    start_nodes = [nid for nid, deg in in_degree.items() if deg == 0]
    end_nodes = [n["id"] for n in merged_nodes if len(adj.get(n["id"], [])) == 0]
    if not start_nodes:
        start_nodes = [merged_nodes[0]["id"]]
    if not end_nodes:
        end_nodes = [merged_nodes[-1]["id"]]

    print(f"[图搜索] 起始={start_nodes}, 终止={end_nodes}")

    new_paths: list[list[str]] = []
    steps = 0

    for start_nid in start_nodes:
        if len(new_paths) >= max_paths:
            break
        stack: list[tuple[str, list[str], list[str]]] = [(start_nid, [], [start_nid])]
        while stack and len(new_paths) < max_paths and steps < max_search_steps:
            steps += 1
            cur_node, edge_path, node_path = stack.pop()
            if cur_node in end_nodes and edge_path:
                sig = tuple(node_path)
                if sig not in existing_signatures:
                    new_paths.append(list(edge_path))
                    existing_signatures.add(sig)
                    print(f"[图搜索] 发现新路径 #{len(new_paths)}: {edge_path}")
                continue
            neighbors = _weighted_neighbor_order(list(adj.get(cur_node, [])), edges)
            for ei, next_node in reversed(neighbors):
                if next_node not in node_path:
                    stack.append((next_node, edge_path + [edges[ei]["id"]], node_path + [next_node]))

    print(f"[图搜索] 共找到 {len(new_paths)} 条新路径（步数={steps}）")
    return new_paths


def _path_to_chain_items(
    edge_ids: list[str],
    merged_graph: dict[str, Any],
) -> list[dict[str, str]]:
    """将边ID序列转换为 agent_chain.json 格式。"""
    edge_map: dict[str, dict[str, Any]] = {e["id"]: e for e in merged_graph["merged_edges"]}
    node_map: dict[str, dict[str, Any]] = {n["id"]: n for n in merged_graph["merged_nodes"]}
    items: list[dict[str, str]] = []
    for eid in edge_ids:
        edge = edge_map.get(eid)
        if edge is None:
            continue
        from_node = node_map.get(edge.get("from", ""), {})
        to_node = node_map.get(edge.get("to", ""), {})
        items.append({
            "phase": edge.get("phase", ""),
            "type": edge.get("type", ""),
            "action": edge.get("action", ""),
            "start_state": edge.get("from_state") or from_node.get("canonical_state", edge.get("from", "")),
            "end_state": edge.get("to_state") or to_node.get("canonical_state", edge.get("to", "")),
            "key": edge.get("key", ""),
        })
    return items


def _generate_crossover_chains_batch(
    population_chain: list[dict[str, Any]],
    generation: int,
    model_name: str,
    crossover_count: int,
    architecture_graph: Optional[dict[str, Any]] = None,
    exploration_gamma: float = DEFAULT_EXPLORATION_GAMMA,
) -> list[tuple[list[dict[str, str]], str]]:
    """基于图搜索的交叉：编号图 -> LLM判断合并 -> 图搜索找新路径。"""
    print("[交叉-图搜索] Step 1: 构建带编号的AOE图")
    numbered_graph = _build_numbered_graph(population_chain, generation, architecture_graph)

    print("[交叉-图搜索] Step 2: LLM判断节点相似度")
    merge_groups = _ask_llm_merge_nodes(numbered_graph, model_name)

    print("[交叉-图搜索] Step 3: 应用节点合并")
    merged_graph = _apply_node_merges(numbered_graph, merge_groups)

    print("[交叉-图搜索] Step 4: 收集已有路径签名")
    existing_signatures = _collect_existing_path_signatures(
        population_chain, merged_graph["nid_to_mid"], numbered_graph["state_to_nid"])
    print(f"[交叉-图搜索] 已有路径签名数: {len(existing_signatures)}")

    print(f"[交叉-图搜索] Step 5: 图搜索寻找 {crossover_count} 条新路径 | gamma={exploration_gamma}")
    new_paths = _find_new_paths(merged_graph, existing_signatures, crossover_count, gamma=exploration_gamma)

    results: list[tuple[list[dict[str, str]], str]] = []
    for i, edge_ids in enumerate(new_paths):
        chain_items = _path_to_chain_items(edge_ids, merged_graph)
        results.append((chain_items, json.dumps(chain_items, ensure_ascii=False, indent=2)))
        print(f"[交叉-图搜索] 后代 #{i+1}: {len(chain_items)} 步")

    shortage = crossover_count - len(results)
    if shortage > 0:
        print(f"[交叉-图搜索] 图搜索仅找到 {len(results)} 条，LLM补充 {shortage} 条")
        for i in range(shortage):
            chain_items, chain_raw = _generate_crossover_chain_llm_fallback(
                population_chain=population_chain, model_name=model_name,
                offspring_index=len(results) + i + 1, total_offspring=crossover_count)
            results.append((chain_items, chain_raw))

    return results


def _generate_crossover_chain_llm_fallback(
    population_chain: list[dict[str, Any]],
    model_name: str,
    offspring_index: int,
    total_offspring: int,
) -> tuple[list[dict[str, str]], str]:
    """LLM 兜底交叉：当图搜索找不到足够新路径时使用。"""
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
    for attempt in range(1, 4):
        print(f"[LLM] 交叉兜底，第 {attempt}/3 次")
        last_response = query_llm(messages, model_name=model_name)
        json_text = _extract_json_text(last_response)
        try:
            parsed = json.loads(json_text)
            _validate_chain_items(parsed)
            return parsed, last_response
        except Exception as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": last_response})
            messages.append({"role": "user", "content": f"输出不满足要求。错误：{last_error}。请重新输出。"})
    raise RuntimeError(f"交叉兜底生成 chain 失败: {last_error or '未知错误'}")

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
    architecture_rate: float = DEFAULT_ARCH_LEARNING_RATE,
    exploration_gamma: float = DEFAULT_EXPLORATION_GAMMA,
    pruning_threshold: float = DEFAULT_PRUNING_THRESHOLD,
    forgetting_horizon: int = DEFAULT_FORGETTING_HORIZON,
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
    if not (0 < architecture_rate <= 1):
        raise ValueError("architecture_rate 必须在 (0, 1] 范围内")
    if not (0 <= exploration_gamma <= 1):
        raise ValueError("exploration_gamma 必须在 [0, 1] 范围内")
    if not (0 <= pruning_threshold <= 1):
        raise ValueError("pruning_threshold 必须在 [0, 1] 范围内")
    if forgetting_horizon <= 0:
        raise ValueError("forgetting_horizon 必须为正整数")

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
    architecture_graph = _ensure_architecture_graph(population_chain, src_generation)

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

    # 1) Crossover: 图搜索交叉 — 编号图 → LLM合并 → 图搜索找新路径 → chain → code
    print(f"[阶段] 交叉 开始 | 数量={crossover_count}")
    if crossover_count > 0:
        crossover_results = _generate_crossover_chains_batch(
            population_chain=population_chain,
            generation=src_generation,
            model_name=model_name,
            crossover_count=crossover_count,
            architecture_graph=architecture_graph,
            exploration_gamma=exploration_gamma,
        )
        for cross_i, (chain_items, chain_raw) in enumerate(crossover_results, start=1):
            _print_progress("交叉", cross_i, crossover_count, f"目标=ex{dst_generation}_p{offspring_index}")
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

    evaluated_offspring = _rank_parents_from_existing_eval(_build_population_chain(dst_generation))
    _update_architecture_graph(
        current_graph=architecture_graph,
        evaluated_population=evaluated_offspring,
        dst_generation=dst_generation,
        alpha=architecture_rate,
        tau=pruning_threshold,
        sigma=forgetting_horizon,
    )

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
    parser.add_argument("--arch-rate", type=float, default=DEFAULT_ARCH_LEARNING_RATE, help="架构图边适应度学习率 alpha，默认 0.5")
    parser.add_argument("--explore-gamma", type=float, default=DEFAULT_EXPLORATION_GAMMA, help="路径采样 exploitation/exploration 权衡 gamma，默认 0.5")
    parser.add_argument("--prune-threshold", type=float, default=DEFAULT_PRUNING_THRESHOLD, help="点边剪枝阈值 tau，默认 0.1")
    parser.add_argument("--forget-horizon", type=int, default=DEFAULT_FORGETTING_HORIZON, help="连续低分遗忘窗口 sigma，默认 3")
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
            architecture_rate=args.arch_rate,
            exploration_gamma=args.explore_gamma,
            pruning_threshold=args.prune_threshold,
            forgetting_horizon=args.forget_horizon,
            model_name=args.model,
            seed=args.seed,
        )
        print(f"========== 完成进化第 {generation} 轮 ==========")


if __name__ == "__main__":
    main()

