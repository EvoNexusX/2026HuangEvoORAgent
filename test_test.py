import importlib.util
import shutil
import json
import os
import tempfile
from pathlib import Path
from types import ModuleType


WORKSPACE_DIR = Path(__file__).resolve().parent
AGENT2_PATH = WORKSPACE_DIR / "2_agent.py"
EA2_PATH = WORKSPACE_DIR / "2_ea_agent.py"
MY_AGENT_PATH = WORKSPACE_DIR / "My_agent.py"


def load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_my_agent_has_required_lines(my_agent_path: Path) -> None:
    code = my_agent_path.read_text(encoding="utf-8")
    required_1 = 'print(f"[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}")'
    required_2 = 'print(f"[Total fails {len(error_datas)}] error datas: {error_datas}")'

    if required_1 not in code or required_2 not in code:
        raise AssertionError("My_agent.py 不包含约定的总评输出格式")


def prepare_real_or_agent(pop_dir: Path) -> None:
    # 直接使用 My_agent.py 作为真实 or_agent.py 进行评测提取测试。
    shutil.copy2(MY_AGENT_PATH, pop_dir / "or_agent.py")

    # My_agent.py 依赖 new_utils.py；为了在临时目录可运行，一并复制。
    new_utils_src = WORKSPACE_DIR / "new_utils.py"
    if not new_utils_src.exists():
        raise FileNotFoundError(f"缺少依赖文件: {new_utils_src}")
    shutil.copy2(new_utils_src, pop_dir / "new_utils.py")


def main() -> None:
    ensure_my_agent_has_required_lines(MY_AGENT_PATH)

    two_agent = load_module("two_agent", AGENT2_PATH)
    two_ea_agent = load_module("two_ea_agent", EA2_PATH)

    with tempfile.TemporaryDirectory(prefix="eval_extract_demo_") as tmp_dir:
        pop_dir = Path(tmp_dir)
        prepare_real_or_agent(pop_dir)

        # 使用真实 My_agent.py 运行，给足时间，避免评测过程被短超时打断。
        result_2 = two_agent._run_agent_eval(str(pop_dir), timeout_sec=3600)
        result_ea = two_ea_agent._run_agent_eval(str(pop_dir), timeout_sec=3600)

        two_agent._persist_eval_result(str(pop_dir), result_2)

        print("=== 2_agent.py extraction result ===")
        print(json.dumps({
            "passed": result_2.get("passed"),
            "format_ok": result_2.get("format_ok"),
            "metrics": result_2.get("metrics"),
        }, ensure_ascii=False, indent=2))

        print("=== 2_ea_agent.py extraction result ===")
        print(json.dumps({
            "passed": result_ea.get("passed"),
            "format_ok": result_ea.get("format_ok"),
            "metrics": result_ea.get("metrics"),
        }, ensure_ascii=False, indent=2))

        if result_2.get("metrics") != result_ea.get("metrics"):
            raise AssertionError("两个脚本提取结果不一致")

        saved_eval_path = pop_dir / "eval_result.json"
        print(f"saved eval json: {saved_eval_path}")
        if saved_eval_path.exists():
            print(saved_eval_path.read_text(encoding="utf-8"))

    print("[OK] 已直接运行 My_agent.py 完成提取测试。")


if __name__ == "__main__":
    main()
