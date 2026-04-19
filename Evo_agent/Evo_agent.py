from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

try:
    from new_utils import query_llm
except Exception:
    query_llm = None


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
POPULATIONS_DIR = os.path.join(WORKSPACE_DIR, "populations")
CURRENT_BEST_AGENT_FILE = os.path.join(POPULATIONS_DIR, "current_best_agent.py")
LOG_FILE = os.path.join(POPULATIONS_DIR, "evo_agent_log.jsonl")
DEFAULT_LLM_MODEL = ""
DEFAULT_SEED_AGENT_PATH = os.path.join(WORKSPACE_DIR, "new_agent.py")
TARGET_SYMBOL = "or_llm_agent"


AGENT_CROSSOVER_TEMPLATE = """You are an EvoAgent genetic operator.

Task: evolve a Python OR agent, not a prompt.

Step 1 (Crossover): Analyze both parent agents, identify complementary strengths, and combine them into one improved agent.
Step 2 (Mutation): Make a small local change to the crossed agent while preserving correctness and modularity.

Parent Agent A:
{parent1}

Parent Agent B:
{parent2}

Current Best Agent:
{best_agent}

Rules:
1) Keep the result as a complete Python agent script.
2) Preserve the OR problem solving workflow: query_llm -> code generation -> execute -> repair -> evaluate.
3) Keep code modular and readable.
4) Output only one tag pair: <agent> ... </agent>
"""


AGENT_MUTATION_TEMPLATE = """You are an EvoAgent mutation operator.

Task: compare the crossed agent with the original parent agent and produce a slightly improved agent variant.

Crossover Agent:
{crossover_agent}

Original Parent Agent:
{parent_agent}

Rules:
1) Keep the output as a valid Python agent script.
2) Apply a small local revision, not a full rewrite.
3) Preserve the OR agent workflow and retry logic.
4) Output only one tag pair: <agent> ... </agent>
"""


FUNCTION_CROSSOVER_TEMPLATE = """You are evolving one Python function for an OR agent.

Target symbol: #sym:{target_symbol}

Task:
1) Analyze both parent function variants and identify deficiencies from historical outputs (robustness, retry logic, infeasible handling, clarity).
2) Perform crossover to produce one improved function.
3) Keep function input/output behavior unchanged.
4) Keep function name unchanged as `{target_symbol}`.

Parent Function A:
{parent1}

Parent Function B:
{parent2}

Current Best Function:
{best_function}

Rules:
1) Output ONLY one tag pair: <function> ... </function>
2) The output must be a complete Python function definition.
3) Keep arguments and return values behavior compatible with the original symbol.
"""


FUNCTION_MUTATION_TEMPLATE = """You are mutating one Python function for an OR agent.

Target symbol: #sym:{target_symbol}

Task:
1) Compare the crossover function with the original parent function.
2) Apply a local mutation to improve reliability/readability.
3) Keep I/O behavior unchanged.
4) Keep function name unchanged as `{target_symbol}`.

Crossover Function:
{crossover_function}

Original Parent Function:
{parent_function}

Rules:
1) Output ONLY one tag pair: <function> ... </function>
2) Output must be a valid complete Python function.
"""


MetricFn = Callable[[Any], float]
EvaluatorFn = Callable[[str, Iterable[Any], Optional[MetricFn], random.Random], float]


@dataclass
class AgentCandidate:
    code: str
    score: float
    generation: int
    origin: str
    parent_a: Optional[str] = None
    parent_b: Optional[str] = None
    parent_a_ref: Optional[str] = None
    parent_b_ref: Optional[str] = None
    uid: Optional[str] = None


def _load_seed_agent_source() -> str:
    """Load the current agent blueprint from new_agent.py as the seed agent."""
    try:
        with open(DEFAULT_SEED_AGENT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        # Fallback template keeps the file self-contained if the seed file is missing.
        return (
            "import argparse\n"
            "import copy\n\n"
            "import new_utils\n"
            "from new_utils import query_llm, save_generated_code, load_dataset, extract_and_execute_python_code, eval_model_result, is_number_string\n\n"
            "DEFAULT_MODEL_NAME = ''\n\n"
            "def generate_or_code_solver(messages_bak, model_name, max_attempts):\n"
            "    messages = copy.deepcopy(messages_bak)\n"
            "    gurobi_code = query_llm(messages, model_name)\n"
            "    save_generated_code(gurobi_code, prefix='agent')\n"
            "    text = f'{gurobi_code}'\n"
            "    attempt = 0\n"
            "    while attempt < max_attempts:\n"
            "        success, error_msg = extract_and_execute_python_code(text)\n"
            "        if success:\n"
            "            messages_bak.append({'role': 'assistant', 'content': gurobi_code})\n"
            "            return True, error_msg, messages_bak\n"
            "        messages.append({'role': 'assistant', 'content': gurobi_code})\n"
            "        messages.append({'role': 'user', 'content': f'代码执行时发生错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整可执行代码。'})\n"
            "        gurobi_code = query_llm(messages, model_name)\n"
            "        save_generated_code(gurobi_code, prefix='agent_fix')\n"
            "        text = f'{gurobi_code}'\n"
            "        attempt += 1\n"
            "    messages_bak.append({'role': 'assistant', 'content': gurobi_code})\n"
            "    return False, None, messages_bak\n"
        )


class EvoAgentGenerator:
    """Discrete multi-agent evolution framework for OR-solving agent code."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        population_size: int = 10,
        offspring_per_generation: int = 10,
        max_generations: int = 8,
        seed: int = 42,
        early_stop_patience: int = 2,
        early_stop_threshold: float = 0.003,
        evaluator: Optional[EvaluatorFn] = None,
    ) -> None:
        self.model_name = model_name
        self.population_size = population_size
        self.offspring_per_generation = offspring_per_generation
        self.max_generations = max_generations
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.rng = random.Random(seed)
        self.evaluator = evaluator or self._random_evaluator
        self.base_agent_source = _load_seed_agent_source()
        self.base_symbol_function = self._extract_symbol_function(self.base_agent_source)
        self.base_symbol_signature = self.base_symbol_function.splitlines()[0].strip()

        self.population: list[AgentCandidate] = []
        self.api_calls = 0
        self.best_global: Optional[AgentCandidate] = None

        os.makedirs(POPULATIONS_DIR, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(POPULATIONS_DIR, "evo_agent.log"), encoding="utf-8"),
            ],
        )

    def _append_json_log(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _extract_agent_tag(self, text: str) -> Optional[str]:
        match = re.search(r"<agent>([\s\S]*?)</agent>", text)
        if match:
            agent = match.group(1).strip()
            return agent if agent else None

        code_match = re.search(r"```python\s*([\s\S]*?)```", text)
        if code_match:
            agent = code_match.group(1).strip()
            return agent if agent else None

        stripped = text.strip()
        return stripped if stripped else None

    def _extract_function_tag(self, text: str) -> Optional[str]:
        match = re.search(r"<function>([\s\S]*?)</function>", text)
        if match:
            candidate = match.group(1).strip()
            return candidate if candidate else None
        code_match = re.search(r"```python\s*([\s\S]*?)```", text)
        if code_match:
            candidate = code_match.group(1).strip()
            return candidate if candidate else None
        stripped = text.strip()
        return stripped if stripped else None

    def _extract_symbol_function(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"Cannot parse seed agent source: {exc}") from exc

        lines = code.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == TARGET_SYMBOL:
                start = node.lineno
                end = getattr(node, "end_lineno", None)
                if end is None:
                    raise ValueError("Python AST does not provide end_lineno for target symbol.")
                return "\n".join(lines[start - 1 : end]).rstrip()
        raise ValueError(f"Cannot find target symbol function: {TARGET_SYMBOL}")

    def _extract_evolvable_function(self, code: str) -> str:
        """Only evolve #sym:or_llm_agent."""
        return self._extract_symbol_function(code)

    def _normalize_evolved_function(self, function_code: str) -> str:
        function_code = function_code.strip()
        # Keep function signature aligned to the original symbol to preserve I/O contract.
        function_code = re.sub(
            r"^def\s+\w+\s*\([^\n]*\):",
            self.base_symbol_signature,
            function_code,
            count=1,
            flags=re.MULTILINE,
        )
        return function_code + "\n"

    def _compose_agent_with_function(self, evolved_function_code: str) -> str:
        base = self.base_agent_source
        original = self.base_symbol_function
        evolved = self._normalize_evolved_function(evolved_function_code)

        # Keep other components unchanged, replace only #sym:or_llm_agent block.
        return base.replace(original, evolved)

    def _llm_generate(self, user_content: str) -> Optional[str]:
        if query_llm is None:
            return None

        model_name = self.model_name
        if model_name.lower().startswith("") or model_name.lower().startswith(""):
            model_name = DEFAULT_LLM_MODEL

        messages = [
            {"role": "system", "content": "You are an expert multi-agent evolution assistant."},
            {"role": "user", "content": user_content},
        ]
        self.api_calls += 1
        try:
            raw = query_llm(messages, model_name)
        except Exception as exc:
            logging.warning("LLM call failed (%s), fallback to local operator.", exc)
            return None
        return self._extract_agent_tag(raw)

    def _llm_generate_function(self, user_content: str) -> Optional[str]:
        if query_llm is None:
            return None

        model_name = self.model_name
        if model_name.lower().startswith("") or model_name.lower().startswith(""):
            model_name = DEFAULT_LLM_MODEL

        messages = [
            {"role": "system", "content": "You are an expert Python function evolution assistant."},
            {"role": "user", "content": user_content},
        ]
        self.api_calls += 1
        try:
            raw = query_llm(messages, model_name)
        except Exception as exc:
            logging.warning("LLM function call failed (%s), fallback to local operator.", exc)
            return None
        return self._extract_function_tag(raw)

    def _local_mutation(self, text: str) -> str:
        replacements = {
            "two-stage": "multi-stage",
            "agent mode": "adaptive agent mode",
            "repair": "debug-fix",
            "prompt": "instruction",
            "evaluation": "assessment",
            "robust": "resilient",
            "concise": "compact",
        }

        mutated = text
        for old, new in replacements.items():
            if old in mutated and self.rng.random() < 0.35:
                mutated = mutated.replace(old, new, 1)

        if self.rng.random() < 0.4:
            mutated += (
                "\n# Evolution note: keep modular solve, repair, and evaluation steps explicit."
            )

        return mutated.strip() + "\n"

    def _local_crossover(self, parent1: str, parent2: str) -> str:
        p1_blocks = [block.strip() for block in re.split(r"\n\s*\n", parent1) if block.strip()]
        p2_blocks = [block.strip() for block in re.split(r"\n\s*\n", parent2) if block.strip()]

        take1 = max(1, len(p1_blocks) // 2)
        take2 = max(1, len(p2_blocks) // 2)
        merged = p1_blocks[:take1] + p2_blocks[-take2:]
        return self._local_mutation("\n\n".join(merged))

    def _write_population_snapshot(self, generation: int, candidates: list[AgentCandidate]) -> None:
        for idx, candidate in enumerate(candidates, start=1):
            uid = candidate.uid or f"ex{generation}_p{idx}"
            folder = os.path.join(POPULATIONS_DIR, uid)
            os.makedirs(folder, exist_ok=True)

            with open(os.path.join(folder, "agent.py"), "w", encoding="utf-8") as f:
                f.write(candidate.code.strip() + "\n")

            meta = {
                "strategy": candidate.origin,
                "generation": generation,
                "population_index": idx,
                "uid": uid,
                "score": candidate.score,
                "parent_a_ref": candidate.parent_a_ref,
                "parent_b_ref": candidate.parent_b_ref,
                "parent_a": candidate.parent_a,
                "parent_b": candidate.parent_b,
            }
            with open(os.path.join(folder, "ea_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def _assign_population_uids(self, generation: int, candidates: list[AgentCandidate]) -> None:
        for idx, candidate in enumerate(candidates, start=1):
            candidate.uid = f"ex{generation}_p{idx}"

    def _save_current_best(self, code: str) -> None:
        with open(CURRENT_BEST_AGENT_FILE, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n")

    def crossover(self, parent_a: str, parent_b: str, best_agent: str) -> str:
        parent1_function = self._extract_evolvable_function(parent_a)
        parent2_function = self._extract_evolvable_function(parent_b)
        best_function = self._extract_evolvable_function(best_agent)
        llm_prompt = FUNCTION_CROSSOVER_TEMPLATE.format(
            target_symbol=TARGET_SYMBOL,
            parent1=parent1_function,
            parent2=parent2_function,
            best_function=best_function,
        )
        llm_output = self._llm_generate_function(llm_prompt)
        if llm_output:
            return llm_output
        return self._local_crossover(parent1_function, parent2_function)

    def mutate(self, crossover_agent: str, parent_agent: str) -> str:
        parent_function = self._extract_evolvable_function(parent_agent)
        llm_prompt = FUNCTION_MUTATION_TEMPLATE.format(
            target_symbol=TARGET_SYMBOL,
            crossover_function=crossover_agent,
            parent_function=parent_function,
        )
        llm_output = self._llm_generate_function(llm_prompt)
        if llm_output:
            return llm_output
        return self._local_mutation(crossover_agent)

    def _random_evaluator(
        self,
        agent_code: str,
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn],
        rng: random.Random,
    ) -> float:
        """Default reproducible scorer used when no custom evaluator is provided."""
        del dataloader, metric_fn

        required_tokens = [
            "def or_llm_agent",
            "def generate_or_code_solver",
            "query_llm",
            "extract_and_execute_python_code",
            "eval_model_result",
            "save_generated_code",
            "run_eval",
            "parse_args",
        ]
        coverage_bonus = sum(1 for token in required_tokens if token in agent_code) * 2.5
        length_bonus = min(len(agent_code) / 15000.0, 0.15)
        randomness = rng.uniform(60.0, 90.0)
        return round(randomness * (1.0 + length_bonus) + coverage_bonus, 4)

    def quality_check(
        self,
        agent_code: str,
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn],
    ) -> float:
        if self.evaluator is not None:
            try:
                return float(self.evaluator(agent_code, dataloader, metric_fn, self.rng))
            except Exception as exc:
                logging.warning("Custom evaluator failed (%s), fallback to heuristic score.", exc)

        return self._random_evaluator(agent_code, dataloader, metric_fn, self.rng)

    def result_update(
        self,
        current_population: list[AgentCandidate],
        offspring: list[AgentCandidate],
        generation: int,
    ) -> list[AgentCandidate]:
        merged = current_population + offspring
        merged.sort(key=lambda item: item.score, reverse=True)
        next_population = merged[: self.population_size]

        self.population = next_population
        self._assign_population_uids(generation, self.population)
        if self.best_global is None or next_population[0].score > self.best_global.score:
            self.best_global = next_population[0]

        self._save_current_best(self.best_global.code)
        self._write_population_snapshot(generation, next_population)
        return next_population

    def _default_seed_agents(self) -> list[str]:
        # Start from unchanged base agent and evolve only the target symbol function.
        base_func = self.base_symbol_function
        variants = [
            self._compose_agent_with_function(base_func),
            self._compose_agent_with_function(self._local_mutation(base_func)),
            self._compose_agent_with_function(self._local_mutation(base_func)),
            self._compose_agent_with_function(self._local_mutation(base_func)),
        ]

        unique: list[str] = []
        for item in variants:
            cleaned = item.strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def initialize_population(
        self,
        manual_agents: list[str],
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn],
    ) -> None:
        print("[EvoAgent] Initializing population...")
        agents: list[str] = []
        for agent_code in manual_agents:
            agent_code = agent_code.strip()
            if agent_code and agent_code not in agents:
                agents.append(agent_code)

        while len(agents) < self.population_size:
            best_seed = agents[0] if agents else self._default_seed_agents()[0]
            base_function = self._extract_evolvable_function(best_seed)
            llm_variant_function = self._llm_generate_function(
                f"Evolve only function #sym:{TARGET_SYMBOL}. Output <function>...</function>.\n\n<function>{base_function}</function>"
            )
            candidate_function = llm_variant_function if llm_variant_function else self._local_mutation(base_function)
            candidate_agent = self._compose_agent_with_function(candidate_function)
            if candidate_agent not in agents:
                agents.append(candidate_agent)

        self.population = []
        for idx, agent_code in enumerate(agents[: self.population_size], start=1):
            score = self.quality_check(agent_code, dataloader, metric_fn)
            candidate = AgentCandidate(code=agent_code, score=score, generation=1, origin="init")
            self.population.append(candidate)

        self.population.sort(key=lambda item: item.score, reverse=True)
        self._assign_population_uids(1, self.population)
        self.best_global = self.population[0]
        self._save_current_best(self.best_global.code)
        self._write_population_snapshot(1, self.population)
        print(f"[EvoAgent] Generation 1 ready: {len(self.population)} individuals saved.")

        avg_score = sum(item.score for item in self.population) / len(self.population)
        logging.info("Init done | best=%.4f avg=%.4f api_calls=%d", self.best_global.score, avg_score, self.api_calls)
        self._append_json_log(
            {
                "event": "init",
                "best": asdict(self.best_global),
                "avg_score": avg_score,
                "api_calls": self.api_calls,
            }
        )

    def run(
        self,
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn] = None,
        manual_agents: Optional[list[str]] = None,
    ) -> AgentCandidate:
        print("[EvoAgent] Start evolution run")
        if not self.population:
            seeds = manual_agents or self._default_seed_agents()
            self.initialize_population(seeds, dataloader, metric_fn)

        prev_avg = sum(item.score for item in self.population) / len(self.population)
        no_improve_rounds = 0

        for generation in range(2, self.max_generations + 2):
            print(f"[EvoAgent] Generation {generation}: crossover + mutation + selection")
            offspring: list[AgentCandidate] = []

            for _ in range(1, self.offspring_per_generation + 1):
                parent_a, parent_b = self._select_two_parents()
                best_agent = self.population[0].code
                crossed_function = self.crossover(parent_a.code, parent_b.code, best_agent)
                mutated_function = self.mutate(crossed_function, parent_a.code)
                child_agent_code = self._compose_agent_with_function(mutated_function)
                child_score = self.quality_check(child_agent_code, dataloader, metric_fn)

                offspring.append(
                    AgentCandidate(
                        code=child_agent_code,
                        score=child_score,
                        generation=generation,
                        origin="ga",
                        parent_a=parent_a.code,
                        parent_b=parent_b.code,
                        parent_a_ref=parent_a.uid,
                        parent_b_ref=parent_b.uid,
                    )
                )

            self.result_update(self.population, offspring, generation)
            print(f"[EvoAgent] Generation {generation} completed, top score: {self.population[0].score:.4f}")

            best_now = self.population[0]
            avg_now = sum(item.score for item in self.population) / len(self.population)
            if self.best_global is None or best_now.score > self.best_global.score:
                self.best_global = best_now
                self._save_current_best(self.best_global.code)

            improve_ratio = (avg_now - prev_avg) / max(prev_avg, 1e-9)
            if improve_ratio < self.early_stop_threshold:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0

            logging.info(
                "Gen %d | best=%.4f avg=%.4f improve=%.4f%% api_calls=%d",
                generation,
                best_now.score,
                avg_now,
                improve_ratio * 100.0,
                self.api_calls,
            )
            self._append_json_log(
                {
                    "event": "generation_end",
                    "generation": generation,
                    "best": asdict(best_now),
                    "avg_score": avg_now,
                    "improve_ratio": improve_ratio,
                    "api_calls": self.api_calls,
                }
            )

            prev_avg = avg_now
            if no_improve_rounds >= self.early_stop_patience:
                logging.info("Early stop at generation %d", generation)
                self._append_json_log({"event": "early_stop", "generation": generation})
                break

        if self.best_global is None:
            raise RuntimeError("Evolution finished without any candidate.")

        final_path = os.path.join(POPULATIONS_DIR, "best_agent.py")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(self.best_global.code.strip() + "\n")
        print(f"[EvoAgent] Evolution finished, global best saved: {final_path}")

        logging.info("Global best score=%.4f saved=%s", self.best_global.score, final_path)
        self._append_json_log({"event": "finished", "best": asdict(self.best_global), "api_calls": self.api_calls})
        return self.best_global

    def _select_two_parents(self) -> tuple[AgentCandidate, AgentCandidate]:
        weights = [max(item.score, 1e-6) for item in self.population]
        parent_a = self.rng.choices(self.population, weights=weights, k=1)[0]
        parent_b = self.rng.choices(self.population, weights=weights, k=1)[0]
        return parent_a, parent_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoAgent discrete multi-agent optimization")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model for agent evolution")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size N")
    parser.add_argument("--offspring", type=int, default=10, help="New agents per generation")
    parser.add_argument("--generations", type=int, default=8, help="Max generation count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stop-threshold", type=float, default=0.003, help="Avg improvement threshold")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="Consecutive rounds for early stop")
    return parser.parse_args()


def main(
    population_size: Optional[int] = 10,
    training_generations: Optional[int] = 8,
) -> None:
    cli_args = parse_args()

    resolved_population_size = population_size if population_size is not None else cli_args.pop_size
    resolved_generations = training_generations if training_generations is not None else cli_args.generations

    # Placeholder dataset, kept only to preserve the same decoupled interface style as Evo_prompt.
    dataloader = [
        {"id": 1, "question": "demo-1"},
        {"id": 2, "question": "demo-2"},
    ]

    engine = EvoAgentGenerator(
        model_name=cli_args.model,
        population_size=resolved_population_size,
        offspring_per_generation=cli_args.offspring,
        max_generations=resolved_generations,
        seed=cli_args.seed,
        early_stop_patience=cli_args.early_stop_patience,
        early_stop_threshold=cli_args.early_stop_threshold,
    )
    best = engine.run(dataloader=dataloader, metric_fn=None, manual_agents=engine._default_seed_agents())

    print("=== EvoAgent finished ===")
    print(f"Best score: {best.score:.4f}")
    print("Best agent saved to populations/current_best_agent.py and populations/best_agent.py")


if __name__ == "__main__":
    main()
