from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

try:
    from new_utils import query_llm
except Exception:
    query_llm = None


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
POPULATIONS_DIR = os.path.join(WORKSPACE_DIR, "populations")
CURRENT_BEST_PROMPT_FILE = os.path.join(POPULATIONS_DIR, "current_best_prompt.txt")
LOG_FILE = os.path.join(POPULATIONS_DIR, "evo_prompt_log.jsonl")
DEFAULT_LLM_MODEL = ""

DEFAULT_BASE_PROMPTS = [
    (
        "You are an operations research expert. Build a correct mathematical model first, then output "
        "complete executable Python Gurobi code only in a fenced python block. Ensure objective, variables, "
        "constraints, solve status check, and final objective print are all included."
    ),
    (
        "Act as a robust OR modeling assistant. Convert the user problem to LP/MIP formulation and produce "
        "fully runnable Python Gurobi code. Keep the code deterministic, readable, and directly executable. "
        "Return code only using ```python ... ```."
    ),
    (
        "Role: OR optimization engineer. Task: infer decision variables, objective, and constraints from the "
        "problem statement; generate complete Python Gurobi solver code with clear variable naming and exact "
        "result output. Return only executable code."
    ),
    (
        "You solve optimization tasks with Gurobi. Think in this order: data parse -> variable design -> "
        "objective -> constraints -> optimize -> report objective value. Output only final Python code in one "
        "code block."
    ),
]

GA_OPERATOR_TEMPLATE = """You are an EvoPrompt genetic operator.

Step 1 (Crossover): Fuse instruction strengths from both parent prompts.
Step 2 (Mutation): Locally rewrite wording with equivalent intent, improve precision, and reduce verbosity.

Parent A:
{parent1}

Parent B:
{parent2}

Current Best Reference:
{best_prompt}

Rules:
1) Keep role and task intent aligned with OR-to-Gurobi code generation.
2) Keep output concise and operational.
3) Do not copy any parent verbatim.
4) Output only one tag pair: <prompt> ... </prompt>
"""

INIT_VARIANT_TEMPLATE = """Generate one compact prompt variant for OR code generation.
Base prompt:
{best_prompt}

Rules:
1) Keep original intent.
2) Improve clarity and structure.
3) Output strictly <prompt> ... </prompt>
"""

MetricFn = Callable[[Any], float]
EvaluatorFn = Callable[[str, Iterable[Any], Optional[MetricFn], random.Random], float]


@dataclass
class PromptCandidate:
    prompt: str
    score: float
    generation: int
    origin: str
    parent_a: Optional[str] = None
    parent_b: Optional[str] = None


class EvoPromptGA:
    """Discrete prompt optimization with GA operators and reproducible random evaluation."""

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

        self.population: list[PromptCandidate] = []
        self.api_calls = 0
        self.best_global: Optional[PromptCandidate] = None

        os.makedirs(POPULATIONS_DIR, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(POPULATIONS_DIR, "evo_prompt.log"), encoding="utf-8"),
            ],
        )

    def _append_json_log(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _extract_prompt_tag(self, text: str) -> Optional[str]:
        match = re.search(r"<prompt>([\s\S]*?)</prompt>", text)
        if not match:
            return None
        prompt = match.group(1).strip()
        return prompt if prompt else None

    def _llm_generate(self, user_content: str) -> Optional[str]:
        if query_llm is None:
            return None

        model_name = self.model_name
        # DeepSeek endpoint does not serve GPT model names; map to a safe default.
        if model_name.lower().startswith("gpt-"):
            model_name = DEFAULT_LLM_MODEL

        messages = [
            {"role": "system", "content": "You are an expert prompt optimizer."},
            {"role": "user", "content": user_content},
        ]
        self.api_calls += 1
        try:
            raw = query_llm(messages, model_name)
        except Exception as exc:
            logging.warning("LLM call failed (%s), fallback to local operator.", exc)
            return None
        return self._extract_prompt_tag(raw)

    def _local_mutation(self, text: str) -> str:
        synonym_pool = {
            "complete": ["full", "end-to-end", "comprehensive"],
            "correct": ["valid", "accurate", "sound"],
            "constraints": ["restriction set", "constraint set", "rules"],
            "objective": ["target", "optimization goal", "objective function"],
            "executable": ["runnable", "directly runnable", "ready-to-run"],
            "concise": ["compact", "brief", "minimal"],
        }

        mutated = text
        for key, values in synonym_pool.items():
            if key in mutated and self.rng.random() < 0.3:
                mutated = mutated.replace(key, self.rng.choice(values), 1)

        suffix_options = [
            "Prefer deterministic variable names and explicit solve-status checks.",
            "Avoid extra explanation and return only the final code block.",
            "Favor stable structure: model, vars, constraints, objective, optimize, print.",
        ]
        if self.rng.random() < 0.5:
            mutated = f"{mutated} {self.rng.choice(suffix_options)}"

        return " ".join(mutated.split())

    def _local_crossover_mutation(self, parent1: str, parent2: str) -> str:
        p1_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", parent1) if p.strip()]
        p2_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", parent2) if p.strip()]

        take1 = max(1, len(p1_parts) // 2)
        take2 = max(1, len(p2_parts) // 2)
        fused = p1_parts[:take1] + p2_parts[-take2:]

        prompt = " ".join(fused)
        return self._local_mutation(prompt)

    def _ga_offspring(self, parent1: str, parent2: str, best_prompt: str) -> str:
        llm_prompt = GA_OPERATOR_TEMPLATE.format(
            parent1=parent1,
            parent2=parent2,
            best_prompt=best_prompt,
        )
        llm_output = self._llm_generate(llm_prompt)
        if llm_output:
            return llm_output
        return self._local_crossover_mutation(parent1, parent2)

    def _init_llm_variant(self, best_prompt: str) -> Optional[str]:
        llm_prompt = INIT_VARIANT_TEMPLATE.format(best_prompt=best_prompt)
        return self._llm_generate(llm_prompt)

    def _random_evaluator(
        self,
        prompt: str,
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn],
        rng: random.Random,
    ) -> float:
        # Demo evaluator: reproducible pseudo-score to show evolution behavior.
        del dataloader, metric_fn
        length_bonus = min(len(prompt) / 900.0, 0.15)
        return round(rng.uniform(60.0, 90.0) * (1.0 + length_bonus), 4)

    def _score_prompt(self, prompt: str, dataloader: Iterable[Any], metric_fn: Optional[MetricFn]) -> float:
        return float(self.evaluator(prompt, dataloader, metric_fn, self.rng))

    def _select_two_parents(self) -> tuple[PromptCandidate, PromptCandidate]:
        weights = [max(p.score, 1e-6) for p in self.population]
        parent_a = self.rng.choices(self.population, weights=weights, k=1)[0]
        parent_b = self.rng.choices(self.population, weights=weights, k=1)[0]
        return parent_a, parent_b

    def _write_prompt_txt(self, generation: int, name: str, prompt: str) -> str:
        gen_dir = os.path.join(POPULATIONS_DIR, f"gen_{generation:02d}")
        os.makedirs(gen_dir, exist_ok=True)
        path = os.path.join(gen_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        return path

    def _replace_new_agent_prompt(self, prompt: str) -> None:
        with open(CURRENT_BEST_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")

    def _write_population_snapshot(self, generation: int, candidates: list[PromptCandidate]) -> None:
        """Save one snapshot per individual using Co-evolving-style folder names: ex{gen}_p{idx}."""
        for idx, candidate in enumerate(candidates, start=1):
            folder = os.path.join(POPULATIONS_DIR, f"ex{generation}_p{idx}")
            os.makedirs(folder, exist_ok=True)

            # Keep prompt in tool.txt to mimic the original populations directory style.
            with open(os.path.join(folder, "tool.txt"), "w", encoding="utf-8") as f:
                f.write(candidate.prompt.strip() + "\n")

            meta = {
                "strategy": candidate.origin,
                "generation": generation,
                "population_index": idx,
                "score": candidate.score,
                "parent_a": candidate.parent_a,
                "parent_b": candidate.parent_b,
            }
            with open(os.path.join(folder, "ea_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def initialize_population(
        self,
        manual_prompts: list[str],
        dataloader: Iterable[Any],
        metric_fn: Optional[MetricFn],
    ) -> None:
        prompts: list[str] = []
        for p in manual_prompts:
            p = p.strip()
            if p and p not in prompts:
                prompts.append(p)

        while len(prompts) < self.population_size:
            best_seed = prompts[0] if prompts else DEFAULT_BASE_PROMPTS[0]
            llm_variant = self._init_llm_variant(best_seed)
            candidate = llm_variant if llm_variant else self._local_mutation(best_seed)
            if candidate not in prompts:
                prompts.append(candidate)

        self.population = []
        for idx, prompt in enumerate(prompts[: self.population_size], start=1):
            score = self._score_prompt(prompt, dataloader, metric_fn)
            item = PromptCandidate(prompt=prompt, score=score, generation=0, origin="init")
            self.population.append(item)
            self._write_prompt_txt(0, f"p{idx:02d}_score_{score:.4f}", prompt)

        self.population.sort(key=lambda x: x.score, reverse=True)
        self.best_global = self.population[0]
        self._replace_new_agent_prompt(self.best_global.prompt)
        self._write_population_snapshot(1, self.population)

        avg_score = sum(p.score for p in self.population) / len(self.population)
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
        manual_prompts: Optional[list[str]] = None,
    ) -> PromptCandidate:
        if not self.population:
            seeds = manual_prompts or list(DEFAULT_BASE_PROMPTS)
            self.initialize_population(seeds, dataloader, metric_fn)

        prev_avg = sum(p.score for p in self.population) / len(self.population)
        no_improve_rounds = 0

        for generation in range(1, self.max_generations + 1):
            offspring: list[PromptCandidate] = []

            for idx in range(1, self.offspring_per_generation + 1):
                parent_a, parent_b = self._select_two_parents()
                best_prompt = self.population[0].prompt
                child_prompt = self._ga_offspring(parent_a.prompt, parent_b.prompt, best_prompt)
                child_score = self._score_prompt(child_prompt, dataloader, metric_fn)

                child = PromptCandidate(
                    prompt=child_prompt,
                    score=child_score,
                    generation=generation,
                    origin="ga",
                    parent_a=parent_a.prompt,
                    parent_b=parent_b.prompt,
                )
                offspring.append(child)
                self._write_prompt_txt(generation, f"child_{idx:02d}_score_{child_score:.4f}", child_prompt)

            merged = self.population + offspring
            merged.sort(key=lambda x: x.score, reverse=True)
            self.population = merged[: self.population_size]
            self._write_population_snapshot(generation + 1, self.population)

            best_now = self.population[0]
            avg_now = sum(p.score for p in self.population) / len(self.population)

            if self.best_global is None or best_now.score > self.best_global.score:
                self.best_global = best_now
            self._replace_new_agent_prompt(self.best_global.prompt)

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

        final_path = self._write_prompt_txt(99, f"best_score_{self.best_global.score:.4f}", self.best_global.prompt)
        logging.info("Global best score=%.4f saved=%s", self.best_global.score, final_path)
        self._append_json_log({"event": "finished", "best": asdict(self.best_global), "api_calls": self.api_calls})
        return self.best_global


# ----------------------------
# Demo hooks (decoupled eval)
# ----------------------------
def demo_metric_fn(_: Any) -> float:
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoPrompt discrete optimization (GA)")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model for evolution operators")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size N")
    parser.add_argument("--offspring", type=int, default=10, help="New prompts per generation")
    parser.add_argument("--generations", type=int, default=8, help="Max generation count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stop-threshold", type=float, default=0.003, help="Avg improvement threshold")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="Consecutive rounds for early stop")
    return parser.parse_args()


def main(
    population_size: Optional[int] = 10,
    training_generations: Optional[int] = 8,
) -> None:
    # Keep CLI support, but allow direct in-code overrides for quick experiments.
    cli_args = parse_args()

    resolved_population_size = population_size if population_size is not None else cli_args.pop_size
    resolved_generations = training_generations if training_generations is not None else cli_args.generations

    # Placeholder dataloader. Replace with your real validation DataLoader/iterator.
    dataloader = [
        {"id": 1, "question": "demo-1"},
        {"id": 2, "question": "demo-2"},
    ]

    engine = EvoPromptGA(
        model_name=cli_args.model,
        population_size=resolved_population_size,
        offspring_per_generation=cli_args.offspring,
        max_generations=resolved_generations,
        seed=cli_args.seed,
        early_stop_patience=cli_args.early_stop_patience,
        early_stop_threshold=cli_args.early_stop_threshold,
    )
    best = engine.run(dataloader=dataloader, metric_fn=demo_metric_fn, manual_prompts=list(DEFAULT_BASE_PROMPTS))

    print("=== EvoPrompt finished ===")
    print(f"Best score: {best.score:.4f}")
    print("Best prompt saved to populations/current_best_prompt.txt and populations/gen_99/")


if __name__ == "__main__":
    main()

