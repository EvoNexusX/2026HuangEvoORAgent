# Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization

## Summary

This repository contains the official implementation of **"Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization"**, a research project on evolving agent architectures for operations research (OR) code generation.

**Paper:** Published on arXiv

The core contribution is **EvoOR-Agent**, a framework that treats both agent architectures and reasoning trajectories as evolvable objects, enabling task-adaptive problem-solving for complex optimization tasks without relying on fixed, hand-crafted workflows.

---

## 1. Introduction

Automating operations research (OR) with large language models (LLMs) remains limited by hand-crafted reasoning--execution workflows. Complex OR tasks require adaptive coordination among problem interpretation, mathematical formulation, solver selection, code generation, and iterative debugging. To address this limitation, we propose EvoOR-Agent, a co-evolutionary framework for automated optimization. The framework represents agent workflows as activity-on-edge (AOE)-style networks, making workflow topology, execution dependencies, and alternative reasoning paths explicit. On this representation, the framework maintains an architecture graph and evolves a population of reasoning individuals through graph-mediated path-conditioned recombination, multi-granularity semantic mutation, and elitist population update. A knowledge-base-assisted experience-acquisition module further injects reusable OR practices into initialization and semantic variation. Empirical results on heterogeneous OR benchmarks show that the proposed framework consistently improves over zero-shot LLMs, fixed-pipeline OR agents, and representative evolutionary agent frameworks. Case studies and ablation analyses further indicate that explicit architecture evolution and graph-supported reasoning-trajectory search contribute to both performance improvement and structural interpretability. These results suggest that treating agent architectures and reasoning trajectories as evolvable objects provides an effective route toward adaptive and interpretable automated optimization.

## 2. Overview

We propose **EvoOR-Agent**, a co-evolutionary framework for automated optimization. The central idea is to treat the internal organization of an OR agent not as a fixed prompting scaffold, but as an evolvable architecture on which different reasoning trajectories can be instantiated and evaluated. Specifically, we abstract agent workflows into an activity-on-edge (AOE)-style network, where reasoning states, execution dependencies, and alternative solution paths are represented as structured components. This representation exposes the workflow topology of an agent and provides a graph-supported search space for subsequent evolutionary optimization.

Building on this representation, the proposed framework couples architecture evolution with reasoning-trajectory evolution. At the architecture level, the framework maintains a global architecture graph by inserting newly discovered reasoning structures, updating edge weights according to empirical fitness, and pruning persistently weak components. At the trajectory level, it evolves a population of reasoning individuals through graph-mediated path-conditioned recombination, multi-granularity semantic mutation, and elitist population update. In addition, an LLM-driven experience-acquisition module constructs a domain-specific knowledge base from reusable OR practices, which is then used to support initialization and knowledge-guided mutation. Through this coupling, the framework can adapt task decomposition, solver selection, tool use, and downstream execution without relying on a fully predefined reasoning--execution pipeline.

Overall, the main contributions of this work are as follows:

1. We formulate agentic OR workflows as explicit **agent architectures** by representing them as AOE-style networks. This converts implicit reasoning--execution organization into a structured and evolvable search space, exposing workflow topology, dependency relations, and alternative execution paths in a form that is analyzable, manipulable, and interpretable.

2. We develop an evolutionary search mechanism for **reasoning trajectories** instantiated on the maintained agent architecture. The proposed framework evolves reasoning individuals through graph-mediated path-conditioned recombination, semantic mutation, and elitist population update, while the architecture graph is updated in tandem based on the execution traces and fitness of evolved individuals. This design enables task-adaptive reasoning organization for problem formulation, solver selection, tool use, and code execution.

We evaluate the proposed framework on heterogeneous OR benchmark families covering mathematical formulation, solver-oriented reasoning, and industrial optimization scenarios. The experiments compare our method with zero-shot reasoning models, fixed-pipeline LLM-based OR agents, specialized OR modeling methods, and representative evolutionary agent frameworks under a unified evaluation protocol. The results show that treating agent architecture and reasoning trajectories as evolvable objects yields consistent improvements over fixed-pipeline approaches and representative evolutionary baselines. The ablation study further shows that both the AOE-style architecture representation and the knowledge-base-assisted evolutionary operators contribute to the final performance. In addition, the case study and evolutionary dynamics analyses provide evidence that the learned architecture graph captures interpretable reasoning structures, including formulation decomposition, solver-routing decisions, and semantic debugging behavior.

### Figures (Customizable)


#### Figure 1: Evolution of OR problem-solving paradigms

![Figure 1 Placeholder](picture/new-1.png)

The upper path illustrates the classical expert-driven OR workflow, in which practitioners formulate mathematical models from task requirements and then design or select suitable solvers. The lower path shows the LLM-driven paradigm, where planning, reflection, and tool use are integrated into an agentic reasoning--execution chain. Existing LLM-based systems improve semantic automation but usually retain fixed workflow structures, whereas our framework treats agent architecture and the reasoning trajectories instantiated on it as evolvable objects.

#### Figure 2: Architecture graph evolution

![Figure 2 Placeholder](picture/new-2.png)
Architecture graph evolution. Individual OR agents are first abstracted into AOE chains, which are merged by phase-local state alignment to form the initial architecture graph. During evolution, newly discovered structures are inserted into the graph, while persistently weak nodes and edges are pruned. The maintained graph defines the current architecture space.

#### Figure 3: Overview of reasoning trajectory evolution on the current architecture graph.

![Figure 3 Placeholder](picture/new-3.png)

Overview of reasoning trajectory evolution on the current architecture graph. An LLM-agent-based experience acquisition workflow retrieves relevant papers and code repositories, and organizes reusable OR practices into a domain-specific knowledge base. The knowledge base supports initialization and semantic mutation. During each generation, path-conditioned recombination samples new reasoning trajectories from the current graph, semantic mutation revises selected individuals at different granularities, and multi-source selection forms the next population.


#### Figure 4 Result Comparison
![Figure 4 Placeholder](picture/1.jpg)
Comparison of accuracy performance on NL4OPT, MAMO (ComplexLP/EasyLP), IndustryOR, and BWOR benchmarks. Results report the mean and standard deviation ($\pm$ SD) over ten independent runs.

## 3. Directory Structure

```text
Co-evolving agent/
├─ initial.py                  # Population initialization entry point
├─ evaluate.py                 # Multi-generational evolution entry point
├─ new_utils.py                # LLM query interface and common utilities
├─ end_agent.py                # Intermediate agent generation (auxiliary)
├─ prompt/                     # Prompt templates (chain/change/knowledge/tool, etc.)
├─ dataset/                    # Evaluation datasets
├─ data/                       # Data processing and analysis scripts
├─ population/                 # Population and evolution results
├─ Evo_agent/                  # Evo_agent reproduction
├─ Evo_prompt/                 # Evo_prompt reproduction
├─ or_llm_agent/               # or_llm_agent reproduction
├─ test/                       # Test and experiment scripts
├─ log/                        # Log directory
└─ requirements.txt            # Python dependencies list
```

---

## 4. Installation

### 4.1 Clone Repository

```bash
git clone https://github.com/EvoNexusX/2026huangCo-evolving-Agent.git
cd "2026huangCo-evolving-Agent"
```

### 4.2 System Requirements

- Python: `>=3.10` (recommended `3.10/3.11`)
- Operating System: Windows / Linux / macOS
- Recommended: Use virtual environment (venv or conda)

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

If your generated code depends on Gurobi, please install and configure it separately:

- Gurobi Optimizer (main package)
- gurobipy (version-matched with Gurobi)

> You can install `gurobipy` individually based on your environment, for example:
>
> ```bash
> pip install gurobipy
> ```

### 4.4 LLM Configuration

The LLM configuration in this repository has been anonymized (model names, API keys, and endpoints are empty).
You need to add configuration in the LLM utility files (e.g., [new_utils.py](new_utils.py) and its copies in experiment subdirectories).

Recommended approach:

1. Set your API Key and Base URL as environment variables in your local system.
2. Add these to `query_llm(...)`:
   - Environment variable key names (e.g., `os.getenv("YOUR_KEY")`)
   - Default model name (via `--model` parameter or function default)
3. Verify that `model_name` matches available models on the service.

> **Tip**: If configuration fails or you need to customize LLM parameters (e.g., model, temperature, max_tokens), please check the source code in [new_utils.py](new_utils.py) and so on, and modify the relevant functions as needed.

### 4.5 Running Instructions

#### A. Initialize Population

```bash
python initial.py -n 6 -m <your_model_name>
```

Parameters:

- `-n/--num`: Total number of individuals to generate (default: 6)
- `-m/--model`: Model name to use (currently empty by default, must be explicitly specified)

#### B. Run Evolution

```bash
python evaluate.py --mode from-scratch -n 6 -m <your_model_name> --seed 42
```

Or resume from checkpoint:

```bash
python evaluate.py --mode resume -n 6 -m <your_model_name> --seed 42
```
> **Tip**: If you need to fine-tune evolution parameters (e.g., selection strategy, crossover/mutation methods, fitness calculation), please refer to the source code in [initial.py](initial.py) and so on, and modify the corresponding evolution loop and evaluation functions.

Common parameters:

- `--mode`: `from-scratch` / `resume`
- `-n/--num`: Population size per generation (`0` means use previous generation size)
- `-m/--model`: Model name to use
- `--seed`: Random seed
- `--elite-rate`: Elite preservation rate
- `--cross-rate`: Crossover rate
- `--learn-rate`: Learning rate in mutation

> **Tip**: If you need to fine-tune evolution parameters (e.g., selection strategy, crossover/mutation methods, fitness calculation), please refer to the source code in [evaluate.py](evaluate.py) and so on, and modify the corresponding evolution loop and evaluation functions.


