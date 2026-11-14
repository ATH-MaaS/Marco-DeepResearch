# Search2Skill — Usage Guide

This document describes how to set up, configure, and run the Search2Skill evaluation code.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
  - [Streaming Evaluation](#streaming-evaluation-table-2-in-paper)
  - [Held-out (Pipeline) Evaluation](#held-out-pipeline-evaluation-table-3-in-paper)
  - [Knowledge Gap Experiment](#knowledge-gap-experiment-section-41-in-paper)
  - [Batch Scripts](#batch-scripts)
- [Evaluation Modes](#evaluation-modes)
- [Datasets](#datasets)
- [Output](#output)
- [SFT Training Data](#sft-training-data)
- [Environment Variables](#environment-variables)

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/ATH-MaaS/Marco-DeepResearch.git
cd Marco-DeepResearch
cd Marco-DeepResearch-Family/Search2Skill
```

Run all subsequent commands from `Marco-DeepResearch-Family/Search2Skill`. This repository contains code and documentation only; evaluation, SFT, and RL training datasets must be prepared separately. See [Datasets](#datasets) before running the examples.

### 2. Install dependencies

Requires **Python ≥ 3.10**.

The bundled `verl/` training code uses Python 3.12 syntax and has additional dependencies in `verl/requirements.txt` and `verl/pyproject.toml`; use a separate Python 3.12 training environment with compatible accelerator libraries.

```bash
pip install -r requirements.txt
```

Key dependencies: `openai`, `datasets`, `pyyaml`, `tavily-python`, `duckduckgo-search`, `rapidfuzz`, `markdownify`, `openpyxl`.

> **Note:** For local model deployment via vLLM, install vLLM separately:
> ```bash
> pip install vllm
> ```

---

## Configuration

All runtime settings are in **`config/config.yaml`**. Replace the `<YOUR_...>` placeholders before running.

### LLM Profiles

The system supports both **local vLLM** and **remote API** (OpenAI-compatible) backends. The `default` field selects which profile is used for the main agent loop.

```yaml
llm:
  default: "agent-llm"
  profiles:
    # Local vLLM
    qwen3-8b-local:
      backend: "vllm"
      model_path: "/path/to/Qwen3-8B"
      port: 8002
      gpu_memory_utilization: 0.9
      tensor_parallel_size: 1
      enable_thinking: true

    # Remote API (DashScope, OpenAI, etc.)
    agent-llm:
      model: "qwen3-32b"
      api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      api_key: "<YOUR_API_KEY>"
```

Additional profiles (`compress-llm`, `filter-llm`, `judge-llm`) are used for page compression, skill filtering, and answer judging respectively. They can share the same endpoint or point to different models.

### Web Search

Three engines are supported:

| Engine | API Key Required | Notes |
|--------|:-:|-------|
| `tavily` | ✅ | Recommended. Get a key at [tavily.com](https://tavily.com) (free tier available). |
| `duckduckgo` | ❌ | No key needed, but rate-limited and less reliable. |
| `google` | ✅ | Requires a search proxy service (see [Environment Variables](#environment-variables)). |

```yaml
web_search:
  engine: "tavily"
  tavily_api_key: "<YOUR_TAVILY_API_KEY>"
```

### Web Visit (Page Fetching)

Three fetch modes are available:

| Mode | Notes |
|------|-------|
| `jina` | **Default.** Uses [Jina Reader](https://r.jina.ai/) to convert pages to Markdown. Free, no key needed. |
| `requests` | Direct HTTP request + HTML-to-Markdown conversion. May fail on JS-rendered pages. |
| `proxy` | Custom search proxy service (see [Environment Variables](#environment-variables)). |

```yaml
web_visit:
  fetch_mode: "jina"
  compress_llm: "compress-llm"
```

### Embedding

Used for skill retrieval. The default is DashScope's `text-embedding-v4`:

```yaml
embedding:
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "<YOUR_EMBEDDING_API_KEY>"
  model: "text-embedding-v4"
  dimensions: 1024
```

---

## Running Evaluations

The main evaluation script is `scripts/eval_benchmark.py`.

### Streaming Evaluation (Table 2 in paper)

Questions arrive sequentially; skills distilled from earlier questions become available to later ones.

```bash
python scripts/eval_benchmark.py \
    --benchmark supergpqa \
    --data_file data/supergpqa_baseline/supergpqa_hard_mathematics_500.json \
    --mode direct search_skill free_tool \
    --model agent-llm \
    --num 200 \
    --concurrency 14 \
    --config config/config.yaml
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--benchmark` | Dataset name (see [Datasets](#datasets)) |
| `--data_file` | Path to a specific JSON data file |
| `--mode` | One or more evaluation modes (see [Modes](#evaluation-modes)) |
| `--model` | LLM profile name from `config.yaml` |
| `--num` | Number of questions to evaluate (`-1` = all) |
| `--concurrency` | Number of concurrent agent workers |
| `--config` | Path to config file (default: `config/config.yaml`) |

### Held-out (Pipeline) Evaluation (Table 3 in paper)

Two-phase protocol: mine skills on a train set, then evaluate with a frozen library on a test set (search disabled).

```bash
python scripts/eval_pipeline.py \
    --benchmark supergpqa_hard_mathematics \
    --train_file data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_train.json \
    --test_file data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_test.json \
    --model agent-llm \
    --test_mode direct skill_only \
    --config config/config.yaml
```

**Pipeline-specific arguments:**

| Argument | Description |
|----------|-------------|
| `--train_file` | Training data for skill mining |
| `--test_file` | Test data for held-out evaluation |
| `--test_mode` | Evaluation mode(s) for the test phase |

### Knowledge Gap Experiment (Section 4.1 in paper)

Evaluates whether the agent can self-identify knowledge gaps:

```bash
python scripts/exp1_knowledge_gap.py \
    --config config/config.yaml \
    --data_file omnimath \
    --student agent-llm \
    --teacher qwen37-max \
    --num 500
```

### Batch Scripts

Run evaluation across all datasets:

```bash
# Streaming evaluation on all datasets
bash scripts/run_all_eval.sh agent-llm config/config.yaml

# Pipeline evaluation on all train/test splits
bash scripts/run_pipeline_eval.sh agent-llm config/config.yaml
```

---

## Evaluation Modes

| Mode | Description |
|------|-------------|
| `direct` | Pure LLM zero-shot (no tools, no agent loop) |
| `code` | Agent + code execution only (no web search) |
| `free_tool` | Agent + web search + code (direct tool access, no skill generation) |
| `search_skill` | Agent + web search + code + skill generation & retrieval |
| `search_skill_no_code` | Same as `search_skill` but without code execution |
| `skill_only` | Agent + code + frozen skill library (no web search, no generation) |
| `skill_only_no_code` | Same as `skill_only` but without code execution |

Use `--mode` with one or more modes:

```bash
--mode direct                    # single mode
--mode direct search_skill       # multiple modes, run sequentially
--mode full                      # shorthand for running all modes
```

---

## Datasets

### Local Data Preparation

Datasets are not bundled with this code release. Prepare your own data in the format below and place it at the paths expected by the evaluation scripts, or pass your own paths with `--data_file`, `--train_file`, and `--test_file`. The `data/` and `verl/data/` directories are ignored by Git.

| Benchmark | Domains | Setting | Data Path |
|-----------|---------|---------|-----------|
| **SuperGPQA** | Hard Mathematics, Mixed Management, Middle Science | Streaming | `data/supergpqa_baseline/` |
| **MMLU-Pro** | Law, Philosophy, History | Streaming | `data/mmlu_pro_law/`, `data/mmlu_pro_philosophy/`, `data/mmlu_pro_history/` |
| **SuperGPQA** (split) | Hard Mathematics, Science P&C | Held-out | `data/supergpqa_reuse_dataset/` |
| **MMLU-Pro** (split) | Philosophy, Law, History | Held-out | `data/mmlu_pro_*/` |
| **EvoAgentBench** | Reasoning & Problem Decomposition | Held-out | `data/evoagentbench/` |
| **LiveCodeBench** | Competitive Programming | Held-out | `data/livecode/` |

### Additional Benchmarks (via HuggingFace `datasets`)

The following benchmarks are loaded on demand (cached locally after first download):

- `mmlu_pro` — Full MMLU-Pro (via `TIGER-Lab/MMLU-Pro`)
- `supergpqa` — Full SuperGPQA (via `m-a-p/SuperGPQA`)
- `gpqa_diamond`, `theoremqa`, `aime2025`, `aime2026`, `scibench`, `chembench4k`

### Data File Format

Evaluation JSON records use the following schema:

```json
{
  "id": "0",
  "question": "Question text...",
  "options": ["A. option1", "B. option2", "..."],
  "ground_truth": "A",
  "domain": "Contract Law",
  "metadata": { "subfield": "...", "difficulty": "..." }
}
```

---

## Output

Results are saved incrementally to `results/` with the following structure:

```
results/
└── <eval_type>/
    └── <benchmark>/
        └── <tag>/
            ├── results.jsonl          # Per-question raw results
            ├── detail.csv             # Per-question metrics
            ├── summary.xlsx           # Aggregated summary by mode/domain
            ├── summary_<mode>.txt     # Human-readable summary
            ├── config.txt             # Run configuration
            └── skill_library.json     # Generated skills (search_skill mode)
```

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `correct` | Whether the answer matches ground truth |
| `score` | Evaluation score (1.0 = correct, fuzzy match for open-ended) |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | Token usage |
| `web_search_count` | Number of web searches performed |
| `num_rounds` | Number of agent loop iterations |

---

## SFT Training Data

To use SFT training data with the existing configuration, prepare your own trajectory file at:

```
data/sft_data/mixed_filtered_v4.jsonl
```

SFT trajectories and RL training data are not included in this code release. Prepare RL datasets separately under `verl/data/` as required by your training configuration.

Each record contains:
- `system`: System prompt
- `conversations`: Multi-turn agent trajectories (with tool calls and observations)
- `meta`: Metadata (source benchmark, skill info, etc.)

Supervised fine-tuning can be reproduced with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) using this file as the training dataset.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API endpoint (for Claude models) |
| `ANTHROPIC_AUTH_TOKEN` | `""` | Anthropic API key |
| `SEARCH_PROXY_API_URL` | `""` | Google search proxy endpoint (if `engine: "google"`) |
| `SEARCH_PROXY_AUTHORIZATION` | `""` | Auth token for search proxy |
| `SEARCH_PROXY_READ_URL` | `""` | Page read proxy endpoint (if `fetch_mode: "proxy"`) |
| `SEARCH_PROXY_READ_SOURCE` | `""` | Source identifier for page read proxy |
| `JINA_READER_URL_PREFIX` | `https://r.jina.ai/` | Jina Reader API prefix |
| `EXSKILL_GOOGLE_SEARCH_CACHE_DIR` | `.cache/google_search_cache/` | Google search cache directory |
| `EXSKILL_LLM_TRAJ_LOGGING` | `false` | Enable verbose LLM trajectory logging |
