# HSCodeComp Agent — Reproduction Code

This directory contains the agent-based reproduction code for the **HSCodeComp** benchmark, enabling automated HS code classification using LLM agents with tool-use capabilities.

## Overview

The system uses [hermes-agent](https://github.com/nousresearch/hermes-agent) as the underlying agent framework, equipped with:
- **Web Search & Extract** — search for CBP rulings, customs databases, and tariff references
- **Local HTS Tree Tools** — query the US Harmonized Tariff Schedule (2025 Rev.15) hierarchy locally
- **GRI Rules Injection** — optionally inject the 6 General Rules of Interpretation into the system prompt

## Requirements

- Python >= 3.11
- An LLM API key (DashScope / OpenAI / Anthropic)
- A web search backend API key (Exa / Firecrawl / Tavily, for hermes-agent's built-in web tools)

## Installation

```bash
# 1. Clone the repository (if not already done)
cd HSCodeComp/code

# 2. Install hermes-agent (from source recommended)
pip install git+https://github.com/nousresearch/hermes-agent.git
# Or if available on PyPI:
# pip install hermes-agent>=0.14.0

# 3. Install this project
pip install -e .

# 4. (Optional) For Claude model support
pip install -e ".[anthropic]"
```

## Configuration

### Step 1: Create `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in your API credentials:

```env
# LLM API (choose based on model)
DASHSCOPE_API_KEY=sk-your-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Web search backend (at least one required)
EXA_API_KEY=your-exa-key
# Or: FIRECRAWL_API_KEY=your-key / TAVILY_API_KEY=your-key
```

### Step 2: Place test data

Place `test_data.jsonl` from the HSCodeComp dataset into `data/`:

```bash
cp ../data/test_data.jsonl ./data/
```

## Quick Start

### Interactive Mode (single query)

```bash
python run.py
# Or specify a model:
python run.py --model glm-5.2
# Single query mode:
python run.py --query "Classify: stainless steel kitchen knife set"
```

### Batch Testing (full benchmark)

```bash
# One-command: inference + evaluation
./run.sh

# Specify model and parallelism
./run.sh -m qwen3.7-max -p 4

# Smoke test (10 questions)
./run.sh -n 10

# Resume interrupted run
./run.sh --resume
```

### Reflection Verification

After initial inference, run reflection to verify and potentially correct predictions:

```bash
./run_reflect.sh -i output/qwen3.7-max_20260626_120219.jsonl
```

### Evaluation Only

```bash
# Evaluate existing results
python eval.py output/result.jsonl

# Compare multiple models
python eval.py output/*.jsonl --compare

# Evaluate reflection results
python eval_reflect.py output_reflect/result.jsonl
```

## Project Structure

```
code/
├── run.py                  # Interactive single-query entry point
├── batch_test.py           # Batch inference runner (parallel + resume)
├── batch_reflect.py        # Reflection verification runner
├── eval.py                 # Multi-level accuracy evaluation
├── eval_reflect.py         # Reflection effect evaluation
├── analyze_cross_usage.py  # CBP CROSS ruling usage statistics
├── agent_factory.py        # Agent creation & config loading
├── custom_hts_tools.py     # Local HTS tree query tools
├── _resources.py           # Path resolution utilities
├── config.yaml             # Model presets & agent configuration
├── run.sh                  # One-click inference + eval script
├── run_reflect.sh          # One-click reflection + eval script
├── prompts/
│   ├── hscode_system.md    # Classification system prompt
│   ├── hscode_reflect.md   # Reflection system prompt
│   └── gri_rules.md        # GRI 6 rules (General Rules of Interpretation)
├── data/
│   ├── test_data.jsonl     # HSCodeComp test set (place here)
│   ├── hts_tree.json       # HTS hierarchy tree
│   └── hts_2025_revision_15.json  # Full USITC HTS data
├── output/                 # Inference results (generated)
├── logs/                   # Session logs (generated)
└── scripts/
    ├── batch_test.sh       # Convenience batch runner
    └── eval_compare.sh     # Multi-model comparison
```

## Supported Models

| Model | Provider | Config Key |
|-------|----------|------------|
| qwen3.7-max | DashScope | `DASHSCOPE_*` |
| glm-5.2 | DashScope | `DASHSCOPE_*` |
| deepseek-v4-pro | DashScope | `DASHSCOPE_*` |
| gpt-5.5 | OpenAI | `OPENAI_*` |
| claude-opus-4-8 | Anthropic | `ANTHROPIC_*` |

To add a new model, edit `config.yaml` under the `models:` section.

## Evaluation Metrics

The evaluation computes accuracy at multiple granularity levels:
- **2-digit** (Chapter level)
- **4-digit** (Heading level)
- **6-digit** (Subheading level, internationally harmonized)
- **8-digit** (US Tariff level)
- **10-digit** (Statistical suffix, exact match)

## Key Options

| Flag | Description |
|------|-------------|
| `-m, --model` | Model preset name (default: qwen3.7-max) |
| `-p, --parallel` | Concurrency level (default: 4) |
| `-n, --limit` | Limit number of questions (for smoke testing) |
| `--resume` | Skip completed cases, retry failed ones |
| `--no-gri-rules` | Disable GRI rules injection in system prompt |
| `--eval-only` | Only run evaluation on existing results |

## License

Apache-2.0
