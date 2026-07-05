# UMEM: Unified Memory Extraction and Management for Generalizable Self‑Evolving Memory

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/abs/2405.xxxxx) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg)](https://huggingface.co/your-model)

⭐ _**MarcoPolo Team**_ ⭐

[_**Alibaba Group**_](https://www.qianwenai.com/)

<img src="https://octodex.github.com/images/original.png" alt="GitHub Octocat" width="22" height="22">[**GitHub**](https://github.com/AIDC-AI/Marco-DeepResearch/tree/main/UMEM)  🤗  [**Models**](<HF_MODEL_URL>)  📝  [**Paper**](<PAPER_URL_OR_ARXIV_URL>)

</div>

---


## 📌 Overview

**UMEM (Unified Memory Extraction and Management)** is a self‑evolving agent framework that **jointly optimizes memory extraction and memory management** to produce **generalizable** long‑term memories.

Most prior self‑evolving memory systems optimize *management* (e.g., select / retain / replace) but treat *extraction* as a static prompting step. This often leads to the **“Rote Memorization” trap**: the agent accumulates **instance‑specific shortcuts/noise** that pollute the memory bank and degrade generalization over time.

<p align="center"><img src="assets/umem_intro.png" alt="UMEM Teaser" width="850" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></p>

UMEM addresses this by training a **Mem‑Optimizer** policy that learns to:
- **distill** reusable experiences into memory entries, and
- decide how to **ADD / UPDATE** the memory bank,

while explicitly optimizing for **cross‑query generalization** via:
- **Semantic Neighborhood Modeling (SNM)**, and
- a **neighborhood‑level marginal utility reward** optimized with **GRPO**,
- plus **Online Memory Evolution** (commit best rollouts to the bank during training).

<p align="center"><img src="assets/umem_overview.png" alt="UMEM Overview" width="850" style="max-width: 100%; height: auto;"></p>


---


## 🔥 News
* [2026/02/11] Initial code release.

---



## 🧠 Method

UMEM consists of three core components:

1) **Executor (Frozen)**  
A frozen LLM/agent that solves tasks with retrieved memories.

2) **Memory Bank (Evolvable)**  
A non‑parametric key–value store of experiences, updated over time.

3) **Mem‑Optimizer (Learned Policy)**  
A policy model that reads executor trajectories and outputs structured memory actions (ADD/UPDATE).

### Semantic Neighborhood Modeling (SNM)

To prevent overfitting to a single instance, UMEM constructs a **semantic neighborhood** $\mathcal{N}_N(q)$ for each training query `q`:
- embed all queries with a pretrained encoder (e.g., **BGE‑M3**),
- retrieve **Top‑N** nearest neighbor queries (paper uses **N = 3**).

Candidate memory updates are evaluated **over the neighborhood**, forcing extracted memories to be reusable across semantically related queries.

### Marginal Utility Reward (Neighborhood‑level)

For each neighbor query $q^{\prime} \in \mathcal{N}_N(q)$, compare:
- **reference execution** (without applying a candidate memory update), vs.
- **memory‑augmented execution** (with the candidate update applied),

and compute marginal utility as:
- **Success Gain**: rewards fixing incorrect cases and penalizes breaking correct ones.
- **Efficiency Regularization**: rewards shorter trajectories **only when correctness is preserved**.

UMEM also uses a **format reward** to enforce strict XML schema outputs.

### GRPO + Online Memory Evolution

During training:
- sample **G** candidate memory actions per query (paper uses **G = 8**),
- compute neighborhood‑level rewards,
- update Mem‑Optimizer with **GRPO**,
- **commit the best‑reward action** into the memory bank (online evolution).

---

## 🚀 Quick Start

### 📁 Repository Structure

```text
UMEM/
├── assets/                     # Static assets (images, logos, etc.)
├── data/                       # Datasets (.parquet, .json) and preprocessing scripts
├── umem_scripts/               # Entry scripts for training and evaluation
│   ├── eval_umem.py            # Main Python script for evaluation logic
│   ├── run_eval.sh             # Shell script to launch evaluation
│   └── run_train.sh            # Shell script to launch training
├── verl/                       # Core source code library
│   ├── __init__.py
│   │
│   ├── trainer/                # Training main loops and algorithm implementations
│   │   ├── main_ppo.py         # Entry point for PPO algorithm
│   │   └── ...
│   │
│   ├── umem/                   # UMEM algorithm-specific components
│   │   ├── __init__.py
│   │   ├── llm_agent/          # Executor and Extractor
│   │   │   └── ...
│   │   └── memory/             # Memory module: Vector DB and KV Cache management
│   │       └── ...
│   │
│   ├── workers/                # Ray distributed workers for parallel computing
│   │   ├── retriever           # Retrieval service worker
│   │   └── ...
│   │
│   └── utils/                  # General utility functions and tools
│       └── ...
│
├── requirements.txt            # Project dependencies list
└── setup.py                    # Installation script (pip install -e .)
```

---

### 🛠️ Installation

```bash
git clone https://github.com/AIDC-AI/Marco-DeepResearch.git
cd Marco-DeepWideSearch-Agent/UMEM

conda create -n umem python=3.10 -y
conda activate umem

pip install -r requirements.txt
pip install -e .
```

---


## 🗃️ Memory Action Format

Mem‑Optimizer outputs structured actions in strict XML schema:

```xml
<experience>
  <value>...</value>
  <operation>ADD</operation>
</experience>
```

or

```xml
<experience>
  <value>...</value>
  <operation>UPDATE 3</operation>
</experience>
```
<!-- 
**Critical constraints (paper Appendix D)**:
1. **NO ANSWER LEAKAGE**: never store option indices or final answer strings.
2. **NO SPECIFICS**: remove instance-specific numbers/names; rewrite as variables/concepts.
3. **NO HALLUCINATION**: do not invent facts. -->

---

## 🏋️ Training (Mem‑Optimizer)

### 1) Prepare Training Data (MMLU)

```bash
python scripts/prepare_mmlu.py \
  --output data/mmlu_ori.jsonl \
  --num_samples 2000
```

### 2) Build Semantic Neighborhoods (SNM)

```bash
python scripts/build_neighborhoods.py \
  --input data/mmlu_ori.jsonl \
  --encoder bge-m3 \
  --top_n 3 \
  --output data/mmlu.jsonl
```

### 3) GRPO Training + Online Memory Evolution

```bash
bash umem_scripts/run_train.sh
```

---

## 🧪 Evaluation

UMEM evaluates in a **streaming** protocol (tasks processed sequentially; memory bank not reset).

```bash
bash umem_scripts/run_eval.sh
```

### Cumulative Performance
![Cumulative Performance](assets/cumulative_curves.png)

---

## 🛡️ License

This project is licensed under **Apache‑2.0** (update if different).

---

## 📖 Citation

If you use UMEM in your research, please cite:

```bibtex
@misc{ye2026umemunifiedmemoryextraction,
      title={UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory}, 
      author={Yongshi Ye and Hui Jiang and Feihu Jiang and Tian Lan and Yichao Du and Biao Fu and Xiaodong Shi and Qianghuai Jia and Longyue Wang and Weihua Luo},
      year={2026},
      eprint={2602.10652},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10652}, 
}
```

---
