# Table-as-Search: Agent Framework for Deep and Wide Information Seeking

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/PLACEHOLDER)

<div align="center">

⭐ _**MarcoPolo Team**_ ⭐

[_**Alibaba Group**_](https://www.qianwenai.com/)

📝 [**Paper**](https://arxiv.org/abs/PLACEHOLDER) | 🤗 [**Dataset**](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch) | 🔧 [**Framework**](https://github.com/AIDC-AI/Marco-DeepWideSearch-Agent/Table-as-Search)

</div>

---

## 📌 Overview

**Table-as-Search** is a production-ready agent framework designed to tackle **deep and wide information seeking tasks** that require both:
- 🔍 **Deep reasoning** over multi-hop retrieval 
- 🌐 **Wide-scale** information collection across multiple entities

This framework significantly outperforms Single-Agent, Multi-Agent ReAct baselines in challenging **Deep and Wide Info-Seeking** scenarios. The framework implements a hierarchical multi-agent architecture with specialized agents for different search strategies, making it suitable for real-world applications like market analysis, competitive intelligence, and business development research.

<div align="center">
  <img src="./assets/overview.png" alt="Table-as-Search Framework Architecture" width="800">
  <p><em>Figure 1: Hierarchical multi-agent architecture of Table-as-Search framework</em></p>
</div>

### 🎯 Performance Highlights

Our framework demonstrates significant advantages over baseline approaches, especially as task difficulty increases. The "**scissor gap effect**" shows that Table-as-Search maintains superior performance even on the most challenging queries.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./assets/Difficulty_vs_Performance_Combined_Gemini-2.5-Flash_All_189_samples_Fair_Comparison_01.png" alt="WideSearch Performance" width="450">
        <p><em>Performance on WideSearch (189 samples)</em></p>
      </td>
      <td align="center">
        <img src="./assets/DeepSearch_BrowseCompZH_Difficulty_vs_Performance_Gemini-2.5-Flash_01.png" alt="DeepSearch Performance" width="450">
        <p><em>Performance on DeepSearch (BrowseComp-ZH)</em></p>
      </td>
    </tr>
  </table>
  <p><em>Figure 2: Performance comparison across difficulty levels. <strong>Table-as-Search</strong> maintains consistent advantage over baselines, with the performance gap widening on harder tasks (the "scissor gap effect").</em></p>
</div>

**Key Observations**:
- ✅ **Consistent superiority** across all difficulty levels
- ✅ **Widening advantage** on harder tasks (Hard/Very Hard categories)
- ✅ **Robust performance** on both WideSearch and DeepSearch benchmarks
- ✅ **All comparisons use Gemini 2.5 Flash** for fair evaluation

---

## 🎯 Key Features

### 🏗️ Architecture

- **Hierarchical Multi-Agent System**
  - 🎭 **Main Agent**: Orchestrates the overall task decomposition and result aggregation
  - 📊 **Tabular Search Agent**: Handles wide-scale entity collection across multiple candidates
  - 🔎 **Deep Search Agent**: Performs detailed attribute extraction for specific entities

- **Advanced Tool Integration**
  - 🌐 Google Search with rate limiting and retry mechanisms
  - 📄 Web page visiting and content extraction (powered by Jina AI)
  - 🗄️ Database-backed table management (MongoDB)

### 🚀 Performance & Reliability

- ⚡ **Parallel Processing**: Multi-worker support for batch inference
- ⏱️ **Timeout Management**: Process-level timeout control for reliable execution
- 🔄 **Retry Mechanism**: Automatic retry on failures with configurable limits
- 💾 **State Persistence**: Resume from where you left off with database-backed checkpointing
- 📊 **Rich Monitoring**: Detailed logging and progress tracking with Rich library

---

## 📁 Repository Structure

```
Table-as-Search/
├── run_widesearch_inference.py       # Single-task WideSearch inference
├── run_deepsearch_inference.py       # Single-task DeepSearch inference  
├── run_widesearch_batch_inference.py # Batch WideSearch with parallel processing
├── run_deepsearch_batch_inference.py # Batch DeepSearch with parallel processing
│
├── benchmark/                        # Benchmark datasets
│   ├── widesearch.jsonl              # WideSearch benchmark
│   ├── gaia-text-only.jsonl          # GAIA benchmark subset
│   └── browsecomp-zh-decrypted.json  # BrowseComp-zh benchmark
│
├── tools/                            # Core tool implementations
│   ├── google_search_tool.py         # Google Search with rate limiting
│   ├── jina_visit.py                 # Web page visiting and extraction
│   ├── db_table_code_v2.py           # Database table management
│   ├── dataloader.py                 # Benchmark data loading utilities
│   ├── context_summary_toolcalling_agent.py  # Context summarization agent
│   └── env_loader.py                 # Custom .env file loader (license-safe)
│
├── prompts/                          # Agent prompt templates
│   ├── widesearch_prompts/           # Prompts for WideSearch tasks
│   │   ├── main_agent_prompt_v4.py
│   │   ├── tabular_search_prompt_v4.py
│   │   └── deep_search_prompt_v4.py
│   └── deepsearch_prompts/           # Prompts for DeepSearch tasks
│       ├── main_agent_prompt_v3_multi_condition.py
│       ├── tabular_search_agent_prompt_v3_multi_condition.py
│       └── deep_search_agent_prompt_v3_multi_condition.py
│
├── patch/                            # Custom patches and enhancements
│   ├── openai_sever_model.py         # Enhanced OpenAI-compatible model
│   ├── monitoring.py                 # Advanced logging and monitoring
│   └── utils.py                      # Utility functions
│
├── scripts/                          # Execution scripts
│   ├── widesearch/
│   │   └── run_ws_gemini_2.5_flash.sh
│   └── deepsearch/
│       └── run_bczh_gemini_2.5_flash.sh
│
└── requirements.txt                  # Python dependencies
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- MongoDB (for database-backed table management)
- API Keys listed in `.env` file:
  - OpenAI API key (or compatible LLM API)
  - Google Search API key (Serper or similar)
  - Jina AI API key (for web page visiting)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AIDC-AI/Marco-DeepWideSearch-Agent
cd Marco-DeepWideSearch-Agent/Table-as-Search
```

### Step 2: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the `Table-as-Search` directory:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_BASE_URL=https://api.openai.com/v1

# Google Search API Configuration
SEARCH_API_KEY=your_serper_api_key_here
SEARCH_API_BASE=XXXXXX

# Jina AI Configuration (for web page visiting)
JINA_KEYS_FILE=path/to/jina_keys.txt  # Optional: file with multiple Jina API keys
```

---

## 🚀 Quick Start

### Running Single-Task Inference

#### WideSearch Task

```bash
python run_widesearch_inference.py \
    --query "List the top 10 programming languages in 2025 with their creators, year of creation, and main use cases" \
    --instance-id "test_001" \
    --main-model-id "gpt-4o" \
    --tabular-model-id "gpt-4o" \
    --deep-model-id "gpt-4o" \
    --output-dir ./outputs/widesearch \
    --db-name widesearch_test
```

#### DeepSearch Task (BrowseComp-zh)

```bash
python run_deepsearch_inference.py \
    --query "找出2024年诺贝尔物理学奖得主的详细信息，包括获奖理由、主要研究领域和代表性论文" \
    --instance-id "deep_001" \
    --main-model-id "gpt-4o" \
    --tabular-model-id "gpt-4o" \
    --deep-model-id "gpt-4o" \
    --output-dir ./outputs/deepsearch \
    --db-name deepsearch_test
```

### Running Batch Inference

For evaluating on benchmark datasets with parallel processing, please refer to the scripts under [scripts](scripts).

---

## 📊 Key Parameters

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--main-model-id` | Main orchestration model | Required |
| `--tabular-model-id` | Model for wide-scale entity search | Same as main |
| `--deep-model-id` | Model for deep attribute extraction | Same as main |

### Execution Control

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-workers` | Number of parallel workers | 1 |
| `--timeout-seconds` | Timeout per task (seconds) | 3600 |
| `--skip-completed` | Skip already completed tasks | False |
| `--clear-db` | Clear database before running | False |
| `--start-idx` | Start index for batch processing | 0 |
| `--end-idx` | End index for batch processing | All |

### Agent Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--main-max-steps` | Max steps for main agent | 30 |
| `--tabular-max-steps` | Max steps for tabular agent | 20 |
| `--deep-max-steps` | Max steps for deep agent | 15 |
| `--max-tool-call-retries` | Max retries on tool failures | 3 |

---

## 🔧 Advanced Usage

### Using the Database-Backed Table System

```python
from tools.db_table_code_v2 import DBTableCodeToolInterface
from pymongo import MongoClient

# Initialize MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DATABASE")]

# Create table tool
table_tool = DBTableCodeToolInterface(
    db=db,
    name_prefix="task_001",
    description="Manage structured tables for information collection"
)

# Use in agent
agent = ToolCallingAgent(
    model=model,
    tools=[table_tool],
    max_steps=20
)
```

## 🧩 Architecture Details

### Hierarchical Multi-Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Agent                           │
│  • Task decomposition                                       │
│  • Sub-agent orchestration                                  │
│  • Result aggregation                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ Tabular Agent    │ │ Deep Agent       │
│ • Wide search    │ │ • Deep reasoning │
│ • Entity list    │ │ • Attribute      │
│ • Quick scan     │ │   extraction     │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌────────────────────┐
         │  Tool Ecosystem    │
         │  • Google Search   │
         │  • Web Visit       │
         │  • Table Manager   │
         │  • Summarization   │
         └────────────────────┘
```

### Agent Responsibilities

#### 🎭 Main Agent
- Analyzes the user query to determine search scope
- Decides when to use tabular vs. deep search strategies
- Manages information flow between sub-agents
- Aggregates results into final structured output

#### 📊 Tabular Search Agent
- Identifies and collects a comprehensive list of candidate entities
- Performs broad, shallow searches to maximize coverage
- Optimized for recall over precision

#### 🔎 Deep Search Agent
- Extracts detailed attributes for specific entities
- Performs multi-hop reasoning to find complex information
- Optimized for precision and completeness

---

## 📚 Related Projects

This framework is part of the **Marco Search Agent** project series:

- 📊 **[DeepWideSearch](../DeepWideSearch)**: Benchmark and evaluation metrics
- 🏷️ **[HSCodeComp](../HSCodeComp)**: Hierarchical rule application benchmark
- 🤖 **[our_smolagents](./our_smolagents)**: Custom agent framework (modified from HuggingFace smolagents)

---

## 🤝 Contributing

We welcome contributions in the following areas:

- 🔧 **Tool Implementations**: Add new search engines, data sources, or extraction tools
- 🎯 **Agent Strategies**: Improve task decomposition and orchestration logic
- 📊 **Evaluation**: Add new benchmarks or evaluation metrics
- 🐛 **Bug Fixes**: Report issues or submit fixes
- 📖 **Documentation**: Improve guides and examples

---

## 🛡️ License

This project is licensed under the **Apache-2.0 License**.

### Third-Party Components

- **our_smolagents**: Modified from [HuggingFace smolagents](https://github.com/huggingface/smolagents) (Apache-2.0)
- **Jina AI**: Web content extraction via Jina Reader API
- **MongoDB**: Database backend (Server Side Public License)

---

## 🙏 Acknowledgements

This project builds upon excellent open-source work:

- 🤗 **HuggingFace smolagents**: Core agent framework foundation
- 🌐 **Jina AI**: Web content extraction and reader API

We thank the respective teams for their contributions to the open-source community.

---

## 📧 Contact

For questions, issues, or collaboration:

- 👨‍💻 **Tian Lan**: [GitHub](https://github.com/gmftbyGMFTBY)
- 👨‍🔬 **Longyue Wang**: [Website](https://www.longyuewang.com/)

---

## ⚠️ Disclaimer

This framework is designed for research and evaluation purposes. When using in production:

- 🔒 Ensure proper API key management and security
- 💰 Monitor API usage and costs
- 🌐 Respect website terms of service and robots.txt
- 📊 Validate output quality before downstream use
- ⚖️ Comply with data privacy regulations in your jurisdiction

The datasets and benchmarks may contain publicly accessible web data. If you believe any content infringes on your rights, please contact us promptly.

---

## 📊 Citation

If you use our proposed Table-as-Search framework in your research, please cite 🤗:

```bibtex
@misc{lan2026tableassearchformulatelonghorizonagentic,
      title={Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion}, 
      author={Tian Lan and Felix Henry and Bin Zhu and Qianghuai Jia and Junyang Ren and Qihang Pu and Haijun Li and Longyue Wang and Zhao Xu and Weihua Luo},
      year={2026},
      eprint={2602.06724},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06724}, 
}
```
