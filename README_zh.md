<p align="center">
    <img src="assets/logo_1.png" width="250" style="margin-bottom: 0.2;"/>
<p>

# 🍓 Marco DeepResearch: 迈向真实场景的高效智能体

[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<img src="https://img.shields.io/github/issues/AIDC-AI/Marco-Search-Agent?color=red" alt="Open Issues">
<img src="https://img.shields.io/github/issues-closed/AIDC-AI/Marco-Search-Agent?color=green" alt="Closed Issues">
<img src="https://img.shields.io/github/stars/AIDC-AI/Marco-Search-Agent?color=yellow" alt="Stars"> 

<div align="center">

🍓 [_**阿里巴巴集团**_](https://www.qianwenai.com/) 🍓

📝  [**HSCodeComp 论文**](https://arxiv.org/abs/2510.19631) | 📝  [**DeepWideSearch 论文**](https://arxiv.org/abs/2510.20168) | 🤗  [**HSCodeComp 数据集**](https://huggingface.co/datasets/AIDC-AI/HSCodeComp) | 🤗  [**DeepWideSearch 数据集**](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch)

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## 简介

**Marco DeepResearch** 是阿里巴巴集团推出的综合性研究计划，通过具有挑战性的基准测试和实际应用推动真实世界的 AI 智能体能力发展。我们的工作致力于缩小 AI 智能体与人类专家之间的差距，揭示并解决领域特定推理、层级规则应用和大规模信息检索中的关键局限。

<div align="center">
  <img src="assets/Timeline_2.png" alt="Marco DeepResearch 发展路线图" width="800">
</div>

### 🎯 核心成果

我们推出了一系列基准测试、框架和优化方法，从对真实世界部署至关重要的基础维度评估和推进智能体能力：

- **🏆 HSCodeComp**: 测试层级规则应用能力，**人类专家表现 95.0%** vs. **最佳 AI 46.8%** (SmolAgent + GPT-5 VLM)
- **🏆 DeepWideSearch**: 挑战深度与广度结合的信息检索，平均 **414 个信息单元**，**4.21 步推理深度**
- **🏆 Table-as-Search**: 生产级层级多智能体框架，在挑战性基准上展示**"剪刀差效应"**
- **🏆 UMEM**: 通过联合优化记忆提取和管理，避免"死记硬背陷阱"的自进化记忆系统

这些基准测试和框架揭示并解决了当前 AI 系统在以下方面的根本差距：
- 垂直领域（关税、法律、医疗、税务）中的复杂层级决策
- 同时进行大规模探索和深度多跳推理
- 结构化信息组织和综合
- 可泛化的长期记忆，无过拟合的自我进化

---

## 🔥 新闻与更新

* **[2026-02]** 🎉 发布 **UMEM（统一记忆提取与管理）** - 一个联合优化提取和管理以实现可泛化智能体记忆的自进化记忆框架。
* **[2026-02]** 🎉 发布 **Table-as-Search（表格即搜索）** - 针对复杂 Agentic Search 任务的结构化规划策略。
* **[2025-02]** 🏆 **DeepWideSearch**：
  - **[A-MapReduce](https://arxiv.org/pdf/2602.01331)** 采用 DeepWideSearch 作为广域搜索系统的主要评估基准，实现 **79.09% 核心实体准确率**、**51.78% 列级 F1** 和 **4.43% 成功率**（开源框架中的最先进水平），为评估智能体搜索能力设立可复现的新标准
* **[2025-10]** 🔥 Marco DeepResearch 首次发布，包含 **DeepWideSearch** 和 **HSCodeComp** 两个基准测试。

---

## 🌟 Marco DeepResearch 实践应用

真实业务部署展示了我们的研究框架如何解决阿里巴巴集团业务场景中的关键挑战。

---

### 📊 招商 BD 智能化（真实场景复杂深宽搜索任务）

**挑战：** 招商 BD 任务同时需要**广度**（跨平台发现大量合格商家）和**深度**（从官网多跳提取联系方式）。在 [DeepWideSearch](Marco-DeepResearch-Family/DeepWideSearch/) 基准上，ReAct 类基线存在规划不清、状态混乱和覆盖缺口。

**我们的方案：Table-as-Search** — 将长视野搜索形式化为**表格补全**：显式状态跟踪、基于半填表格的清晰规划、以及宽表（表格）与深度（多跳）子智能体的层级编排。

**成果：** 在真实 BD 数据集上，Table-as-Search 在困难任务上实现 **40%+ 提升**（成功率 15.2% → 55.8%），实体召回率 89.3%（vs. 62.1%）、属性完整性 85.7%（vs. 58.4%）。已落地 BD 工作流，显著提升作业效率。

<div align="center">
  <img src="assets/TaS_exp_1.png" alt="Table-as-Search 性能" width="85%">
  <p><em><b>跨任务难度性能：</b>Table-as-Search（蓝）vs. Multi-Agent ReAct（橙）与 Single-Agent（灰）基线。</em></p>
</div>

---

### 🏷️ 跨境贸易 HSCode 分类

**问题：垂域下的层级规则应用**

根据不完整的产品信息（如来自 ERP 或商品目录）预测目的国 10 位 HS 编码及税率，需要**层级规则应用**：关税规则边界模糊、逻辑隐含，对智能体的精确应用构成挑战。任务定义与相关工作见基准论文 [HSCodeComp](assets/HSCodeComp.pdf)。

**我们的做法：先建基准，再做工具增强智能体**

我们首先建立 **HSCodeComp** 基准，发现当前先进智能体表现远逊于人类专家。随后设计以 Marco 为编排的智能体框架：(1) **多模态输入解析**（标题、属性、图片 → 规范化属性），(2) **检索增强推理**（Deep Search：历史标注、专家知识、海关裁定），(3) **工具化核验**（税率查询、章节注释、裁定校验），(4) **结构化输出**与可审计证据链。

**效果：相对基线明显提升，相对人类仍有较大差距**

在 10 位 HS 编码准确率上，Marco Agent 达到 **65.0%** Top-1，优于 GPT-5 系智能体（46.8%）、Agentorchestra（41.3%）和 Claude Sonnet 4（11.9%）。下图表明工具增强决策显著优于通用智能体；但与人类专家（95.0%）仍存在较大差距，**仍有很大提升空间**。

<div align="center">
  <img src="assets/HSCode_our_performance.png" alt="HSCode 基准效果" width="40%">
  <p><em><b>HSCodeComp 基准（10 位准确率）：</b>Marco Agent（65.0%）vs. 基线及人类专家（95.0%）。</em></p>
</div>

---

### 💬 客服智能化（自进化智能体）

**问题：规则细微且持续变化**

在电商商品审核场景中，规则多模态、细微且不断演化。当智能体判断与专家标注不一致（如将正品误判为「假货」）时，以往需 **3–5 天人工调优** 才能修正。

**我们的方案：自进化智能体 + UMEM**

**自进化智能体** 从智能体判断与专家标注的差距中学习：提取细粒度洞察（如「高端品牌需结合视觉水印核对『正品』描述」）并写入长期记忆。引擎是我们提出的 [**UMEM**（统一记忆抽取与管理）](./Marco-DeepResearch-Family/UMEM/)方法：将交互轨迹提炼为可执行、可泛化的洞察，而非简单检索历史。闭环为 **Action → Rewarding**（与 Ground Truth 对比、发现 Badcase）**→ Memory Extraction**（反思、生成候选规则）**→ Validation**（安全门控后更新 Memory 或重试）。

**效果：调优效率约 30–50 倍提升，质量同步提升**

全流程从 3–5 天压缩为 **约 10 分钟自主闭环**。自进化智能体相对人工调优基线在白底图审核上 **+11%**、短标题审核 **+2%**。在基准测试中，**UMEM** 在多种环境下均稳定优于 ReMem、Memp 等先进记忆基线。此外，我们也在其他推理基准上评估了 UMEM（见下图），大量实验表明 UMEM 能够学习高度可泛化的记忆并提升后续任务表现。

<div align="center">
  <img src="assets/benchmark_comparison_umem_Gemini-2-5-Flash.png" alt="UMEM 基准对比" width="85%">
  <p><em><b>UMEM vs. 基线</b>（如 Gemini 2.5 Flash）：UMEM 在各评估设置下均带来提升。</em></p>
</div>

---

## 📦 资源下载

### 数据集

| 基准测试 | HuggingFace | GitHub | 论文 |
|----------|-------------|--------|------|
| **HSCodeComp** | [🤗 AIDC-AI/HSCodeComp](https://huggingface.co/datasets/AIDC-AI/HSCodeComp) | [📁 HSCodeComp/data](Marco-DeepResearch-Family/HSCodeComp/data/test_data.jsonl) | [📝 arXiv](https://arxiv.org/abs/2510.19631) |
| **DeepWideSearch** | [🤗 AIDC-AI/DeepWideSearch](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch) | [📁 DeepWideSearch/data](Marco-DeepResearch-Family/DeepWideSearch/data/) | [📝 arXiv](https://arxiv.org/abs/2510.20168) |
| **Table-as-Search** | [🤗 Table-as-Search Paper](https://huggingface.co/papers/2602.06724) | [📁 Table-as-Search Codebase](Marco-DeepResearch-Family/Table-as-Search/) | [📝 arXiv](https://arxiv.org/abs/2602.06724) |
| **UMEM** | [🤗 UMEM Paper](https://huggingface.co/papers/2602.06724) | [📁 UMEM Codebase](Marco-DeepResearch-Family/UMEM/) | [📝 arXiv](https://arxiv.org/abs/2602.10652) |

---

## 🚀 快速开始

### 仓库结构

```
Marco-DeepResearch/
├── Marco-DeepResearch-Family/   # 所有项目的统一目录
│   ├── HSCodeComp/              # 层级规则应用基准
│   │   ├── data/                # 632 个专家标注的产品样本
│   │   ├── eval/                # 评估脚本
│   │   └── README.md
│   ├── DeepWideSearch/          # 深度与广度结合的信息检索基准
│   │   ├── data/                # 220 个复杂多跳查询
│   │   ├── eval/                # 评估脚本
│   │   ├── scripts/             # 批量评估工具
│   │   └── README.md
│   ├── Table-as-Search/         # 层级多智能体框架
│   │   ├── tools/               # 核心工具实现
│   │   ├── prompts/             # 智能体提示模板
│   │   └── README.md
│   ├── UMEM/                    # 自进化记忆系统
│   │   ├── verl/                # 核心源代码
│   │   ├── umem_scripts/        # 训练和评估脚本
│   │   └── README.md
│   ├── README.md                # 系列概览（英文）
│   └── README_zh.md             # 系列概览（中文）
├── assets/                      # 共享资源和可视化
└── README.md                    # 主项目 README
```

### 安装

每个项目都有自己的依赖。进入特定的项目目录：

```bash
# HSCodeComp
cd Marco-DeepResearch-Family/HSCodeComp
pip install -r requirements.txt

# DeepWideSearch
cd Marco-DeepResearch-Family/DeepWideSearch
pip install -r requirements.txt

# Table-as-Search
cd Marco-DeepResearch-Family/Table-as-Search
pip install -r requirements.txt

# UMEM
cd Marco-DeepResearch-Family/UMEM
pip install -r requirements.txt
pip install -e .
```

### 运行评估

**HSCodeComp**:
```bash
cd Marco-DeepResearch-Family/HSCodeComp
python eval/test_llm.py \
  --model_name your_model \
  --data_path data/test_data.jsonl \
  --output_path results/
```

**DeepWideSearch**:
```bash
cd Marco-DeepResearch-Family/DeepWideSearch
bash scripts/batch_eval.sh
```

**Table-as-Search**:
```bash
cd Marco-DeepResearch-Family/Table-as-Search
python run_widesearch_inference.py --query "your query" --instance-id "test_001"
```

**UMEM**:
```bash
cd Marco-DeepResearch-Family/UMEM
bash umem_scripts/run_eval.sh
```

详细的设置和使用说明，请参考：
- [HSCodeComp README](Marco-DeepResearch-Family/HSCodeComp/README.md) - 层级规则应用评估
- [DeepWideSearch README](Marco-DeepResearch-Family/DeepWideSearch/README.md) - 深广搜索评估
- [Table-as-Search README](Marco-DeepResearch-Family/Table-as-Search/README.md) - 框架使用和部署
- [UMEM README](Marco-DeepResearch-Family/UMEM/README.md) - 记忆系统训练和评估

---

## 🌟 Marco DeepResearch 系列

Marco DeepResearch 计划涵盖多个基准测试和框架，解决真实世界智能体系统中的不同挑战。访问我们的 [**Marco DeepResearch 系列**](Marco-DeepResearch-Family/README_zh.md) 目录了解每个项目的详细信息：

- **📑 [HSCodeComp](Marco-DeepResearch-Family/HSCodeComp/README.md)**: 电商领域的层级规则应用
- **🌐 [DeepWideSearch](Marco-DeepResearch-Family/DeepWideSearch/README.md)**: 深度与广度结合的智能体信息检索
- **📊 [Table-as-Search](Marco-DeepResearch-Family/Table-as-Search/README.md)**: 生产级层级多智能体框架
- **🧠 [UMEM](Marco-DeepResearch-Family/UMEM/README.md)**: 面向自进化智能体的统一记忆提取与管理

<div align="center">
  <a href="Marco-DeepResearch-Family/README_zh.md">
    <img src="https://img.shields.io/badge/探索完整系列-blue?style=for-the-badge&logo=read-the-docs" alt="探索 Marco DeepResearch 系列">
  </a>
</div>

---

## 👨🏻‍💻 致谢

主要贡献者来自阿里巴巴集团 AI团队（ATH-MaaS）。如有问题或合作意向，请联系：
- [Tian Lan](https://github.com/gmftbyGMFTBY)
- [Longyue Wang](https://www.longyuewang.com/)

**特别感谢**:
- **HSCodeComp**: 感谢人工关税专家的细致标注（时薪 >$34）
- **DeepWideSearch**: 基于 ByteDance-Seed 的开源 [WideSearch](https://github.com/ByteDance-Seed/WideSearch) 框架构建（MIT 许可证）

---

## 🛡️ 许可证

本项目采用 **Apache-2.0 许可证**。详见 [LICENSE](LICENSE)。

---

## ⚠️ 免责声明

我们的数据集使用公开可访问的数据源构建：
- **HSCodeComp**: 来自真实电商平台的产品数据
- **DeepWideSearch**: 基于 [BrowseComp](https://openai.com/index/browsecomp/)、[BrowseComp-ZH](https://arxiv.org/abs/2504.19314) 和 [WideSearch](https://github.com/ByteDance-Seed/WideSearch) 数据集

由于这些任务的复杂性和数据源的多样性，我们无法保证完全没有版权问题或不当内容。如果您认为有任何内容侵犯了您的权利或产生了不当内容，请联系我们以便及时解决。

---

## 📬 引用

如果您觉得我们的工作有用，请考虑引用：

```bibtex
@misc{yang2025hscodecomprealisticexpertlevelbenchmark,
      title={HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application}, 
      author={Yiqian Yang and Tian Lan and Qianghuai Jia and Li Zhu and Hui Jiang and Hang Zhu and Longyue Wang and Weihua Luo and Kaifu Zhang},
      year={2025},
      eprint={2510.19631},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.19631}, 
}

@misc{lan2025deepwidesearchbenchmarkingdepthwidth,
      title={DeepWideSearch: Benchmarking Depth and Width in Agentic Information Seeking}, 
      author={Tian Lan and Bin Zhu and Qianghuai Jia and Junyang Ren and Haijun Li and Longyue Wang and Zhao Xu and Weihua Luo and Kaifu Zhang},
      year={2025},
      eprint={2510.20168},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.20168}, 
}

@misc{lan2026tableassearchformulatelonghorizonagentic,
      title={Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion}, 
      author={Tian Lan and Felix Henry and Bin Zhu and Qianghuai Jia and Junyang Ren and Qihang Pu and Haijun Li and Longyue Wang and Zhao Xu and Weihua Luo},
      year={2026},
      eprint={2602.06724},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06724}, 
}

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
