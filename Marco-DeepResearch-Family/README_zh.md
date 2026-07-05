# 🌟 Marco DeepResearch 系列

<div align="center">

🍓 [_**阿里巴巴集团**_](https://www.qianwenai.com/) 🍓

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## 概述

**Marco DeepResearch 系列**涵盖了一套全面的基准测试和框架，解决真实世界智能体系统中的不同挑战。从层级规则应用到大规模信息检索和自进化记忆，我们的工作连接了基础研究与实际部署需求。

---

## 📑 [HSCodeComp](../HSCodeComp/README.md)

**评估电商领域的层级规则应用**

<div align="center">
  <img src="../assets/HSCODE-workflow.png" alt="HSCodeComp 工作流程" width="100%">
</div>

### 挑战

应用包含模糊语言和隐含决策逻辑的复杂层级关税规则，从嘈杂的产品列表中预测 10 位协调制度（HS）编码。

### 核心数据

- 📊 **632** 个专家标注的产品，涵盖 27 个 HS 章节和 32 个电商类别
- 🎯 从嘈杂的产品列表中预测 **10 位** HS 编码
- 👨‍💼 **人类专家表现**: 95.0% 准确率
- 🤖 **最佳 AI 系统** (SmolAgent + GPT-5 VLM): 46.8% 准确率
- ⏱️ **人工标注成本**: 领域专家时薪 >$34

### 影响

揭示了智能体在垂直领域（法律、医疗、海关、税务）处理层级推理能力的关键局限。

### 资源

- 📝 [论文](https://arxiv.org/abs/2510.19631) | 🤗 [数据集](https://huggingface.co/datasets/AIDC-AI/HSCodeComp) | 📘 [文档](HSCodeComp/README.md)

---

## 🌐 [DeepWideSearch](../DeepWideSearch/README.md)

**评估深度与广度结合的智能体信息检索**

<div align="center">
  <img src="../assets/DWS-workflow.png" alt="DeepWideSearch 工作流程" width="100%">
</div>

### 挑战

同时发现大量实体（广度）并对每个实体执行深度多跳推理（深度），生成结构化表格（实体 × 属性）。

### 核心数据

- 📊 **220** 个需要结构化表格输出（实体 × 属性）的复杂查询
- 🔢 平均每个答案包含 **414** 个信息单元
- 🧠 平均 **4.21** 步推理深度（多跳步骤）
- 🌍 双语评估：**中英文**
- 🤖 **最佳 AI 系统** (WebSailor + Claude Sonnet 4): 2.39% 成功率
- 📈 **被 A-MapReduce 采用**（复旦大学）作为主要评估基准

### 影响

证明即使是最先进的智能体也难以在大规模场景下结合广度探索和深度推理。

### 资源

- 📝 [论文](https://arxiv.org/abs/2510.20168) | 🤗 [数据集](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch) | 📘 [文档](DeepWideSearch/README.md)

---

## 📊 [Table-as-Search](../Table-as-Search/README.md)

**深度与广度结合的层级多智能体框架**

<div align="center">
  <img src="Table-as-Search/assets/overview.png" alt="Table-as-Search 框架" width="100%">
</div>

### 挑战

构建一个生产级智能体框架，同时擅长深度多跳推理和大规模多实体信息收集。

### 核心创新

层级多智能体架构，采用专门的搜索策略，显著优于单智能体和多智能体 ReAct 基线。

### 性能亮点

<div align="center">
  <table>
    <tr>
      <td width="50%">
        <img src="Table-as-Search/assets/Difficulty_vs_Performance_Combined_Gemini-2.5-Flash_All_189_samples_Fair_Comparison_01.png" alt="WideSearch 结果" width="100%">
        <p align="center"><em>WideSearch 基准测试（189 样本）</em></p>
      </td>
      <td width="50%">
        <img src="Table-as-Search/assets/DeepSearch_BrowseCompZH_Difficulty_vs_Performance_Gemini-2.5-Flash_01.png" alt="DeepSearch 结果" width="100%">
        <p align="center"><em>DeepSearch 基准测试（BrowseComp-ZH）</em></p>
      </td>
    </tr>
  </table>
  <p><em>性能对比展示 <strong>"剪刀差效应"</strong>：随着任务难度增加，Table-as-Search 保持优越性能，且优势不断扩大。</em></p>
</div>

### 架构组件

- 🎭 **主智能体**: 任务分解和编排
- 📊 **表格搜索智能体**: 大规模实体收集
- 🔎 **深度搜索智能体**: 多跳属性提取
- 🛠️ **工具生态**: Google 搜索、网页访问、数据库支持的表格管理

### 关键成果

- ✅ 在所有难度级别（简单 → 非常困难）上**持续保持优势**
- ✅ 在较难任务上**性能差距扩大**，相比基线提升显著
- ✅ **生产级就绪**，具备并行处理、超时管理和状态持久化
- ✅ **公平比较**，所有方法均使用 Gemini 2.5 Flash

### 资源

- 📝 [论文](https://arxiv.org/abs/2602.06724) | 🔧 [框架代码](Table-as-Search/) | 📘 [文档](Table-as-Search/README.md)

---

## 🧠 [UMEM](../UMEM/README.md)

**统一记忆提取与管理：面向泛化的自进化记忆系统**

<div align="center">
  <img src="UMEM/assets/umem_intro.png" alt="UMEM 框架" width="100%">
</div>

### 挑战

构建自进化智能体记忆系统，避免**"死记硬背陷阱"**——防止积累实例特定的捷径和噪声，导致泛化能力退化。

### 核心创新

通过学习的 Mem-Optimizer 策略**联合优化**记忆提取和管理，明确优化跨查询的泛化能力。

<div align="center">
  <img src="UMEM/assets/umem_overview.png" alt="UMEM 架构" width="100%">
</div>

### 核心组件

- 🔒 **冻结执行器**: 使用检索记忆解决任务的 LLM/智能体
- 💾 **可进化记忆库**: 随时间更新的非参数键值存储
- 🎯 **Mem-Optimizer 策略**: 输出结构化记忆操作（ADD/UPDATE）的学习策略

### 技术亮点

- 🔬 **语义邻域建模（SNM）**: 构建语义邻域以在相关查询间评估记忆更新
- 📊 **边际效用奖励**: 优化泛化的邻域级奖励机制
- 🔄 **GRPO + 在线进化**: 训练期间将最佳奖励操作提交到记忆库
- 📝 **严格 XML 模式**: 强制格式约束以防止答案泄露和幻觉

### 性能表现

- ✅ 通过邻域级优化**防止单实例过拟合**
- ✅ **可泛化的记忆**能够跨语义相关查询迁移
- ✅ 在保持正确性的同时通过轨迹长度缩减实现**效率提升**
- ✅ 流式评估协议中的**累积性能改进**

<div align="center">
  <img src="UMEM/assets/cumulative_curves.png" alt="UMEM 性能" width="70%">
  <p><em>累积性能展示通过自进化实现的持续改进</em></p>
</div>

### 资源

- 📝 论文（即将发布）| 🔧 [框架代码](UMEM/) | 📘 [文档](UMEM/README.md)

---

## 🔗 相关资源

### 基准测试与数据集

| 项目 | HuggingFace | GitHub | 论文 |
|------|-------------|--------|------|
| **HSCodeComp** | [🤗 数据集](https://huggingface.co/datasets/AIDC-AI/HSCodeComp) | [📁 数据](HSCodeComp/data/) | [📝 arXiv](https://arxiv.org/abs/2510.19631) |
| **DeepWideSearch** | [🤗 数据集](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch) | [📁 数据](DeepWideSearch/data/) | [📝 arXiv](https://arxiv.org/abs/2510.20168) |
| **Table-as-Search** | — | [📁 代码](Table-as-Search/) | [📝 arXiv](https://arxiv.org/abs/2602.06724) |
| **UMEM** | 即将发布 | [📁 代码](UMEM/) | 即将发布 |

### 基准测试采用情况

我们的基准测试正在被顶尖研究机构积极使用：

- **🎓 A-MapReduce（复旦大学）**: 采用 **DeepWideSearch** 作为广域智能体搜索系统的主要评估基准，实现了 79.09% 核心实体准确率和 4.43% 成功率。[📄 论文](https://arxiv.org/pdf/2602.01331)

---

## 📊 性能总结

### 层级规则应用
- **HSCodeComp**: 人类专家（95.0%）与最佳 AI（46.8%）之间存在 48.2% 的性能差距
- 展示了在垂直领域改进层级推理的关键需求

### 深宽信息检索
- **DeepWideSearch**: 当前 SOTA 在复杂结构化检索上仅达到 2.39% 成功率
- **Table-as-Search**: 通过表格中心设计在困难任务上提升 40%+

### 自进化记忆
- **UMEM**: 防止死记硬背同时保持累积性能改进
- 可泛化记忆能够跨语义邻域迁移

---

## 🎯 研究方向

我们的基准测试和框架系列识别并解决了关键挑战：

1. **层级决策制定**: 超越平面推理，处理嵌套规则和约束
2. **规模与深度权衡**: 平衡广度探索与深度多跳推理
3. **结构化信息管理**: 有效组织大规模检索结果
4. **可泛化长期记忆**: 从经验中学习而不过拟合实例

---

## 👨🏻‍💻 联系方式

主要贡献者来自阿里巴巴集团 AI团队（ATH-MaaS）。如有问题或合作意向，请联系：
- [Tian Lan](https://github.com/gmftbyGMFTBY)
- [Longyue Wang](https://www.longyuewang.com/)

---

## 📬 引用

如果您觉得我们的工作有用，请引用相关论文：

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

@misc{lan2026tableassearch,
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
