# MiniMind-64M 大模型全流程复现与工程实践记录

## 📖 项目简介

本项目基于开源项目 [MiniMind](https://github.com/jingyaogong/minimind)（类 Llama3 的轻量级 Decoder-only 架构）进行完整复现。

我是一名 211 计算机专业的学生，求职方向为大模型算法/工程实习生。为了深入理解大模型底层原理与训练全流程，我做了两件事：

1. **底层原理学习与代码手写**：我系统学习了 Transformer 架构，并手写实现了多头自注意力、RoPE、GQA、KV Cache、RMSNorm 等核心模块（学习代码未包含在本仓库中，但已充分理解其数学推导与 PyTorch 实现）。
2. **全流程训练与工程排错**：基于官方代码，在本地消费级显卡（RTX 3060）和云端（RTX 3090）上独立完成了从预训练、指令微调、LoRA 到人类偏好对齐（DPO）的完整训练闭环，并解决了多个训练过程中的真实 Bug。

本仓库记录了我的训练环境、Loss 数据、工程问题排查过程以及最终模型权重。

## 💻 复现环境与工具链

- **操作系统**：Windows 11
- **开发工具**：VS Code、Anaconda、Git
- **硬件设备**：NVIDIA GeForce RTX 3060 (12GB)
- **核心框架**：PyTorch、Transformers、Datasets、uv
- **训练平台**：AutoDL 云服务器（RTX 3090 租用） + 本地 PC 辅助训练

## 📊 训练阶段与 Loss 记录

| 阶段 | 训练步数 | 初始 Loss | 最终 Loss | 备注 |
|------|----------|-----------|-----------|------|
| 预训练 (Pretrain) | 35,300 步 | 7.50 | 2.60 | 云端 RTX 3090，采用 Early Stopping |
| 指令微调 (SFT) | 7,300 步 | 2.00 | 1.60 ~ 1.90 | 本地 RTX 3060，官方精简数据集 |
| LoRA 微调 | 630 步 | 2.00 | 1.70 ~ 1.80 | 医疗数据集，参数量占比 0.61% |
| 偏好对齐 (DPO) | 15,000 步 | 0.80 | 0.50 ~ 0.70 | 1 epoch，双模型对齐训练 |

> 由于训练日志未导出图表，以上数据均来自终端记录。

## 🐛 核心工程问题排查与解决

在复现过程中，我遇到并解决了以下真实问题，这让我对大模型训练机制有了更深刻的理解：

### 1. SFT 阶段模型只“重写”不“回答”
- **现象**：模型输出高度重复，且倾向于复述用户提问，而不是给出答案。
- **排查**：检查数据加载逻辑，发现计算 Loss 时未正确进行 **Loss Mask**，导致用户 Prompt 也被当作目标拟合。
- **解决**：屏蔽用户部分的 Loss 后，模型收敛正常，泛化能力显著提升。

### 2. LoRA 微调引发“灾难性遗忘”
- **现象**：在医疗数据集上微调后，模型原有的通用对话能力大幅衰退。
- **分析**：64M 小模型在注入陌生领域知识时极易发生灾难性遗忘。
- **决策**：确立了后续引入 RAG 替代纯参数记忆的技术路线，将模型能力与知识解耦。

### 3. DPO 对齐训练中的 Reward Hacking 风险
- **现象**：DPO 训练 Loss 下降过快，生成文本极短且句式僵化。
- **分析**：模型学会了迎合偏好数据中的“简短回答”偏好，而放弃了完整逻辑表达。
- **决策**：在 Loss 稳定时采用早停策略（Early Stopping），保留泛化能力最好的 Checkpoint，没有盲目跑完 10 个 epoch。

## 📦 模型权重获取

由于 GitHub 对大文件（>100MB）有限制，完整的 `.pth` 权重文件未直接上传。

- **模型权重（Pretrain / SFT / DPO / LoRA）**：
  - 123 云盘链接：[点击下载](https://4002001500.share.123pan.cn/123pan/aFjeMh-6NiHd?pwd=UU5g)
  - 提取码：`UU5g`
- **数据集**：请参考 MiniMind 官方文档，自行下载 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`、`dpo.jsonl` 等文件，放置到 `dataset/` 目录下。

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 SFT 训练（示例）
cd trainer
python train_full_sft.py

# 3. 启动 DPO 训练（示例）
python train_dpo.py

# 4. 终端推理测试
python eval_llm.py --weight full_sft
python eval_llm.py --weight dpo
