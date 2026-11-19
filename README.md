<p align="center">
  <img src="assets/voxbase_tts_brand_transparent.png" alt="VoxBase-TTS" width="80%">
</p>

## 项目定位

---

> 在 [VoxBase-s2s](https://github.com/GHJ20001017/VoxBase-s2s) 中，我们曾围绕 LLM 探索 Speech-to-Speech 模型的训练与实现。沿着这条路线继续前行，我们希望将注意力延伸到基于离散 Codec Token 的 Masked TTS，尝试以更清晰、更直接的建模路径，逐步构建一个易于理解、训练与复现的 TTS 模型。

## 项目介绍

VoxBase-TTS 是一个面向学习、研究和工程验证的文本到语音实验项目。项目以 **Qwen3-0.6B** 为语言建模主干，将文本、可选的参考文本与参考语音 codec tokens 组织为统一序列，并通过 Masked Codec Token 预测学习目标语音的离散表示。

项目当前聚焦于 **codec token 建模与可复现训练流程**，而不是完整的端到端音频产品。仓库提供数据读取、模型训练、断点恢复、TensorBoard、离线 token 推理和测试代码，方便逐步理解从文本条件到离散语音表示生成的完整过程。

传统 TTS 通常在文本、声学特征和波形之间采用多阶段流水线；VoxBase-TTS 选择先建模离散音频 token，将每一帧的多个 codebook token 聚合到同一个 Transformer 位置，再由独立预测头恢复各 codebook 的类别。

这种设计的目标是：

- 复用 Qwen3 的文本表示和上下文建模能力；
- 使用 Masked Token Learning 训练离散语音表示，而不是逐 token 自回归生成；
- 通过可选的参考文本和参考 codec tokens 支持说话人或表达条件的实验；

## 模型结构

<p align="center">
  <img src="assets/voxbase-tts-overview.png" alt="VoxBase-TTS 模型概览" width="55%">
</p>

- **文本输入**：目标文本经过 Qwen tokenizer 和文本 embedding，不做 G2P 预处理。
- **参考条件**：可选的参考文本和对齐 codec tokens 会放在目标条件之前；参考部分不参与目标音频的 MASK 损失。
- **音频表示**：每个 codebook 使用独立的可训练 embedding 和分类头。设码本数量为 `Q`、类别数为 `K`，每张 embedding 表包含 `K + 1` 行，额外一行用于音频 MASK。
- **帧内聚合**：同一帧的 `Q` 个音频 embedding 求和并除以 `sqrt(Q)`，每帧的多个 codebook token 聚合为一个 Transformer 序列位置。
- **双向主干**：Qwen3 主干使用双向注意力，支持 `sdpa` 和 `eager` 两种 attention backend；不依赖 Flash Attention 2。
- **Masked 预测**：训练时只在真实目标 `Q × T` 位置中采样 MASK，不 MASK padding，也不要求整帧一起 MASK。MASK 比例会作为时间条件加入目标音频 embedding。
- **多码本输出**：共享主干特征经过 `RMSNorm → Linear → SiLU` 后，由每个 codebook 的独立分类头预测 `K` 个类别；各分类头共享特征，但不共享 logits，也不按 codebook 自回归。

## 环境安装

请先安装与本机 GPU 驱动匹配的 CUDA 版 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

## 启动训练

先下载数据集：

```bash
modelscope download --dataset ghjghj1017/tts-8cb \
  --local_dir /gpu3/guhj/VoxBase-TTS/dataset/tts_8q
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/train.sh configs/train.yaml
```

## 推理：cosine 解码

### 生成 cosine codec tokens

```bash
bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '你好，欢迎使用 VoxBase-TTS。' \
  --num-frames 100 \
  --decoding-schedule cosine \
  --max-inference-steps 16 \
  --output generated_codes_cosine.pt \
  --mimi-model /path/to/mimi-diagnostics
```

## 推理音频结果

![推理音频结果 1](https://github.com/user-attachments/assets/11df38c9-8baf-4bf5-9b14-4dd54b5818e1)

![推理音频结果 2](https://github.com/user-attachments/assets/723a7207-39e0-4a5c-8a06-0eae30ff4d0b)

![推理音频结果 3](https://github.com/user-attachments/assets/0d3ae8fb-2eda-407f-829d-1322ca7f340b)

## 项目总结

目前，VoxBase-TTS 已经能够通过逐步去噪的 Masked Codec Token 解码方式，将文本生成对应的语音；同时也能够在一定程度上引入参考音频及其文本条件，用于音色克隆和语音生成控制。项目仍有以下方向需要继续完善：

1. **目标帧数预测**：当前在测试集上可以直接获得文本对应的目标音频帧数，而真实推理时仍需要手动指定帧数。后续需要加入目标音频帧数估计模型，否则语速、停顿和整体时长可能难以准确控制。
2. **音质与码本容量**：当前模型使用 8 个码本，受离散表示容量限制，音频重建时仍可能损失部分细节，表现为一定程度的模糊。增加码本数量有望保留更多音频信息、提升重建音质，但也会带来更高的显存开销和训练难度。
3. **后训练与性能优化**：后续还需要引入后训练等优化方法，进一步提升生成质量、稳定性以及参考音频条件下的音色一致性。

## ⚖️ 开源协议

本项目采用 MIT License 开源协议。
