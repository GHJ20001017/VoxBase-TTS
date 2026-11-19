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

启动训练：

```bash
bash scripts/train.sh configs/train.yaml
```

也可以直接运行并覆盖 epoch 数：

```bash
python train.py --config configs/train.yaml
bash scripts/train.sh configs/train.yaml --epochs 2
```

程序只支持单进程、单 GPU，不支持 DDP；不要使用 `torchrun` 启动。

## 推理：生成、对比与解码

### 单条 codec token 推理

推理需要训练输出目录中的 checkpoint、`backbone_config.json` 和 `tokenizer/`。默认使用旧的置信度阈值解码；通过 `--decoding-schedule cosine` 可以启用按每个码本置信度排序、逐步填充的 cosine 解码：

```bash
# 默认：旧 threshold 解码
bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '你好，欢迎使用 VoxBase-TTS。' \
  --num-frames 100 \
  --output generated_codes.pt

# cosine 解码：每轮按各码本的置信度排序并填充对应 quota
bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '你好，欢迎使用 VoxBase-TTS。' \
  --num-frames 100 \
  --decoding-schedule cosine \
  --max-inference-steps 16 \
  --output generated_codes_cosine.pt
```

推理从全 MASK 的 `[8, T]` codec 网格开始。`threshold` 会固定达到置信度阈值的位置，并在最后一轮强制填充剩余位置；`cosine` 会在每一轮按照每个码本内部的置信度排序，分别填充该码本的 quota。已接受的 token 不会在后续轮次被覆盖。

可选参考条件必须同时提供 codes 文件和对齐文本：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '这是要生成的目标文本。' \
  --num-frames 100 \
  --reference-codes reference_codes.pt \
  --reference-text '这是参考音频对应的文本。' \
  --decoding-schedule cosine \
  --confidence-threshold 0.8 \
  --max-inference-steps 16 \
  --device cuda --precision fp16 \
  --output generated_with_reference.pt
```

输出 `.pt` 中包含 `codes`、`confidence`、`below_threshold`、`forced_mask`、剩余 MASK 历史、每码本剩余历史、实际 schedule steps 和解码方式等诊断字段。生成的 `codes` 是形状 `[8, T]` 的整数 codec tokens，不是 WAV。

### 四条测试样本的旧解码与 cosine 对比

此前抽取的样本应按 `sample_000.npz`、`sample_025.npz`、`sample_050.npz`、`sample_075.npz` 放在同一目录中。每个样本需要包含 `target`、`target_text`、`reference_same`、`reference_same_text`、`reference_other` 和 `reference_other_text` 等字段。

```bash
python scripts/evaluate_inference.py \
  --checkpoint outputs/tts/best.pt \
  --samples-dir /path/to/source/codes \
  --output-dir outputs/inference_comparison \
  --device cuda
```

脚本会加载 checkpoint，针对 `same`、`other`、`none` 三种参考条件分别运行旧的 `threshold` 解码和 `cosine` 解码，并保存：

```text
outputs/inference_comparison/
├── codes/
│   ├── sample_000.npz
│   ├── sample_025.npz
│   ├── sample_050.npz
│   └── sample_075.npz
├── metrics.jsonl       # 每次解码的准确率、置信度、MASK 历史和耗时
├── settings.json
└── complete.json
```

每个输出 NPZ 会保留原始样本字段，并新增 `generated_threshold_*` 与 `generated_cosine_*` 码本。若原始样本包含旧解码结果，脚本还会记录 threshold 结果的逐元素一致性。

### codec tokens 重建 WAV

需要本地的 Mimi codec 解码器目录和 `soundfile`：

```bash
python scripts/decode_codes.py \
  --codes-dir outputs/inference_comparison/codes \
  --output-dir outputs/inference_comparison/audio \
  --mimi-model /path/to/mimi-diagnostics \
  --sample-rate 24000
```

脚本会读取目标码本、旧 threshold 结果和 cosine 结果，调用 `MimiModel` 重建 WAV，并生成包含时长、RMS、峰值和 SHA-256 的 `manifest.json`。生成的 WAV 和诊断输出属于本地实验产物，不应提交到仓库。
