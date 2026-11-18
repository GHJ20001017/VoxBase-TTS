# VoxBase-TTS

<p align="center">
  <img src="assets/voxbase-tts-overview.png" alt="VoxBase-TTS 模型概览" width="55%">
</p>

<p align="center"><strong>基于离散 Codec Token 的 Masked TTS 实验项目</strong></p>

## 项目定位

VoxBase-TTS 是一个面向学习、研究和工程验证的文本到语音实验项目。项目以 **Qwen3-0.6B** 为语言建模主干，将文本、可选的参考文本与参考语音 codec tokens 组织为统一序列，并通过 Masked Codec Token 预测学习目标语音的离散表示。

项目当前聚焦于 **codec token 建模与可复现训练流程**，而不是完整的端到端音频产品。仓库提供数据读取、模型训练、断点恢复、TensorBoard、离线 token 推理和测试代码，方便逐步理解从文本条件到离散语音表示生成的完整过程。

> 当前推理输出是 codec token 的 `.pt` 文件，不是 WAV。仓库尚未集成 codec 编码器/解码器、自动时长预测或完整的主观音质评测流程。

## 项目介绍

传统 TTS 通常在文本、声学特征和波形之间采用多阶段流水线；VoxBase-TTS 选择先建模离散音频 token，将每一帧的多个 codebook token 聚合到同一个 Transformer 位置，再由独立预测头恢复各 codebook 的类别。

这种设计的目标是：

- 复用 Qwen3 的文本表示和上下文建模能力；
- 使用 Masked Token Learning 训练离散语音表示，而不是逐 token 自回归生成；
- 通过可选的参考文本和参考 codec tokens 支持说话人或表达条件的实验；
- 保持训练、验证、断点恢复和指标记录过程可检查、可复现。

当前实现仍属于实验阶段。token 准确率只能反映离散类别的命中情况，不能直接等价为语音自然度、音色相似度或可懂度。

## 模型结构

上方概览图用于概括整体数据流，实际实现以代码为准：

- **文本输入**：目标文本经过 Qwen tokenizer 和文本 embedding，不做 G2P 预处理。
- **参考条件**：可选的参考文本和对齐 codec tokens 会放在目标条件之前；参考部分不参与目标音频的 MASK 损失。
- **音频表示**：每个 codebook 使用独立的可训练 embedding 和分类头。设码本数量为 `Q`、类别数为 `K`，每张 embedding 表包含 `K + 1` 行，额外一行用于音频 MASK。
- **帧内聚合**：同一帧的 `Q` 个音频 embedding 求和并除以 `sqrt(Q)`，因此 32q 不会让每帧占用 32 个 Transformer 序列位置。
- **双向主干**：Qwen3 主干使用双向注意力，支持 `sdpa` 和 `eager` 两种 attention backend；不依赖 Flash Attention 2。
- **Masked 预测**：训练时只在真实目标 `Q × T` 位置中采样 MASK，不 MASK padding，也不要求整帧一起 MASK。MASK 比例会作为时间条件加入目标音频 embedding。
- **多码本输出**：共享主干特征经过 `RMSNorm → Linear → SiLU` 后，由每个 codebook 的独立分类头预测 `K` 个类别；各分类头共享特征，但不共享 logits，也不按 codebook 自回归。

### 图示说明

图中方块和箭头是结构示意，不代表实际 tokenizer 输出、音频帧率或 codec 的具体实现细节。要得到可播放音频，还需要接入与训练数据兼容的 codec 解码器。

## 项目结构

```text
VoxBase-TTS/
├── assets/
│   └── voxbase-tts-overview.png  # 模型概览图
├── configs/train.yaml             # 模型、数据、训练和 MASK 配置
├── data/tts_data.py               # Parquet 读取、参考配对和 batch padding
├── model/
│   ├── bidirectional_qwen3.py     # Qwen3 双向注意力主干
│   └── VoxBase_TTS.py             # 音频 embedding、MASK 训练和迭代生成
├── train.py                       # 训练、验证、checkpoint 和日志
├── infer.py                       # 离线生成 codec tokens
├── scripts/
│   ├── train.sh                   # 训练启动入口
│   └── infer.sh                   # 推理启动入口
├── tests/                         # 单元测试和可选真实数据测试
├── requirements.txt
└── README.md
```

## 环境安装

请先安装与本机 GPU 驱动匹配的 CUDA 版 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

项目依赖 `transformers==5.14.1`，使用该版本提供的 Qwen3 双向 mask API。依赖文件中声明 `torch>=2.6`，但并不代表所有 PyTorch、CUDA 和 Transformers 组合都经过验证；此前验证过的组合包括 PyTorch 2.11.0 和 CUDA 13.0。

### 95 服务器环境

项目曾在 95 服务器的独立 venv 中验证：

```bash
source /gpu3/guhj/envs/voxbase-tts/bin/activate
cd /gpu3/guhj/VoxBase-TTS
```

该环境不是 Conda 环境。已验证 Python 3.12.12、PyTorch 2.11.0+cu130、Transformers 5.14.1、NumPy 2.4.3、TensorBoard 2.21.0 和 tqdm 4.70.1。A800 上完成过小模型 FP32、FP16、BF16 前向/反向及 AdamW 更新检查；这不等于完整模型长序列训练的显存或音质保证。

## 数据集

数据需要预先转换为包含离散 codec tokens 的 Parquet 文件。训练、验证和测试分别使用独立 split 目录：

```text
tts_dataset/
├── train/
│   └── .../*.parquet
├── validation/
│   └── .../*.parquet
└── test/
    └── .../*.parquet
```

读取器会递归搜索传入 split 根目录中的 Parquet 文件，并排除 `indexes/`；**不会自动追加或优先读取 `shards/`**。数据按 row group 惰性读取，并使用有界缓存，不会一次性把全部 codes 加载到内存。

### 核心字段

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 样本标识 |
| `text` | 音频对应文本 |
| `codes` | 二维整数数组，形状必须为 `[Q, T]` |
| `num_codebooks` | 码本数量 `Q` |
| `num_frames` | 音频 token 帧数 `T` |
| `dataset` | 参考配对时使用的数据集标识 |
| `speaker_id` | 参考配对时使用的说话人标识 |

有效 codec 索引范围为 `[0, K)`。原始数据不应写入音频 MASK 或 padding；数据中的 `Q` 必须与模型配置一致。

### 参考样本配对

`reference_prob` 控制使用参考条件的概率。参考样本仅在同一 split 内配对，并按 release、dataset、speaker 和码本数量分组；候选样本会排除相同 `sample_id`。

缺少有效 speaker、没有合适候选，或候选不满足码本条件时，样本不使用参考条件，不会退化为自身样本或其他说话人的样本。即使 `reference_prob: 1.0`，也不表示每条样本一定存在参考样本。

## 训练

以 [configs/train.yaml](configs/train.yaml) 为起点，并根据实际机器修改数据路径和输出路径。仓库中的数据路径只是开发环境示例，不代表数据已经随代码发布。

```yaml
model:
  pretrained_name_or_path: Qwen/Qwen3-0.6B
  num_codebooks: 32
  codebook_size: 2048
  attention_backend: sdpa
  freeze_backbone: false
  gradient_checkpointing: true

data:
  train_dir: /path/to/tts_dataset/train
  validation_dir: /path/to/tts_dataset/validation
  test_dir: /path/to/tts_dataset/test
  reference_prob: 0.8
  num_workers: 2

training:
  output_dir: outputs/tts
  device: cuda
  precision: fp16
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 10
  learning_rate: 0.0001
  warmup_steps: 500
  tensorboard: true
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

### 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHON_BIN=/gpu3/guhj/envs/voxbase-tts/bin/python \
bash scripts/train.sh configs/train.yaml
```

程序内部会将当前可见的显卡视为 `cuda:0`，配置保持 `device: cuda` 即可。`CUDA_VISIBLE_DEVICES` 不会独占显卡，启动前仍需检查显存占用。

### Batch、学习率和长度边界

有效 batch size 为：

```text
有效 BS = batch_size × gradient_accumulation_steps
```

学习率对所有可训练参数统一生效，采用线性 warmup 后接余弦衰减；默认峰值为 `1e-4`，warmup 为 500 个 optimizer updates。loss 和梯度按累积组中被 MASK 的 token 总数归一化。AMP 溢出跳过更新时，不推进 optimizer step 和学习率调度。

`masking.max_text_tokens` 和 `masking.max_sequence_length` 是拒绝超限样本的边界。项目不会静默截断或自动过滤超长样本；打包长度包括参考文本/音频、目标文本/音频和边界标记。

## TensorBoard 与训练进度

终端使用 tqdm 显示每个 epoch 的 batch 进度、ETA、速度、loss、准确率、学习率、optimizer step 和梯度范数。

TensorBoard 默认开启，日志位于 `<output_dir>/tensorboard`：

```bash
tensorboard --logdir outputs/tts/tensorboard --host 127.0.0.1 --port 6006
```

记录的主要指标包括：

- `train/loss`、`validation/loss`：被 MASK 目标位置的平均交叉熵；
- `train/accuracy`、`validation/accuracy`：被 MASK 位置的 codebook 索引命中率；
- `train/learning_rate`：当前 optimizer step 使用的学习率；
- `train/grad_norm`：梯度裁剪前的全局范数；
- `train/num_masked`、`validation/num_masked`：统计中的 MASK token 数量。

`acc` 是离散类别的精确命中率，不是语音质量指标。设置 `training.tensorboard: false` 可以关闭 TensorBoard。

## Checkpoint 与断点续训

训练输出通常包括：

```text
outputs/tts/
├── last.pt
├── best.pt
├── backbone_config.json
├── tokenizer/
└── tensorboard/
```

恢复训练：

```bash
bash scripts/train.sh configs/train.yaml --resume outputs/tts/last.pt
```

测试集评估：

```bash
bash scripts/train.sh configs/train.yaml --resume outputs/tts/best.pt --test-only
```

断点恢复会保存模型、优化器、scheduler、AMP scaler、配置、数据顺序和随机数状态。严格恢复要求影响优化和数据重放的设置保持一致，包括模型、数据、MASK、batch size、梯度累积、学习率、warmup、epoch 数和精度等。恢复 epoch 中途可能需要重放已跳过的 batch，以恢复参考采样和 RNG 状态。

只加载可信 checkpoint：训练 checkpoint 使用 `torch.load(weights_only=False)`，不可信 pickle 文件可能执行任意代码。

## 切换 8q 训练

8q 和 32q 使用相同的模型代码，但必须使用匹配的数据和新的输出目录：

```yaml
model:
  num_codebooks: 8
  codebook_size: 2048

data:
  train_dir: /path/to/tts_8q/train
  validation_dir: /path/to/tts_8q/validation
  test_dir: /path/to/tts_8q/test

training:
  output_dir: outputs/tts_8q
```

目标和参考 `codes` 都必须为 `[8, T]`。代码不会自动从 32q 截取前 8q；不要使用 32q checkpoint 恢复 8q 训练。由于多码本在帧内聚合，8q 不会简单地使 Transformer 序列长度变为 32q 的四分之一，速度和显存变化需要实测。

## 推理：生成 codec tokens

推理需要训练输出目录中的 checkpoint、`backbone_config.json` 和 `tokenizer/`：

```bash
bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '你好，欢迎使用 VoxBase-TTS。' \
  --num-frames 100 \
  --output generated_codes.pt
```

GPU 推理示例：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '这是要生成的目标文本。' \
  --num-frames 100 \
  --reference-codes reference_codes.pt \
  --reference-text '这是参考音频对应的文本。' \
  --confidence-threshold 0.8 \
  --max-inference-steps 16 \
  --device cuda --precision fp16 \
  --output generated_with_reference.pt
```

生成从全部 MASK 开始，逐轮填充尚未接受的位置；置信度达到阈值的位置会被固定，最后一轮会强制填完剩余位置。输出是整数 codec tokens，形状为 `[Q, T]`，不会自动解码成 WAV。帧数到时长的换算取决于数据所使用的 codec 帧率。

## 测试

在依赖完整的环境中运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

测试覆盖：

- Qwen3 双向注意力和特殊标记；
- codec 数据读取、row-group 缓存和参考配对；
- MASK 采样、codebook 预测和迭代生成；
- 训练重放、AMP 溢出、checkpoint 和严格恢复；
- tqdm 进度条、TensorBoard 和离线推理。

真实数据测试需要显式设置 `TTS_TEST_DATA_DIR`，否则默认跳过。

## 当前边界

当前版本尚未提供：

- 多卡 DDP 训练；
- 自动音频时长预测；
- WAV 编码和解码链路；
- 完整的主观音质、自然度和可懂度评测；
- 已验证的长序列完整模型显存基准；
- 随仓库发布的数据集、预训练权重或训练输出。

因此，小模型测试通过、loss 下降或单个 token 指标改善，都不能单独作为完整 TTS 系统质量验收。
