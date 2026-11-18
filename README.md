# VoxBase-TTS

基于 **Qwen3 双向注意力主干**的 Masked Codec Token TTS 实验实现。模型以文本和可选的参考语音 codec tokens 为条件，通过预测被 MASK 的离散音频 token 学习文本到语音表示的映射。

支持可配置的码本数量（例如 8q、32q）、单卡训练、混合精度、梯度累积、断点续训、TensorBoard 和迭代补全推理。

> 当前推理输出是 **codec token 的 `.pt` 文件，不是 WAV**。仓库尚未集成 codec 编码器/解码器、音频长度预测或完整的音质评测流程；不能直接输入原始参考 WAV，也不能只凭 token 准确率判断语音质量。

## 1. 项目结构

```text
VoxBase-TTS/
├── configs/train.yaml          # 模型、数据、训练、MASK 和评估配置
├── data/tts_data.py            # Parquet 惰性读取、参考配对与 batch padding
├── model/
│   ├── bidirectional_qwen3.py  # Qwen3 双向注意力主干与 TTS 特殊标记
│   └── VoxBase_TTS.py          # 音频 embedding、MASK 训练与迭代生成
├── train.py                   # 单进程训练、验证、checkpoint 与日志
├── infer.py                   # 从训练 checkpoint 离线生成 codec tokens
├── scripts/train.sh           # 训练启动入口
├── scripts/infer.sh           # 推理启动入口
├── tests/                     # 小模型离线测试与可选真实数据测试
└── requirements.txt
```

## 2. 模型如何工作

### 输入与主干

- 文本使用 Qwen tokenizer 和文本 embedding，不做 G2P 预处理。
- 默认加载 `Qwen/Qwen3-0.6B`，将主干注意力改成双向注意力；支持 `sdpa`、`eager`，不支持 Flash Attention 2。
- 每个码本有独立、可训练的音频 embedding。设码本数量为 `Q`、每个码本类别数为 `K`，每张 embedding 表包含 `K + 1` 行，最后一行是音频 MASK。
- 同一音频帧的 `Q` 个 embedding 求和并除以 `sqrt(Q)`，得到一个 Transformer 位置。因此，32q 并不是每帧占用 32 个序列位置。
- 音频 MASK 不加入 Qwen 文本词表。文本/参考/音频的边界通过专用 TTS 标记表示。
- 可选参考条件包括参考文本及其对齐的干净 codec tokens；参考部分不参与目标 MASK 预测损失。

### MASK 训练与预测头

每次 forward 为每个样本采样 MASK 比例，在真实的 `Q × T` 目标位置中随机选取对应数量的位置，不 MASK padding，也不要求整帧一起 MASK。

目标音频 embedding 在进入主干前加上由 MASK 比例生成的时间条件。主干输出经过共享的 `RMSNorm → Linear → SiLU` 特征变换，再由每个码本的独立分类头预测 `K` 类索引。各分类头共享特征，但不共享类别 logits，也不按码本自回归生成。

只在被 MASK 的位置计算交叉熵；分类投影与交叉熵使用 FP32。默认训练整个主干，也可以通过 `model.freeze_backbone` 冻结主干。

## 3. 环境安装

先安装适配本机 GPU/驱动的 CUDA 版 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

项目固定 `transformers==5.14.1`，依赖该版本的 Qwen3 与双向 mask API，不建议随意降级。已验证过的 PyTorch 版本为 2.11.0；`torch>=2.6` 是依赖声明，不代表所有版本组合都经过验证。

### 95 服务器的专用环境

此前已在 95 创建独立的 **venv** 环境，使用：

```bash
source /gpu3/guhj/envs/voxbase-tts/bin/activate
cd /gpu3/guhj/VoxBase-TTS
```

它不是 Conda 环境，不使用 `conda activate`。也可以不激活，直接使用 `/gpu3/guhj/envs/voxbase-tts/bin/python`。

已验证环境包括 Python 3.12.12、PyTorch 2.11.0+cu130、Transformers 5.14.1、NumPy 2.4.3 和 TensorBoard 2.21.0。A800 上通过了小模型 FP32/FP16/BF16 前向、反向及 AdamW 更新检查；这不等于验证了完整模型长序列训练的显存和音质。

## 4. 数据格式

数据必须预先转换为 Parquet 中的离散 codec tokens。训练、验证、测试分别提供独立目录：

```text
tts_dataset/
├── train/
│   └── .../*.parquet
├── validation/
│   └── .../*.parquet
└── test/
    └── .../*.parquet
```

读取器递归搜索给定 split 目录中的 Parquet（排除 `indexes/`），**不要求也不会自动追加 `shards/`**。使用 row group 惰性读取和有界缓存，而非一次把全部音频 codes 加载到内存。

核心字段：

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本标识 |
| `text` | 音频对应文本 |
| `codes` | 二维整数数组，形状为 `[Q, T]`，不是 `[T, Q]` |
| `num_codebooks` | 码本数量 `Q` |
| `num_frames` | 音频 token 帧数 `T` |
| `dataset` | 用于参考配对分组的数据集标识 |
| `speaker_id` | 用于参考配对的说话人标识 |

有效 codec 索引范围为 `[0, K)`，不应在原始数据里写入音频 MASK 或 padding。`codes` 的码本数量必须与模型配置一致。

### 参考样本配对

`reference_prob` 控制使用参考条件的概率。配对限制在同一 split 内，并按 release、dataset、speaker 和码本数量分组；从其他有效样本中选择，排除相同 `sample_id`。

缺失/无效 speaker 或没有合适候选时不使用参考，不退化为自身或其他说话人的样本。只有确认存在候选后才进行概率采样。`reference_prob: 1.0` 也不意味着每条样本必然有参考，更不意味着每轮固定使用同一参考。

## 5. 训练配置与启动

以 [configs/train.yaml](configs/train.yaml) 为起点。**仓库中的数据路径是原机器路径，必须按实际数据位置修改**，不要假定数据已迁移到项目目录。

以下是配置片段，应合并到现有配置中；路径为示例：

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

首次训练需要能够访问预训练模型，或将 `pretrained_name_or_path` 指向已准备好的本地模型目录。数据及预训练权重准备好后，在仓库根目录启动：

```bash
bash scripts/train.sh configs/train.yaml
```

也可以直接运行，或覆盖新训练的 epoch 数：

```bash
python train.py --config configs/train.yaml
bash scripts/train.sh configs/train.yaml --epochs 2
```

### 指定一张 GPU

例如只让程序使用 `nvidia-smi` 中编号为 2 的卡：

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHON_BIN=/gpu3/guhj/envs/voxbase-tts/bin/python \
bash scripts/train.sh configs/train.yaml
```

程序内部将这张卡视为 `cuda:0`，配置保持 `device: cuda` 即可。该设置不独占显卡，启动前仍应检查占用。当前训练器只支持单进程/单卡，不支持 DDP，不要使用多进程 `torchrun` 启动。

### Batch、梯度累积与学习率

完整累积组的有效样本 batch size 为：

```text
有效 BS = batch_size × gradient_accumulation_steps
```

例如 `128 × 16 = 2048`。进度条按实际 DataLoader batch 推进，`step` 则按成功的优化器更新推进；每个 epoch 最后的不足整组部分也会更新。loss 和梯度按整组被 MASK 的 token 总数归一化，而不是简单平均每个样本的 loss。

学习率对所有可训练参数统一生效，采用 **线性 warmup → 余弦衰减**，目前没有主干/新模块分组学习率。默认峰值 `1e-4`，warmup 为 500 个优化器更新步：第 1 次更新用 `2e-7`，第 10 次用 `2e-6`，第 500 次用 `1e-4`。之后根据全部 epochs 的预计更新步数做余弦衰减。若总更新数不足 warmup，就不会进入衰减阶段。

AMP 溢出跳过更新时，不推进 optimizer step 和学习率调度。训练中的权重保持 FP32，`fp16`/`bf16` 控制 CUDA autocast；CPU 训练使用 FP32。批量大小和长度上限不是显存保证，长样本仍可能 OOM。

`masking.max_text_tokens` 和 `masking.max_sequence_length` 是报错边界：超限会拒绝样本，**不会静默截断或自动过滤**。打包长度包括参考文本/音频、目标文本/音频及边界标记。

## 6. 进度条、指标与 TensorBoard

终端进度条显示 epoch、batch 完成量、耗时、ETA、速度，以及最新优化器更新对应的 `loss`、`acc`、`lr`、`step`、`grad`。

- **loss**：被 MASK 目标位置的平均交叉熵。
- **acc**：在被 MASK 的位置，分类头 `argmax` 索引与真实 codebook 索引完全相等的比例。跨样本/码本按 token 汇总，不要求整帧的全部码本同时正确。
- **grad**：梯度裁剪前的全局梯度范数。
- 训练显示的是最近一次累积组的指标，不是整个 epoch 的累计指标。
- 验证指标汇总本轮评估实际读取的 batch。默认每 500 个更新步及 epoch 结束时验证，最多评估 50 个 batch；训练过程不读取 test split。

TensorBoard 默认开启，日志目录为 `<output_dir>/tensorboard`：

```bash
tensorboard --logdir outputs/tts/tensorboard --host 127.0.0.1 --port 6006
```

远端运行时需将远端的 `127.0.0.1:6006` 转发到本机，再在本机打开 <http://127.0.0.1:6006>。若使用菜单式跳板机进入目标服务器，SSH 转发不一定跟随菜单到目标机，需要真正连接目标服务器的转发通道。

在 Scalars 中查看：

| 曲线 | 含义 |
|---|---|
| `train/loss`、`validation/loss` | 训练/验证 loss |
| `train/accuracy`、`validation/accuracy` | 训练/验证 MASK token 准确率 |
| `train/learning_rate` | 当前记录对应更新实际使用的学习率 |
| `train/grad_norm` | 裁剪前梯度范数 |
| `train/epoch` | 从 0 开始的 epoch 编号；终端显示从 1 开始 |
| `train/num_masked`、`validation/num_masked` | 对应统计中的 MASK token 数 |

横轴为优化器 step。TensorBoard 准确率存为 0～1 的比例，终端显示百分比。默认在第 1 步及每 `training.log_every: 10` 步写训练指标，验证时写验证指标。设 `training.tensorboard: false` 可以关闭。

例如 K=2048 时，均匀随机猜测的准确率约为 `0.0488%`。token 准确率只是精确命中率，不等同于音频质量、自然度或可懂度。

## 7. Checkpoint、续训与测试集评估

训练输出目录包含：

```text
outputs/tts/
├── last.pt                # 最近一次保存的训练状态
├── best.pt                # 验证 loss 改善时保存
├── backbone_config.json   # 离线恢复模型结构所需
├── tokenizer/             # 含 TTS 标记的 tokenizer
└── tensorboard/           # 默认开启的事件日志
```

`last.pt` 不是每一步都保存：按 `save_every`、验证改善及 epoch 结束等时机更新。checkpoint 包含模型、优化器、scheduler、AMP scaler、配置、数据顺序和 RNG 状态。

```bash
# 恢复相同训练任务
bash scripts/train.sh configs/train.yaml --resume outputs/tts/last.pt

# 显式评估 test split
bash scripts/train.sh configs/train.yaml --resume outputs/tts/best.pt --test-only
```

注意：

- 新训练拒绝覆盖非空输出目录。恢复要求 `output_dir` 与 checkpoint 所在目录一致。
- 严格续训要求模型、数据、MASK 配置及影响优化/重放的参数保持一致，包括 batch size、梯度累积、学习率、warmup、epochs、精度等；不能只改 YAML 就在原 checkpoint 上更换这些设置。
- epoch 中途恢复会重放跳过的 batch 以恢复参考采样等 RNG 流，因此恢复初期可能有等待时间。应保持数据内容、软件、设备和 worker 设置不变；不保证 CUDA 跨平台逐位一致。
- TensorBoard 恢复时清除恢复点之后的过期曲线。已经运行的进程不会因编辑 YAML 或源码自动更新。
- 旧 `max_steps` checkpoint 不支持当前 epoch 恢复机制；不兼容模型结构的旧权重也不能严格加载。
- **只加载可信 checkpoint**：训练 checkpoint 使用 `torch.load(weights_only=False)`，不可信 pickle 可能执行任意代码。

## 8. 切换 8 码本训练

不需要修改模型代码。在一份独立配置中设置：

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

`codebook_size` 是每个码本的类别数量，不是码本数量；只有数据的类别数变化时才改它。目标和参考 codes 都必须为 `[8, T]`，代码不会自动从 32q 截取前 8q。使用新输出目录重新训练，不要用 32q checkpoint 做 `--resume`。

由于多码本在每帧内聚合，32q 改成 8q 不会使 Transformer 序列长度变成四分之一，整体速度和显存变化需要实测。

## 9. 推理：文本生成 codec tokens

推理需要同目录下的训练 checkpoint、`backbone_config.json` 和 `tokenizer/`，不需要重新下载预训练主干。必须显式指定目标音频 token 帧数：

```bash
bash scripts/infer.sh \
  --checkpoint outputs/tts/best.pt \
  --text '你好，欢迎使用 VoxBase-TTS。' \
  --num-frames 100 \
  --output generated_codes.pt
```

默认 CPU/FP32。GPU 推理可加 `--device cuda --precision fp16`，并用 `CUDA_VISIBLE_DEVICES` 选择显卡。

可选参考条件必须同时提供 codes 文件和对齐文本：

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

参考 `.pt` 可以是整数 Tensor `[Q, R]`，或包含 `codes` Tensor 的字典。codec 索引必须在 `[0, K)`。输出父目录须已存在，已有输出文件不会被覆盖。

生成从全部 MASK 开始，只预测尚未填充的位置；置信度达到阈值则接受并固定，不再重预测已接受位置。全部填满时提前结束，最后一轮强制填完剩余位置。默认阈值 0.8、最多 16 轮；置信度不是正确性的保证，增加轮数也不保证提高音质。

输出字段：

| 字段 | 内容 |
|---|---|
| `codes` | 最终整数 codec tokens，形状 `[Q, T]` |
| `confidence` | 对应预测概率 |
| `below_threshold` | 最终置信度低于阈值的位置 |
| `forced_mask` | 最后一轮强制填充的位置 |
| `num_steps` | 实际迭代轮数 |
| `remaining_history` | 未填充位置数量的变化记录 |
| `text`、`num_frames` | 目标文本与指定帧数 |

帧数到时长的换算依赖数据所用 codec 的帧率，项目不会自动推断时长。要得到可播放的音频，需另行接入与训练数据兼容的 codec 解码器。

## 10. 测试与能力边界

在安装好依赖的环境、仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

测试覆盖小模型双向注意力、数据与参考配对、MASK 计数、训练重放、AMP 溢出处理、TensorBoard、进度条和离线推理等。真实数据测试需要显式设置 `TTS_TEST_DATA_DIR`，默认跳过。

目前没有多卡 DDP、自动时长预测、WAV 编解码链路或已验证的主观音质结论。不要把小模型测试通过、初始 loss 下降或单个 token 指标当作完整 TTS 系统质量验收。
