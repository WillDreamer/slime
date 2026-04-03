# MoE Expert Balance Tracking

本文档记录 MoE expert balance 监控功能的使用方法和实验注意事项。

## 功能概述

在 inference (SGLang rollout) 和 training (Megatron) 两侧测量 token 在各 expert 之间的分配均衡程度，指标会自动记录到 wandb/tensorboard。

## 指标说明

| 指标 | 含义 | 理想值 | 直觉 |
|------|------|--------|------|
| `load_balance_ratio` | max_count / mean_count | 1.0 | 最忙的 expert 比平均多忙多少倍 |
| `cv` | std / mean | 0.0 | 分配的离散程度 |
| `normalized_entropy` | Shannon 熵 / log2(num_experts) | 1.0 | 越高越均匀 |
| `dead_expert_count` | 没接收到任何 token 的 expert 数 | 0 | 完全没被用到的 expert |
| `max_expert` / `min_expert` | 最多/最少 token 的 expert 编号 | - | 看是否某些 expert 持续偏高/偏低 |

## 如何启用

### Inference 侧 (SGLang Rollout)

**无需额外配置。** 只要满足以下条件就自动生效：
- 模型是 MoE（`args.num_experts` 存在）
- SGLang 返回了 `routed_experts`（启用 `--use-rollout-routing-replay` 时会自动返回）

指标前缀: `rollout/moe_balance/`

### Training 侧 (Megatron)

需要在启动脚本中设置环境变量：

```bash
export MOE_BALANCE_TRACKING=1
```

这个设计为 opt-in 是因为 training 侧每个 MoE layer 每次 forward 都会触发统计，有一定的开销（主要是 GPU→CPU 的 detach + transfer）。

指标前缀: `train_moe_balance/`

## Wandb 中的指标路径

### Inference 侧
```
rollout/moe_balance/mean_cv                    # 所有层 CV 的平均
rollout/moe_balance/max_cv                     # CV 最大的层
rollout/moe_balance/mean_load_balance_ratio    # 所有层 LBR 的平均
rollout/moe_balance/mean_normalized_entropy    # 所有层 entropy 的平均
rollout/moe_balance/total_dead_experts         # 所有层 dead expert 总数
rollout/moe_balance/layer_{i}/cv              # 第 i 层的 CV
rollout/moe_balance/layer_{i}/load_balance_ratio
rollout/moe_balance/layer_{i}/normalized_entropy
rollout/moe_balance/layer_{i}/dead_expert_count
rollout/moe_balance/layer_{i}/max_expert       # 第 i 层最忙的 expert 编号
rollout/moe_balance/layer_{i}/min_expert       # 第 i 层最闲的 expert 编号
```

### Training 侧
```
train_moe_balance/mean_cv
train_moe_balance/max_cv
train_moe_balance/mean_load_balance_ratio
train_moe_balance/mean_normalized_entropy
train_moe_balance/layer_{i}/...               # 同上
```

## 保存原始数据用于离线可视化

除了 wandb 的 summary 指标，还可以将每个 step 每个 layer 每个 expert 的 token count 原始数据保存到磁盘，方便后续做 heatmap 等详细分析。

### 启用方式

设置环境变量指定保存目录：

```bash
export MOE_BALANCE_DATA_DIR=/path/to/save/dir
```

两侧（inference + training）都会自动保存到该目录下。

### 保存格式

```
$MOE_BALANCE_DATA_DIR/
  rollout/
    step_0.npz
    step_1.npz
    ...
  train/
    step_0.npz
    step_1.npz
    ...
```

每个 `.npz` 文件包含：
- `expert_counts`: shape `[num_moe_layers, num_experts]`，每个 expert 被路由到的 token 数
- `step`: 当前 step 编号
- `num_tokens`: 该 step 的总 token 数

### 加载和可视化示例

```python
import numpy as np
import matplotlib.pyplot as plt
import glob

# 加载所有 rollout step 的数据
files = sorted(glob.glob("moe_balance_data/rollout/step_*.npz"))
all_counts = []
steps = []
for f in files:
    data = np.load(f)
    all_counts.append(data["expert_counts"])  # [num_layers, num_experts]
    steps.append(int(data["step"]))

# 堆叠为 [num_steps, num_layers, num_experts]
counts_3d = np.stack(all_counts, axis=0)

# 画某一层随 step 变化的 expert 分布 heatmap
layer_idx = 10
plt.figure(figsize=(16, 6))
plt.imshow(counts_3d[:, layer_idx, :].T, aspect="auto", cmap="hot")
plt.xlabel("Step")
plt.ylabel("Expert ID")
plt.title(f"Layer {layer_idx}: Token Count per Expert over Steps")
plt.colorbar(label="Token Count")
plt.tight_layout()
plt.savefig("expert_heatmap_layer10.png", dpi=150)

# 画所有层在某个 step 的 expert 分布
step_idx = 0
plt.figure(figsize=(16, 8))
plt.imshow(counts_3d[step_idx], aspect="auto", cmap="hot")
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title(f"Step {steps[step_idx]}: Token Count per Expert per Layer")
plt.colorbar(label="Token Count")
plt.tight_layout()
plt.savefig("expert_heatmap_step0.png", dpi=150)
```

### 磁盘占用估算

每个 `.npz` 文件大小约为 `num_layers * num_experts * 8 bytes`（int64），压缩后通常更小：
- Qwen3-30B-A3B (128 experts, ~48 MoE layers): 约 50KB/step
- GLM4.5-355B (160 experts, ~56 MoE layers): 约 70KB/step
- 跑 1000 步大约 50-70MB，非常轻量

## 实验注意事项

### 1. Inference vs Training 的 balance 可能不同

Inference 时 batch 通常更大、序列更长，token 的多样性更高，balance 往往比 training 时更好。Training 侧每个 microbatch 较小，balance 波动会更大。对比两侧数据时注意 `num_tokens` 指标，确保在相近量级下比较。

### 2. Routing Replay 的影响

Slime 默认在 RL training 中使用 routing replay（rollout 阶段记录 routing，training 阶段重放），这意味着 training 侧的 balance 指标实际反映的是 inference 时的 routing 决策。如果你想观察 training 本身的 routing 行为，需要关闭 routing replay（去掉 `--use-rollout-routing-replay`）。

### 3. Aux Loss 在 RL 中被关闭

Slime 的 MoE 模型脚本中 `--moe-aux-loss-coeff 0`，也就是 RL training 阶段不使用 auxiliary balance loss。这意味着 RL training 过程中 expert balance 完全是自然演变的——正好适合你研究 "RL training 过程中 balance 如何变化"。

### 4. 关注 Dead Experts

如果某个 expert 在大量 step 中持续为 dead（`dead_expert_count > 0`），说明存在 expert collapse 问题。可以在 wandb 中画 `layer_{i}/dead_expert_count` 随 step 变化的曲线。

### 5. 建议的对比实验

- **RL training 前后**: 对比 step 0 和最终 step 的 balance 指标
- **不同模型规模**: Qwen3-30B-A3B (128 experts) vs GLM4.5-355B (160 experts)
- **不同 task**: search task vs 其他 task，看 routing 分布是否随 task 变化
- **Inference vs Training**: 同一个 checkpoint，对比两侧的 balance

### 6. 性能开销

- Inference 侧: 几乎无开销，`rollout_routed_experts` 本来就在 pipeline 中传递
- Training 侧 (`MOE_BALANCE_TRACKING=1`): 每个 MoE layer 每次 forward 多一次 `detach()` + CPU copy。对于大模型（如 60+ MoE layers）可能有 1-3% 的额外开销。正式实验时按需开关。

## 如何跑实验

本节介绍如何用 slime 的 debug 模式分别测量 inference 和 training 侧的 expert balance，不需要跑完整 RL training loop。

### 前置准备

#### 1. 下载模型 checkpoint

以 Qwen3-30B-A3B（128 experts）为例：

```bash
# HF checkpoint（给 SGLang 用）
# 需要下载到某个路径，比如 /path/to/Qwen3-30B-A3B
# 可以用 huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /path/to/Qwen3-30B-A3B

# Megatron 格式 checkpoint（给 training 用）
# 需要用 slime 的转换工具从 HF 格式转换：
python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /path/to/Qwen3-30B-A3B \
  --save /path/to/Qwen3-30B-A3B_torch_dist \
  --tensor-model-parallel-size 4 \
  --expert-model-parallel-size 8
```

#### 2. 准备数据

准备一个 `.jsonl` 文件，每行格式如下：
```json
{"prompt": "...", "label": "..."}
```

可以用现有的 `dapo-math-17k.jsonl` 或其他任务数据。`--input-key` 和 `--label-key` 需要和 jsonl 中的 key 对应。

#### 3. 环境

- 需要用 slime 的 docker 镜像（包含 patched SGLang），因为 `--use-rollout-routing-replay` 需要 SGLang 返回 `routed_experts`，这是 slime 对 SGLang 的 patch 功能
- 确保 `PYTHONPATH` 包含 Megatron-LM

### GPU 资源需求

Rollout（SGLang inference）和 Training（Megatron）的 GPU 需求是不同的：

#### Rollout（SGLang inference）

SGLang 的 GPU 数量由 `--rollout-num-gpus-per-engine` 控制，跟 Megatron 的 TP/PP/EP 无关。

Qwen3-30B-A3B 总参数 ~60GB (BF16)，H100 有 80GB 显存：

| 精度 | GPU 数量 | 说明 |
|------|----------|------|
| BF16 | 1 | 勉强能放下，但 KV cache 空间很小，batch size 受限 |
| BF16 | 2-4 | **推荐**，充足的 KV cache 空间，可以跑较大 batch |
| BF16 | 8 | 完整 TP，吞吐最高，但对于 balance 实验不需要 |
| INT4/FP8 | 1 | 量化后模型更小，1 卡即可（slime 有现成的 INT4 配置） |

**纯 rollout 实验推荐 2-4 张 H100 即可。**

#### Megatron Forward-Only（本实验 Step 2）

设置 `MOE_BALANCE_FORWARD_ONLY=1` 后只跑 forward pass，不算梯度不跑 optimizer，GPU 内存 ≈ 模型权重 + activations。

| 配置 | TP | PP | EP | GPU 数 | 每 GPU 内存 |
|------|----|----|-----|--------|------------|
| 推荐 | 2 | 1 | 4 | **4** | ~20-25 GB |
| 最小 | 1 | 1 | 4 | 4 | ~20-25 GB |
| 宽裕 | 4 | 1 | 8 | 8 | ~15 GB |

注意：EP 必须能整除 `num_experts`（128），合法值是 1, 2, 4, 8, 16, 32, 64, 128

#### Megatron Training（之后跑 RL training 时）

完整 training 需要梯度 + optimizer states，标准配置需要 **8 GPU**：

| 配置 | TP | PP | EP | 总 GPU |
|------|----|----|-----|--------|
| 标准（BF16） | 4 | 1 | 8 | 8 |
| FP8 | 1 | 4 | 4 | 8 |

#### 其他可选的 MoE 模型

| 模型 | Experts | Top-K | Rollout GPU | Forward-Only GPU | Training GPU |
|------|---------|-------|-------------|------------------|-------------|
| Qwen3-30B-A3B | 128 | 8 | 2-4 | 4 | 8 |
| GLM4.7-30B-A3B | 64 | 4 | 2-4 | 4 | 8 |
| Moonlight-16B-A3B | 256 | 8 | 2-4 | 4 | 8 |

### Step 1: Rollout-Only（SGLang inference 侧 balance 测量）

用 `--debug-rollout-only` 只启动 SGLang inference，不加载 Megatron。

```bash
#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen3-30B-A3B.sh"

# ============ 根据你的环境修改以下路径 ============
HF_CKPT="/path/to/Qwen3-30B-A3B"
PROMPT_DATA="/path/to/your_data.jsonl"
SAVE_ROLLOUT_DATA="/path/to/debug_rollout/rollout_{rollout_id}.pt"
export MOE_BALANCE_DATA_DIR="/path/to/moe_balance_data"
# ================================================

NUM_GPUS=4  # 根据你申请到的 GPU 数量调整（2-4 张 H100 推荐）

ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON='{
  "env_vars": {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --debug-rollout-only \
   --save-debug-rollout-data "${SAVE_ROLLOUT_DATA}" \
   --rollout-num-gpus ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint "${HF_CKPT}" \
   --prompt-data "${PROMPT_DATA}" \
   --input-key prompt \
   --label-key label \
   --apply-chat-template \
   --rm-type deepscaler \
   --num-rollout 3 \
   --rollout-batch-size 32 \
   --n-samples-per-prompt 1 \
   --rollout-max-response-len 8192 \
   --rollout-temperature 1 \
   --global-batch-size 32 \
   --rollout-num-gpus-per-engine ${NUM_GPUS} \
   --sglang-mem-fraction-static 0.7 \
   --use-rollout-routing-replay \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0
```

**关键参数说明：**
- `--debug-rollout-only`: 只跑 inference，不加载 Megatron
- `--rollout-num-gpus` / `--rollout-num-gpus-per-engine`: 控制 SGLang 用多少 GPU。纯 rollout 只需要 2-4 张 H100，不需要 8 张
- `--save-debug-rollout-data`: 把 rollout 数据（包含 routed_experts）保存为 `.pt` 文件，给 Step 2 用
- `--use-rollout-routing-replay`: **必须**，让 SGLang 返回每个 token 的 expert routing 信息
- `--num-rollout 3`: 跑 3 个 batch，每个 batch 32 个 prompt。可以根据数据量调整
- `MOE_BALANCE_DATA_DIR`: 设置后自动保存原始 expert counts 到该目录

**输出：**
- `$MOE_BALANCE_DATA_DIR/rollout/step_0.npz`, `step_1.npz`, ... — 每个 step 的原始 expert counts
- `$SAVE_ROLLOUT_DATA` 路径下的 `.pt` 文件 — 完整 rollout 数据（给 Step 2 用）
- 日志中会打印 balance 指标（mean_cv, mean_lbr, dead_experts 等）

### Step 2: Forward-Only（Megatron training 侧 balance 测量）

用 `--load-debug-rollout-data` 加载 Step 1 保存的数据，通过 `MOE_BALANCE_FORWARD_ONLY=1` **只跑 forward pass**，不算梯度、不更新权重。这样 GPU 内存只需要放模型权重，不需要 optimizer states 和 gradients，可以用更少的 GPU。

```bash
#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen3-30B-A3B.sh"

# ============ 根据你的环境修改以下路径 ============
MEGATRON_CKPT="/path/to/Qwen3-30B-A3B_torch_dist"
LOAD_ROLLOUT_DATA="/path/to/debug_rollout/rollout_{rollout_id}.pt"
export MOE_BALANCE_TRACKING=1
export MOE_BALANCE_FORWARD_ONLY=1
export MOE_BALANCE_DATA_DIR="/path/to/moe_balance_data"
export PYTHONPATH="/path/to/Megatron-LM/:${PYTHONPATH}"
# ================================================

NUM_GPUS=4  # forward-only 不需要梯度和 optimizer，4 GPU 即可

ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "'"${PYTHONPATH}"'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MOE_BALANCE_TRACKING": "1",
    "MOE_BALANCE_FORWARD_ONLY": "1",
    "MOE_BALANCE_DATA_DIR": "'"${MOE_BALANCE_DATA_DIR}"'"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --load-debug-rollout-data "${LOAD_ROLLOUT_DATA}" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   --load "${MEGATRON_CKPT}" \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 20480 \
   --advantage-estimator grpo \
   --optimizer adam --lr 1e-6 --lr-decay-style constant \
   --optimizer-cpu-offload \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash
   # 注意：不加 --use-rollout-routing-replay
   # 这样 training 侧使用自己的 router 做 routing 决策
   # 可以对比 inference 和 training 的 routing mismatch
```

**关键参数说明：**
- `MOE_BALANCE_FORWARD_ONLY=1`: **核心**，只跑 forward pass，跳过 backward 和 optimizer step。GPU 内存 ≈ 模型权重 + forward activations，无梯度无 optimizer states
- `--load-debug-rollout-data`: 加载 Step 1 保存的 `.pt` 文件，自动设置 `debug_train_only=True`
- `MOE_BALANCE_TRACKING=1`: 启用 training 侧 expert balance 追踪
- `--optimizer-cpu-offload`: optimizer states 放 CPU（虽然 forward-only 不会用到 optimizer，但 Megatron 初始化时会创建，offload 到 CPU 避免占 GPU）
- `--tensor-model-parallel-size 2 --expert-model-parallel-size 4`: 4 GPU 配置（TP=2, EP=4）。每个 GPU 放 ~15GB 模型权重，H100 80GB 完全够用
- **不加 `--use-rollout-routing-replay`**: training 侧用自己的 router 做 routing，可以对比 inference vs training 的 mismatch
- 注意：`RUNTIME_ENV_JSON` 中要传所有 `MOE_BALANCE_*` 环境变量，因为 Ray worker 进程需要这些

**GPU 内存估算（Qwen3-30B-A3B, 4 GPU, forward-only）：**

| 项目 | 每 GPU 内存 |
|------|------------|
| 非 expert 参数 (TP=2) | ~2 GB |
| Expert 参数 (128 experts / EP=4) | ~13 GB |
| Forward activations | ~5-10 GB |
| Optimizer states (CPU offload) | 0 GB |
| Gradients (forward-only 不计算) | 0 GB |
| **总计** | **~20-25 GB** |

H100 80GB 完全够，甚至有余量可以跑较大的 batch。

**输出：**
- `$MOE_BALANCE_DATA_DIR/train/step_0.npz`, `step_1.npz`, ... — training 侧每个 step 的原始 expert counts
- 日志中会打印 `train_moe_balance/` 前缀的指标

### Step 3: 离线分析

两侧数据都保存在 `$MOE_BALANCE_DATA_DIR` 下，可以用以下代码做可视化：

```python
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_expert_counts(pattern):
    """加载某一侧所有 step 的 expert counts"""
    files = sorted(glob.glob(pattern))
    all_counts, steps = [], []
    for f in files:
        data = np.load(f)
        all_counts.append(data["expert_counts"])  # [num_layers, num_experts]
        steps.append(int(data["step"]))
    return np.stack(all_counts, axis=0), steps  # [num_steps, num_layers, num_experts]

# 加载数据
rollout_counts, rollout_steps = load_expert_counts("moe_balance_data/rollout/step_*.npz")
train_counts, train_steps = load_expert_counts("moe_balance_data/train/step_*.npz")

# === Heatmap: 某一层 expert 分布随 step 变化 ===
layer_idx = 10
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, counts, title in zip(axes, [rollout_counts, train_counts], ["Rollout", "Train"]):
    im = ax.imshow(counts[:, layer_idx, :].T, aspect="auto", cmap="hot")
    ax.set_xlabel("Step")
    ax.set_ylabel("Expert ID")
    ax.set_title(f"{title} - Layer {layer_idx}")
    plt.colorbar(im, ax=ax, label="Token Count")
plt.tight_layout()
plt.savefig("expert_balance_comparison_layer10.png", dpi=150)

# === Heatmap: 所有层在某个 step 的 expert 分布 ===
step_idx = 0
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, counts, title in zip(axes, [rollout_counts, train_counts], ["Rollout", "Train"]):
    im = ax.imshow(counts[step_idx], aspect="auto", cmap="hot")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title} - Step {step_idx}")
    plt.colorbar(im, ax=ax, label="Token Count")
plt.tight_layout()
plt.savefig("expert_balance_all_layers_step0.png", dpi=150)

# === CV per layer 对比 ===
def compute_cv_per_layer(counts_3d):
    """对每个 step 每层计算 CV，返回 [num_steps, num_layers]"""
    mean = counts_3d.mean(axis=2, keepdims=True)
    std = counts_3d.std(axis=2, keepdims=True)
    cv = np.where(mean > 0, std / mean, 0).squeeze(-1)
    return cv

rollout_cv = compute_cv_per_layer(rollout_counts)  # [num_steps, num_layers]
train_cv = compute_cv_per_layer(train_counts)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cv, title in zip(axes, [rollout_cv, train_cv], ["Rollout", "Train"]):
    im = ax.imshow(cv.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title} - CV per Layer")
    plt.colorbar(im, ax=ax, label="CV")
plt.tight_layout()
plt.savefig("cv_per_layer_comparison.png", dpi=150)
```

### 关于 Routing Replay 的影响

本实验 **默认关闭 `--use-rollout-routing-replay`**，目的是研究 inference 和 training 之间的 routing mismatch。

- **关闭（默认，本实验推荐）**: Training 侧使用自己的 router 做 routing 决策。此时 training 侧的 balance 反映的是 Megatron forward pass 中 router 的实际行为，可能和 rollout 侧不同（因为 microbatch size 不同、数据顺序不同、框架实现差异等）。**这正是我们想观察的 mismatch。**
- **开启 `--use-rollout-routing-replay`**: Training 侧重放 rollout 时记录的 routing 决策，两侧指标会趋同。这是 slime RL training 的默认模式（为了训练稳定性），但不适合研究 routing mismatch。

#### Mismatch 可能的来源
1. **Batch size 差异**: Inference 时 batch 通常更大，token 多样性更高
2. **数据顺序**: Megatron 的 microbatch 切分方式和 SGLang 的 batch 构成不同
3. **数值精度**: SGLang 和 Megatron 的 router 实现可能有微小数值差异（fp16 vs bf16、算子实现等）
4. **Router 权重同步时机**: 如果在 RL training 中观察，权重更新后 router 行为会变化

### Slurm 集群上的运行

如果你的 H100 集群是通过 Slurm 管理的：

**Step 1 (Rollout-Only) — 4 GPU 即可：**
```bash
#!/bin/bash
#SBATCH --job-name=moe-balance-rollout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu      # 根据你的集群调整

# 如果用 docker/apptainer
apptainer exec --nv /path/to/slime.sif bash run_moe_balance_rollout.sh
```

**Step 2 (Forward-Only) — 4 GPU 即可：**
```bash
#!/bin/bash
#SBATCH --job-name=moe-balance-fwd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

apptainer exec --nv /path/to/slime.sif bash run_moe_balance_forward.sh
```

## 涉及文件

| 文件 | 作用 |
|------|------|
| `slime/utils/expert_balance.py` | 核心指标计算 |
| `slime/ray/rollout.py` | Inference 侧集成（`compute_metrics_from_samples`） |
| `slime/utils/routing_replay.py` | Training 侧数据采集（`TrainingExpertBalanceTracker`） |
| `slime/utils/train_metric_utils.py` | Training 侧指标写入 wandb |
