# 配置文件·入口文件·数据格式

> 核心文件：`configs/`，`inference.py`，`interactive_inference.py`，`train.py`

---

## 一、配置文件总览

| 文件 | 用途 |
|------|------|
| `configs/default_config.yaml` | 默认参数基础，训练时被合并 |
| `configs/longlive_inference.yaml` | 标准推理（120帧，本地注意力） |
| `configs/longlive_inference_infinity.yaml` | 无限长视频推理（chunk 解码） |
| `configs/longlive_interactive_inference.yaml` | 多 prompt 交互推理 |
| `configs/longlive_train_init.yaml` | 阶段一：基础训练 |
| `configs/longlive_train_long.yaml` | 阶段二：长视频 + LoRA 训练 |

---

## 二、推理配置对比

### 2.1 `longlive_inference.yaml`（标准推理）

```yaml
# ── 去噪调度 ──────────────────────────────────────
denoising_step_list: [1000, 750, 500, 250]   # 4步去噪
warp_denoising_step: true   # 把整数 index 映射为真实 sigma 值

# ── 模型结构 ──────────────────────────────────────
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  local_attn_size: 12        # 滑动窗口 12 帧
  timestep_shift: 5.0        # Flow Matching sigma 偏移
  sink_size: 3               # Frame Sink 3 帧

# ── 推理控制 ──────────────────────────────────────
num_output_frames: 120       # 总帧数（8s@15fps）
global_sink: true
context_noise: 0

# ── 权重 ──────────────────────────────────────────
generator_ckpt: longlive_models/models/longlive_base.pt
lora_ckpt: longlive_models/models/lora.pt
adapter:
  type: lora
  rank: 256
  alpha: 256
```

### 2.2 `longlive_inference_infinity.yaml`（无限长推理）

主要区别：

```yaml
num_output_frames: 1050      # 约 70 秒@15fps
model_kwargs:
  use_infinite_attention: true   # 启用 chunk VAE 解码
```

无限长推理时 VAE 解码改用 `decode_to_pixel_chunk`，每次只解码 120 帧，避免 OOM。

---

## 三、训练配置对比

### 3.1 阶段一（`longlive_train_init.yaml`）

```yaml
# 基础模型（14B 作为 real_score 教师）
real_name: Wan2.1-T2V-14B    # 教师模型（不更新）
fake_name: Wan2.1-T2V-1.3B  # 判别器

# 训练规模
max_iters: 3000
total_batch_size: 64          # 8卡 × batch_size=1 × gradient_acc=8
lr: 1.0e-05

# 不用 LoRA（全参数微调）
# adapter 部分没有或被注释

# 流式训练关闭
streaming_training: false     # 阶段一只训练 21 帧（短视频）
streaming_chunk_size: 21
```

### 3.2 阶段二（`longlive_train_long.yaml`）

```yaml
# 从阶段一的 checkpoint 开始
generator_ckpt: checkpoints/longlive_init.pt

# 用 LoRA（接阶段一参数，只更新 LoRA 层）
adapter:
  type: lora
  rank: 256
  apply_to_critic: true     # fake_score 也用 LoRA

# 流式训练（核心：跨 chunk 共享 KV Cache）
streaming_training: true
streaming_chunk_size: 21      # 每次训练 21 帧
streaming_max_length: 240     # 最长生成 240 帧（16s）

# Prompt 切换训练（DMDSwitch）
distribution_loss: dmd_switch
switch_prompt_path: prompts/vidprom_filtered_extended_switch.txt
switch_choices: [21, 39, 57, ...]  # 可能的切换帧位置

# 全局 sink 在训练时关闭（避免跨 batch 的 sink 干扰）
global_sink: false

# 梯度检查点（节省显存）
gradient_checkpointing: true
```

---

## 四、两阶段配置关键差异汇总

| 参数 | 阶段一 | 阶段二 |
|------|--------|--------|
| `streaming_training` | false | true |
| `distribution_loss` | `dmd` | `dmd_switch` |
| `global_sink` | false | false |
| `adapter` | 无（全参数） | LoRA rank=256 |
| `generator_ckpt` | Wan 原始权重 | 阶段一输出 |
| `real_name` | Wan-14B | Wan-14B |
| `streaming_chunk_size` | 21 | 21 |
| `streaming_max_length` | N/A | 240 |
| `max_iters` | ~3000 | ~3000 |
| GPU 天数 | ~20 | ~12 |

---

## 五、入口文件详解

### 5.1 `inference.py`

```python
# 关键流程：
config = OmegaConf.load(args.config_path)

# 分布式初始化（可选）
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(...)

# 构建 Pipeline（不加载权重）
pipeline = CausalInferencePipeline(config, device=device)

# 加载基础权重
state_dict = torch.load(config.generator_ckpt)
pipeline.generator.load_state_dict(state_dict["generator"])

# 加载 LoRA（可选）
if config.adapter:
    pipeline.generator.model = configure_lora_for_model(...)
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt)

# 推理循环
for batch in dataloader:
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        low_memory=True,
    )
    write_video(output_path, video * 255, fps=16)
```

### 5.2 `interactive_inference.py`

与 `inference.py` 的差异：

```python
# 使用 InteractiveCausalInferencePipeline 代替 CausalInferencePipeline
pipeline = InteractiveCausalInferencePipeline(config, device=device)

# 读取多段 prompt 和切换位置（从 JSONL 格式文件）
for item in jsonl_data:
    prompts_list = item["prompts"]          # List[str]，每段一条
    switch_frames = item["switch_frames"]   # List[int]

# 交互推理
video = pipeline.inference(
    noise=sampled_noise,
    text_prompts_list=[[p] for p in prompts_list],
    switch_frame_indices=switch_frames,
    low_memory=True,
)
```

### 5.3 `train.py`

```python
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)  # 合并默认参数

if config.trainer == "score_distillation":
    trainer = ScoreDistillationTrainer(config)

trainer.train()
```

---

## 六、数据格式

### 6.1 单 prompt 文本文件（`data_path`）

```
# vidprom_filtered_extended.txt
A serene mountain landscape with pine trees...
A busy city street at night with neon lights...
A cat playing in a sunlit garden...
```

每行一个 prompt，`TextDataset` 逐行读取：

```python
# utils/dataset.py
class TextDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "prompts": [self.prompts[idx]],
            "extended_prompts": [self.extended_prompts[idx]] if exists else None
        }
```

### 6.2 交互推理 JSONL（`example/interactive_example.jsonl`）

```jsonl
{"prompts": ["A cat runs across a field", "The cat leaps onto a stone"], "switch_frames": [30]}
{"prompts": ["Waves crashing on a beach", "A lighthouse blinks in the fog"], "switch_frames": [45]}
```

每行一个 JSON 对象，包含多段 prompt 和切换帧索引。

### 6.3 Switch 训练数据（`switch_prompt_path`）

```
# vidprom_filtered_extended_switch.txt
# 格式：每两行一组（两段 prompt，用于 DMDSwitch 训练）
A dog chases a ball in the park.
The dog splashes into a pond.
```

---

## 七、`default_config.yaml` 的默认值

```yaml
# configs/default_config.yaml（训练时合并的基础配置）
causal: true
num_train_timestep: 1000
min_score_timestep: 0
warp_denoising_step: true
normalization: true
dfake_gen_update_ratio: 5    # 每 5 步 generator 更新 1 次 fake_score
gradient_accumulation_steps: 1
log_iters: 10
vis_interval: 500
save_interval: 500
max_checkpoints: 3
auto_resume: true
use_ema: false
ema_weight: 0.9999
```

---

## 八、Shell 脚本的作用

```bash
# inference.sh
torchrun --nproc_per_node=1 --master_port=29500 \
    inference.py --config_path configs/longlive_inference.yaml

# inference_infinity.sh（无限长）
torchrun --nproc_per_node=1 --master_port=29500 \
    inference.py --config_path configs/longlive_inference_infinity.yaml

# train_init.sh（阶段一，8卡）
torchrun --nproc_per_node=8 --master_port=29500 \
    train.py --config_path configs/longlive_train_init.yaml \
    --logdir logs --disable-wandb --no-one-logger

# train_long.sh（阶段二，8卡）
torchrun --nproc_per_node=8 --master_port=29500 \
    train.py --config_path configs/longlive_train_long.yaml \
    --logdir logs --disable-wandb --no-one-logger
```
