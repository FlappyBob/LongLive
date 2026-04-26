# 配置系统与入口文件

**文件**：
- [train.py](../train.py) / [train_init.sh](../train_init.sh) / [train_long.sh](../train_long.sh)
- [inference.py](../inference.py) / [interactive_inference.py](../interactive_inference.py)
- [configs/](../configs/)

---

## 一、配置文件总览

所有超参数通过 YAML 配置文件管理，使用 OmegaConf 加载：

| 文件 | 场景 | 关键差异 |
|------|------|---------|
| `longlive_train_init.yaml` | 阶段一训练 | 700 步，21 帧，self-forcing，no LoRA |
| `longlive_train_long.yaml` | 阶段二训练 | 3000 步，240 帧，streaming，LoRA rank=256 |
| `longlive_inference.yaml` | 标准推理 | 120 帧，local_attn=12，sink=3 |
| `longlive_interactive_inference.yaml` | 交互推理 | 240 帧，global_sink=true |
| `longlive_inference_infinity.yaml` | 超长推理 | 1050 帧，Infinity 模式 |
| `default_config.yaml` | 基础默认值 | 所有配置的基础继承 |

---

## 二、两阶段训练对比

### 阶段一（train_init.sh → longlive_train_init.yaml）

```yaml
# 目标：从 Wan2.1-T2V-1.3B 微调出基础因果模型
distribution_loss: dmd         # 不含 switch
streaming_training: false       # 标准 21 帧全序列训练
max_iters: 700
lr: 2e-6
model_kwargs:
  local_attn_size: -1          # 全局注意力（阶段一简单一些）
  sink_size: 0
# 无 LoRA，直接全参数训练
```

### 阶段二（train_long.sh → longlive_train_long.yaml）

```yaml
# 目标：从阶段一 checkpoint 出发，训练长视频生成 + prompt 切换能力
generator_ckpt: checkpoints/longlive_init.pt   # 加载阶段一权重
distribution_loss: dmd_switch   # 含 prompt 切换训练
streaming_training: true        # 流式训练 240 帧
streaming_chunk_size: 21
streaming_max_length: 240
model_kwargs:
  local_attn_size: 12           # 局部注意力（节省显存）
  sink_size: 3
adapter:
  type: lora
  rank: 256                     # LoRA 微调，比全参数省显存
max_iters: 3000
```

---

## 三、推理模式对比

### 标准推理（longlive_inference.yaml）

```yaml
num_output_frames: 120          # 生成 120 帧（约 8 秒）
model_kwargs:
  local_attn_size: 12
  sink_size: 3
global_sink: true               # 推理时用 global sink
generator_ckpt: longlive_models/models/longlive_base.pt
lora_ckpt: longlive_models/models/lora.pt
```

### 交互推理（longlive_interactive_inference.yaml）

```yaml
num_output_frames: 240          # 生成 240 帧（约 16 秒）
global_sink: true               # 必须开启，防止 prompt 切换后上下文丢失
```

### Infinity 推理（longlive_inference_infinity.yaml）

```yaml
num_output_frames: 1050         # 生成 1050 帧（约 70 秒）
model_kwargs:
  use_infinite_attention: true  # 使用 CausalWanModelInfinity
```

---

## 四、model_kwargs 的传递链

```
config.model_kwargs
    ↓
WanDiffusionWrapper(**model_kwargs, is_causal=True)
    local_attn_size → CausalWanModel(local_attn_size=...)
    sink_size       → CausalWanModel(sink_size=...)
    timestep_shift  → FlowMatchScheduler(shift=...)
    use_infinite_attention → 选择 CausalWanModelInfinity
```

---

## 五、inference.py 入口逻辑

```python
# 核心流程：
config = OmegaConf.load(args.config)
pipeline = CausalInferencePipeline(config, device)

# 加载 LoRA checkpoint
if config.lora_ckpt:
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_weights)

# 循环生成
for prompt in prompts:
    noise = torch.randn([1, num_output_frames, 16, 60, 104])
    video = pipeline.inference(noise, [prompt])
    save_video(video, output_path)
```

---

## 六、训练 shell 脚本（硬件配置）

```bash
# train_long.sh
torchrun \
  --nproc_per_node=8 \    # 每节点 8 GPU
  --nnodes=4 \             # 4 节点（共 32 GPU）
  train.py \
  --config configs/longlive_train_long.yaml
```

总计 32 × H100，对应约 32 GPU-天的训练量（官方说明）。

---

## 七、数据格式

训练数据只需要**纯文本提示词**文件：

```
# prompts/vidprom_filtered_extended.txt
A majestic eagle soaring over mountain peaks
Waves crashing on a sandy beach at sunset
...（每行一个提示词）
```

`TextDataset` 逐行读取，通过 `DistributedSampler` 分发到各 GPU。

Switch 训练额外需要一个 switch prompt 文件（提供切换后的第二段提示词）。

**无需视频数据**：LongLive 是纯文本蒸馏训练，完全不需要真实视频数据集。
