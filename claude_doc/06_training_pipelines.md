# 训练 Pipeline：SelfForcing·Streaming·梯度控制

> 核心文件：  
> [pipeline/self_forcing_training.py](../pipeline/self_forcing_training.py)  
> [pipeline/streaming_training.py](../pipeline/streaming_training.py)  
> [pipeline/streaming_switch_training.py](../pipeline/streaming_switch_training.py)

---

## 一、三种训练 Pipeline 的关系

```
SelfForcingTrainingPipeline    ← 基础：单 chunk 生成，标准 SelfForcing
    └── StreamingTrainingPipeline  ← 扩展：多 chunk，跨 chunk 共享 KV Cache
            └── StreamingSwitchTrainingPipeline  ← 再扩展：含 prompt 切换
```

---

## 二、SelfForcing 训练基础概念

LongLive 继承自 Self-Forcing 框架。Self-Forcing 的核心思路：

```
传统扩散训练（DDPM）:
  给定干净视频 x_0，加噪到 x_t，训练模型预测 x_0
  问题：测试时每步都用模型自己的输出作为下一步输入（自回归），
        但训练时用真实 x_0 加噪（"老师强制"），存在训练推理 gap

Self-Forcing:
  训练时也用模型自己生成的视频（而不是真实视频）作为去噪起点
  → 训练和推理对齐
  → 配合 DMD 损失（分布匹配蒸馏）保证生成质量
```

---

## 三、StreamingTrainingPipeline

文件：[pipeline/streaming_training.py:19-233](../pipeline/streaming_training.py)

### 3.1 初始化

```python
class StreamingTrainingPipeline:
    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=3, context_noise=0, **kwargs):
        
        self.local_attn_size = kwargs.get("local_attn_size", -1)
        slice_last_frames = kwargs.get("slice_last_frames", 21)
        
        # KV Cache 大小 = 局部窗口 + 额外帧（用于 warmup chunk 的 overlap）
        self.kv_cache_size = (self.local_attn_size + slice_last_frames) * frame_seq_length
```

### 3.2 `generate_chunk_with_cache`

```python
def generate_chunk_with_cache(
    self,
    noise,              # 当前 chunk 的随机噪声 [B, chunk_frames, 16, 60, 104]
    conditional_dict,
    current_start_frame=0,
    requires_grad=True, # False=warmup chunk（不计算梯度）
    return_sim_step=False,
):
```

**外层循环：逐帧块**

```python
all_num_frames = [num_frame_per_block] * (chunk_frames // num_frame_per_block)

for block_index, current_num_frames in enumerate(all_num_frames):
    noisy_input = noise[:, local_start:local_start+current_num_frames]
    
    # 决定这个 block 是否需要梯度
    if block_index * num_frame_per_block < start_gradient_frame_index:
        grad_ctx = torch.no_grad()
    else:
        grad_ctx = contextlib.nullcontext()
    
    with grad_ctx:
        # 内层：多步去噪
        for step_idx, current_timestep in enumerate(denoising_step_list):
            ...
```

**内层循环：多步去噪（同推理，但有梯度）**

```python
_, denoised_pred = self.generator(
    noisy_image_or_video=noisy_input,
    kv_cache=self.kv_cache1,          # ← 跨 chunk 的同一个 cache 对象
    current_start=(current_start_frame + local_start) * frame_seq_length,
)
```

**Context Pass（训练时也有）**

```python
self.generator(
    noisy_image_or_video=denoised_pred,   # 干净帧
    timestep=context_noise * ones(...),    # t=0
    kv_cache=self.kv_cache1,
    current_start=...,
)
```

### 3.3 梯度控制（requires_grad 参数）

```python
if not requires_grad:
    start_gradient_frame_index = chunk_frames   # 全程 no_grad
else:
    start_gradient_frame_index = 0             # 全程开梯度
```

**Trainer 的调用方式**（[trainer/distillation.py:~1096](../trainer/distillation.py)）：

```python
# Step 1: Warmup chunk（requires_grad=False）
# 生成前 N 帧，写入 KV Cache，但不计算梯度（不参与 loss）
output_warmup = pipeline.generate_chunk_with_cache(
    noise=warmup_noise,
    current_start_frame=0,
    requires_grad=False
)

# Step 2: Training chunk（requires_grad=True）
# 生成后续帧，KV Cache 里已有前 N 帧的历史，梯度开启
output_train = pipeline.generate_chunk_with_cache(
    noise=train_noise,
    current_start_frame=warmup_frames,
    requires_grad=True
)

# 用 output_train 计算 DMD loss，反向传播
loss = dmd.compute_loss(output_train)
loss.backward()
```

---

## 四、KV Cache 在训练中的跨 chunk 持久性

```python
class StreamingTrainingPipeline:
    def initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache1 = [...]   # 初始化一次
    
    def generate_chunk_with_cache(self, ...):
        # self.kv_cache1 是同一个对象，不重置！
        # 第一次调用写入 chunk0 的 K/V
        # 第二次调用读取 chunk0 的 K/V + 写入 chunk1 的 K/V
        ...
```

**这模拟了推理时的行为**：推理时 `self.kv_cache1` 也是同一个对象贯穿整个视频生成。训练时通过这种方式让模型学会"利用历史 KV Cache 保持一致性"。

---

## 五、`exit_flags` 梯度控制（精细版）

在 `generate_chunk_with_cache` 内部，还有更精细的 per-block 梯度控制：

```python
# 每个帧块的去噪步骤中：
if step_idx == 0 and block_index < start_gradient_block:
    # 第一步且在梯度起始 block 之前：不保存中间激活
    torch.cuda.empty_cache()

# 最后一步，且需要梯度：
if requires_grad and block_index >= start_gradient_block:
    denoised_pred.requires_grad_(True)   # 开启梯度追踪
```

---

## 六、Block Mask 的准备

训练路径（`kv_cache=None`）使用 FlexAttention，需要预先准备 BlockMask：

```python
# 在 streaming_training.py 的 generate_chunk_with_cache 中
block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
    device=device,
    num_frames=current_num_frames,
    frame_seqlen=1560,
    num_frame_per_block=self.num_frame_per_block,
    local_attn_size=self.local_attn_size
)
self.generator.model.block_mask = block_mask
```

训练时 block_mask 是可以缓存的（相同参数的 mask 每次一样），但实际代码中每次 chunk 都重新生成（保守做法，确保正确性）。

---

## 七、两阶段训练概览

| | 阶段一（init）| 阶段二（long） |
|--|--------------|--------------|
| 配置 | `longlive_train_init.yaml` | `longlive_train_long.yaml` |
| 目标 | 基础因果视频生成能力 | 长视频一致性 |
| 模型 | DMD（完整参数微调） | DMDSwitch + LoRA |
| local_attn_size | -1（全局）或小窗口 | 12（局部，控制显存）|
| 训练帧数 | 21 帧（短视频） | 21+21=42 帧（warmup+train） |
| GPU-days | ~20（32×H100） | ~12（32×H100） |

**阶段二为什么加 LoRA？**

阶段一已经训练了完整参数（~2.5GB）。阶段二在此基础上用 LoRA（~几十MB）做增量微调，专注于长视频的时序一致性，同时保持阶段一学到的基础能力。

---

## 八、teacher_forcing 路径（训练时）

在 `CausalWanSelfAttention` 中，当 `kv_cache=None` 且序列长度是普通长度的 2 倍时，认为是 teacher forcing：

```python
# causal_model.py:132-135
is_tf = (s == seq_lens[0].item() * 2)
if is_tf:
    # 序列 = [clean_frames | noisy_frames]（拼接后一起处理）
    # clean 部分和 noisy 部分分别做 rope_apply，保证位置编码一致
    q_chunk = torch.chunk(q, 2, dim=1)   # 拆成两半
    ...
    roped_query = torch.cat(roped_query, dim=1)  # 重新拼接
```

Teacher Forcing 时，干净帧和噪声帧一起输入，块掩码（`_prepare_teacher_forcing_mask`）控制：
- 干净帧只能看过去的干净帧（因果）
- 噪声帧只能看过去的干净帧 + 同一 block 内的噪声帧（不跨 block 互看）

这样可以一次 forward 同时处理所有帧，比逐帧去噪快很多，适合训练初始化阶段。
