# 训练 Pipeline：SelfForcing 与 Streaming

**文件**：
- [pipeline/self_forcing_training.py](../pipeline/self_forcing_training.py)
- [pipeline/streaming_training.py](../pipeline/streaming_training.py)

---

## 一、定位

训练时的 Pipeline 与推理 Pipeline 最大的区别：**需要保留梯度**，用于反向传播更新 Generator 权重。

两个训练 Pipeline 的分工：

| Pipeline | 用途 | 序列长度 | 特点 |
|----------|------|----------|------|
| `SelfForcingTrainingPipeline` | 阶段一训练（init）| 21 帧 | 完整 unroll，支持 `slice_last_frames` 梯度掩码 |
| `StreamingTrainingPipeline` | 阶段二训练（long）| 21~240 帧，分 chunk | 流式生成，每次生成一个 21 帧 chunk |

---

## 二、SelfForcingTrainingPipeline

### 核心方法：inference_with_trajectory()

```python
inference_with_trajectory(
    noise: [B, T, C, H, W],
    initial_latent=None,         # I2V 时的起始帧
    return_sim_step=False,
    slice_last_frames=21,        # 只对最后 N 帧计算梯度
    **conditional_dict
) → (output, timestep_from, timestep_to)
```

**梯度控制逻辑**：

```
start_gradient_frame_index = num_output_frames - slice_last_frames

for block_index in [0, 1, ..., num_blocks-1]:
    current_start_frame = block_index × num_frame_per_block

    if current_start_frame < start_gradient_frame_index:
        with torch.no_grad():   ← 不需要梯度的历史帧
            denoised_pred = generator(...)
    else:
        denoised_pred = generator(...)  ← 需要梯度的最后 N 帧
```

这样只有最后 `slice_last_frames`（默认 21）帧参与梯度计算，节省显存。

### 随机退出步骤（exit_flags）

```python
exit_flags = generate_and_sync_list(num_blocks, num_denoising_steps, device)
# exit_flags: [2, 1, 3, 0, ...]  ← 每帧块随机选择从哪一步退出去噪
```

每个 block 随机选择在第几步"退出"去噪循环（0~N-1），从那步开始计为 final output：
- 不是 exit_flag 的步骤：`no_grad` 运行，只是为了获得更准确的去噪起点
- exit_flag 步骤：可能开启梯度，输出 `denoised_pred` 用于 DMD loss

**同步机制**：由 rank 0 生成随机 exit_flags 后 `dist.broadcast` 到所有 GPU，保证梯度同步。

### KV Cache 清空 vs 保持

每次调用 `inference_with_trajectory` 都会重新初始化 cache：
```python
self._initialize_kv_cache(batch_size, dtype, device)
self._initialize_crossattn_cache(batch_size, dtype, device)
```

---

## 三、StreamingTrainingPipeline

阶段二训练的核心，支持超出单次能放入显存的长序列训练。

### generate_chunk_with_cache()

```python
generate_chunk_with_cache(
    noise: [B, chunk_frames=21, C, H, W],
    conditional_dict: dict,
    current_start_frame: int,   # 在整个序列中的位置
    requires_grad: bool,
    return_sim_step: bool
) → (output_chunk, timestep_from, timestep_to)
```

与 `inference_with_trajectory` 的区别：
- 只处理一个 chunk（21 帧），不处理全序列
- `current_start_frame` 参数表明这个 chunk 在整个序列中的位置
- KV Cache **不在 chunk 之间重置**（跨 chunk 持续保留），这是流式训练的核心

**梯度控制**：
```python
if not requires_grad:
    start_gradient_frame_index = chunk_frames  # 全程 no_grad
else:
    start_gradient_frame_index = 0             # 全程开梯度
```

### 流式训练 vs 整体训练对比

```
整体训练（SelfForcingTrainingPipeline）:
  ┌────────────────────────────────────────────────────────┐
  │  帧 0~20（no_grad）... 帧 42~62（grad）                 │
  │  一次性全部 unroll，最后 N 帧开梯度                      │
  └────────────────────────────────────────────────────────┘

流式训练（StreamingTrainingPipeline）:
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │  chunk 0   │→│  chunk 1   │→│  chunk 2   │ ...
  │  帧 0~20   │ │  帧 21~41  │ │  帧 42~62  │
  │  (no_grad) │ │  (no_grad) │ │  (grad)    │
  └────────────┘ └────────────┘ └────────────┘
  KV Cache 在 chunk 间持续传递，但只在最后几个 chunk 计算梯度
```

---

## 四、Context Noise 更新（两个 Pipeline 共有）

每帧生成后都要重跑一次 context pass：

```python
context_timestep = context_noise × ones(...)   # 通常 context_noise = 0
context_noisy = scheduler.add_noise(denoised_pred, randn_like(...), context_timestep)

with torch.no_grad():
    generator(
        noisy_image=context_noisy,
        timestep=context_timestep,
        kv_cache=kv_cache1,
        current_start=...
    )
```

当 `context_noise=0` 时，`context_noisy = denoised_pred`（干净帧），写入 cache 的是完全 clean 的 K/V。

---

## 五、local_attn_size 动态调度

`SelfForcingTrainingPipeline` 支持 **per-step 动态调整** `local_attn_size`：

```python
# 如果 local_attn_size 是一个 list（每个去噪步骤一个值）
if isinstance(self.local_attn_size, list):
    for step_idx, timestep in enumerate(denoising_step_list):
        self.generator.model.local_attn_size = local_attn_size[step_idx]
        self._set_all_modules_max_attention_size(local_attn_size[step_idx])
```

应用场景：早期高噪声步骤用更小的窗口（快速），后期低噪声步骤用更大窗口（精细）。
