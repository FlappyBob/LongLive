# CausalInferencePipeline — 单提示词推理流水线

**文件**：[pipeline/causal_inference.py](../pipeline/causal_inference.py)

---

## 一、定位

`CausalInferencePipeline` 是**推理的基础类**，实现了单提示词、逐帧自回归视频生成的完整流程。
`InteractiveCausalInferencePipeline` 和 `SwitchCausalInferencePipeline` 都继承自它。

---

## 二、初始化

```python
CausalInferencePipeline(args, device, generator=None, text_encoder=None, vae=None)
```

创建三个模型（或复用外部传入的）：
- `self.generator` = `WanDiffusionWrapper(is_causal=True, ...)`
- `self.text_encoder` = `WanTextEncoder()`
- `self.vae` = `WanVAEWrapper()`

关键超参数：
```python
self.denoising_step_list       # [1000, 750, 500, 250] 每帧的去噪步骤
self.frame_seq_length = 1560   # 每帧 token 数（硬编码）
self.num_transformer_blocks = 30
self.local_attn_size           # 来自 args.model_kwargs.local_attn_size
self.num_frame_per_block       # 每次生成几帧（默认 1 或 3）
```

---

## 三、inference() — 核心推理循环

```python
inference(
    noise: [B, T, 16, 60, 104],    # 纯高斯噪声（用户提供种子即可）
    text_prompts: List[str],         # 文本提示词
    return_latents: bool = False,
    low_memory: bool = False
) → video: [B, T, 3, 480, 832]  (像素值 [0,1])
```

**完整执行流程：**

```
Step 1: 文本编码
  text_encoder(text_prompts) → conditional_dict

Step 2: 初始化缓存
  _initialize_kv_cache(batch_size, dtype, device)
      → kv_cache1: List[30 × {k,v,global_end,local_end}]
  _initialize_crossattn_cache(...)
      → crossattn_cache: List[30 × {k,v,is_init}]

Step 3: 时序去噪循环 (outer loop, 逐帧块)
  for current_start_frame in [0, 1, 2, ..., num_blocks-1]:
    noisy_input = noise[:, current_start_frame : +num_frame_per_block]

    Step 3.1: 空间去噪循环 (inner loop, 多步去噪)
      for timestep in [1000, 750, 500, 250]:
        _, denoised_pred = generator(
            noisy_input, conditional_dict, timestep,
            kv_cache=kv_cache1,
            current_start=current_start_frame × 1560
        )
        if 不是最后一步:
            noisy_input = scheduler.add_noise(denoised_pred, noise, next_timestep)

    Step 3.2: 记录 clean 帧
      output[:, current_start_frame] = denoised_pred

    Step 3.3: KV Cache 更新（context pass）
      generator(
          noisy_image=denoised_pred,   ← 用 clean 帧
          timestep = context_noise,    ← t ≈ 0
          kv_cache=kv_cache1,
          current_start=...
      )
      ← 这次 forward 只是为了把 clean 帧的 K/V 写入 cache，不保存输出

    current_start_frame += num_frame_per_block

Step 4: VAE 解码
  video = vae.decode_to_pixel(output)  或  decode_to_pixel_chunk（Infinity 模式）
  video = (video * 0.5 + 0.5).clamp(0, 1)  ← [-1,1] → [0,1]
```

---

## 四、为什么需要 Context Pass？

去噪完成后，KV Cache 中存储的是**带噪声**的 K/V（因为 denoise 循环最后一步是从高噪声状态预测 x0，但写入 cache 的是这个噪声帧的 K/V）。

为了让后续帧能获得**干净的历史上下文**，需要额外跑一次 `t=0`（或 `context_noise`）的前向：

```
context pass 作用:
  输入: denoised_pred（clean 帧）+ t ≈ 0
  输出: 被丢弃
  副作用: 将 clean 帧的 K/V 写入 cache，覆盖之前的噪声 K/V

这样下一帧生成时，看到的历史是干净的帧，而不是中间噪声状态。
```

这是 **Self-Forcing**（自强迫）训练中的核心设计：训练时学会从干净上下文预测，推理时也维持干净上下文。

---

## 五、KV Cache 初始化细节

```python
# local attention 模式
kv_cache_size = local_attn_size × frame_seq_length  # e.g. 12 × 1560 = 18720

# global attention 模式（local_attn_size == -1）
kv_cache_size = num_output_frames × frame_seq_length  # e.g. 120 × 1560

# 每 Block 的 kv_cache:
{
    "k": zeros([B, kv_cache_size, 12, 128]),
    "v": zeros([B, kv_cache_size, 12, 128]),
    "global_end_index": tensor([0]),
    "local_end_index":  tensor([0])
}
```

---

## 六、Profiling 支持

设置 `profile=True` 时，会用 CUDA Events 精确测量各阶段耗时：
- 初始化（文本编码 + cache 初始化）
- 扩散生成（每 block 单独计时）
- VAE 解码

---

## 七、low_memory 模式

```python
if low_memory:
    output_device = 'cpu'   # 生成的 latent 放 CPU
    # 文本编码器需要时 offload 到 CPU，释放显存给生成器
    move_model_to_device_with_memory_preservation(self.text_encoder, ...)
```

这样大容量的文本编码器权重（UMT5-XXL ~15GB）在推理生成阶段可以从 GPU 移走，节省峰值显存。
