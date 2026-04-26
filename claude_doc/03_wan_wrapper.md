# WanDiffusionWrapper / WanTextEncoder / WanVAEWrapper

**文件**：[utils/wan_wrapper.py](../utils/wan_wrapper.py)

---

## 一、定位

这三个类是**模型加载与接口层**，将底层的 Wan2.1 原始模块封装成统一的前向接口，供 Pipeline 和 Model 层调用。

```
┌──────────────────────────────────────────────────────┐
│  Pipeline / Model 层                                  │
│  generator(noisy_input, cond, t, kv_cache, ...)      │
│  text_encoder(text_prompts)                          │
│  vae.encode / vae.decode                             │
└────────────┬──────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────┐
│  WanDiffusionWrapper │ WanTextEncoder │ WanVAEWrapper  │
│  (utils/wan_wrapper.py)                               │
└────────────┬──────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────┐
│  wan/modules/causal_model.py  (CausalWanModel)        │
│  wan/modules/t5.py            (UMT5-XXL)              │
│  wan/modules/vae.py           (_video_vae)            │
└───────────────────────────────────────────────────────┘
```

---

## 二、WanDiffusionWrapper

### 构造函数关键逻辑

```python
WanDiffusionWrapper(
    model_name="Wan2.1-T2V-1.3B",
    timestep_shift=8.0,
    is_causal=True,          # True → CausalWanModel; False → WanModel
    local_attn_size=-1,      # -1 全局; 12 局部窗口
    sink_size=0,             # Frame Sink 帧数
    use_infinite_attention=False  # True → CausalWanModelInfinity
)
```

模型加载：
```python
if is_causal:
    if use_infinite_attention:
        self.model = CausalWanModelInfinity.from_pretrained(...)
    else:
        self.model = CausalWanModel.from_pretrained(...)
else:
    self.model = WanModel.from_pretrained(...)  # 教师/Critic 用
```

`seq_len` 计算：
```python
self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760
```
这个值是 transformer 接受的最大序列长度，用于位置编码上限。

### forward() — 核心转换

输入：
```
noisy_image_or_video: [B, F, C=16, H=60, W=104]  ← 时间维在第 1 位
conditional_dict:      {"prompt_embeds": [B, 512, 4096]}
timestep:             [B, F]  ← 每帧独立的 timestep（因果模式）
kv_cache:             List[dict] × 30 blocks
```

输出：
```
flow_pred:  [B, F, C, H, W]   ← Transformer 原始输出
pred_x0:   [B, F, C, H, W]   ← 通过 flow_pred 推导出的 clean latent
```

关键步骤：
```python
# 维度转置: [B,F,C,H,W] → [B,C,F,H,W]（模型内部用的维度顺序）
flow_pred = self.model(
    noisy_input.permute(0, 2, 1, 3, 4),
    ...
).permute(0, 2, 1, 3, 4)

# flow matching: x0 = xt - σt × flow_pred
pred_x0 = _convert_flow_pred_to_x0(flow_pred, xt=noisy_input, timestep)
```

### flow_pred 与 x0 的转换公式

```
Flow Matching 公式:
  xt = (1 - σt) × x0 + σt × noise
  flow_pred = noise - x0    ← Transformer 预测这个

因此:
  x0 = xt - σt × flow_pred

其中 σt 通过 timestep 查 scheduler.sigmas 表得到。
```

`_convert_x0_to_flow_pred`（反向转换，用于 Critic loss）：
```
flow_pred = (xt - x0) / σt
```

### seq_len 参数的作用

传入 `CausalWanModel._forward_inference` 的 `seq_len` 用于 `assert seq_lens.max() <= seq_len`，是一个序列长度上限检查，同时也决定了位置编码的空间范围。

---

## 三、WanTextEncoder

```python
WanTextEncoder():
    self.text_encoder = umt5_xxl(encoder_only=True, ...)  # 加载 UMT5-XXL 文本编码器
    self.tokenizer = HuggingfaceTokenizer(seq_len=512)   # 最大 512 token
```

**forward(text_prompts: List[str]) → dict**：
```python
ids, mask = self.tokenizer(text_prompts)    # tokenize
context = self.text_encoder(ids, mask)      # [B, seq_len, 4096]
context[v:] = 0.0                           # padding 置零
return {"prompt_embeds": context}           # [B, 512, 4096]
```

UMT5-XXL 参数规模约 4B，但在推理/训练时均冻结（`requires_grad_(False)`），常驻 GPU 或按需 offload。

---

## 四、WanVAEWrapper

### 编码

```python
encode_to_latent(pixel: [B, C=3, F, H=480, W=832])
    → latent: [B, F, C=16, H=60, W=104]
```

VAE 使用固定的归一化常数（`mean` 16维、`std` 16维），编码时：
```python
scale = [mean, 1/std]
latent = vae.encode(pixel, scale)
# 输出维度转置: [B,C,F,H,W] → [B,F,C,H,W]
```

### 解码（两种模式）

**decode_to_pixel**（标准解码）：
```python
decode_to_pixel(latent: [B, F, C=16, H=60, W=104])
    → pixel: [B, F, C=3, H=480, W=832]
```

**decode_to_pixel_chunk**（分块解码，用于超长视频 >120 帧）：
```python
for start_idx in range(0, num_frames, chunk_size=120):
    decoded_chunk = vae.decode(chunk)
    decoded_chunks.append(decoded_chunk.cpu())   # 解码后立即转 CPU 释放显存
    vae.clear_cache()                            # 清理 VAE 内部 cache
decoded = torch.cat(decoded_chunks, dim=1)
```

Infinity 模式下（1050 帧）使用分块解码，避免显存 OOM。

---

## 五、FlowMatchScheduler

```python
scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(1000, training=True)
```

核心方法：
- `add_noise(x0, noise, timestep)` → `xt = (1-σt)·x0 + σt·noise`
- `timesteps` → 1000 步的 timestep 张量（用于 warp_denoising_step 映射）

`timestep_shift=8.0` 是 Wan 模型的流匹配参数，控制 σ(t) 曲线的偏移，让采样集中在高信噪比区域。
