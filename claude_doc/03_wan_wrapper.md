# WanDiffusionWrapper / TextEncoder / VAE

> 核心文件：[utils/wan_wrapper.py](../utils/wan_wrapper.py)

---

## 一、三大组件总览

```
┌─────────────────────────────────────────────────────────────┐
│                    WanTextEncoder                           │
│  UMT5-XXL (encoder only) → prompt_embeds [B, 512, 4096]   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    WanDiffusionWrapper                      │
│  CausalWanModel → flow_pred → x0_pred                     │
│  包含 FlowMatchScheduler（噪声管理）                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    WanVAEWrapper                            │
│  视频 VAE：pixel [B,3,F,H,W] ↔ latent [B,F,16,60,104]    │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、WanTextEncoder

文件：[wan_wrapper.py:16-57](../utils/wan_wrapper.py)

### 2.1 架构

```python
class WanTextEncoder(torch.nn.Module):
    def __init__(self):
        self.text_encoder = umt5_xxl(encoder_only=True, ...)  # 仅编码器部分
        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/.../google/umt5-xxl/",
            seq_len=512,        # 固定截断到 512 tokens
            clean='whitespace'  # 预处理：合并空白
        )
```

### 2.2 前向传播

```python
def forward(self, text_prompts: List[str]) -> dict:
    ids, mask = self.tokenizer(text_prompts, return_mask=True)
    seq_lens = mask.gt(0).sum(dim=1)          # 每条文本的实际长度
    context = self.text_encoder(ids, mask)     # [B, 512, 4096]
    
    for u, v in zip(context, seq_lens):
        u[v:] = 0.0   # 把 padding 位置清零（不是直接截断）
    
    return {"prompt_embeds": context}          # [B, 512, 4096]
```

**关键细节**：
- padding 位置清零而非删除，保持 `[B, 512, 4096]` 固定 shape
- 下游 `CausalWanModel.text_embedding` 的 Linear 层输入维度固定为 4096

### 2.3 显存优化

推理时文本编码器默认在 GPU（见 `WanTextEncoder.__init__` 的 `.cuda()`），但 `low_memory=True` 时通过 `DynamicSwapInstaller` 把它"换出"到 CPU，腾出空间给扩散模型：

```python
# inference.py:137
DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
```

---

## 三、WanDiffusionWrapper

文件：[wan_wrapper.py:171-378](../utils/wan_wrapper.py)

### 3.1 初始化

```python
class WanDiffusionWrapper(torch.nn.Module):
    def __init__(self, model_name, timestep_shift=8.0, is_causal=False,
                 local_attn_size=-1, sink_size=0, use_infinite_attention=False):
        
        if is_causal:
            self.model = CausalWanModel.from_pretrained(...)   # ← LongLive 的核心
        else:
            self.model = WanModel.from_pretrained(...)          # ← 原版 Wan（训练初始化）
        
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, ...)
        self.scheduler.set_timesteps(1000, training=True)
        
        # seq_len: 用于计算 attention 的序列长度上限
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760
```

### 3.2 forward 入口（六条路径）

```python
def forward(self, noisy_image_or_video, conditional_dict, timestep, 
            kv_cache=None, crossattn_cache=None, current_start=None,
            classify_mode=False, clean_x=None, aug_t=None,
            sink_recache_after_switch=False):
    
    # 1. 提取文本嵌入
    prompt_embeds = conditional_dict["prompt_embeds"]
    
    # 2. 处理 timestep 维度（因果模式是 per-frame，非因果是 per-batch）
    input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
    
    # 3. 调用底层模型（六条路径）
    if kv_cache is not None:
        # ① 推理路径（因果，有 KV Cache）
        flow_pred = self.model(x.permute(0,2,1,3,4), t=..., kv_cache=kv_cache, ...)
    elif clean_x is not None:
        # ② Teacher Forcing 训练
        flow_pred = self.model(x.permute(0,2,1,3,4), clean_x=clean_x, ...)
    elif classify_mode:
        # ③ DMD 判别器分类模式
        flow_pred, logits = self.model(x, classify_mode=True, ...)
    else:
        # ④ 标准训练（无 KV Cache）
        flow_pred = self.model(x.permute(0,2,1,3,4), ...)
    
    # 4. flow_pred → x0
    pred_x0 = self._convert_flow_pred_to_x0(flow_pred, noisy_input, timestep)
    
    return flow_pred, pred_x0
```

---

## 四、Flow Matching：flow_pred ↔ x0 转换

Flow Matching 的核心公式：

```
x_t = (1 - σ_t) × x_0 + σ_t × ε          # 加噪过程
flow_pred = ε - x_0 = (x_t - x_0) / σ_t  # 模型学习的目标

# 反推 x_0（_convert_flow_pred_to_x0）:
x_0 = x_t - σ_t × flow_pred
```

### 4.1 `_convert_flow_pred_to_x0`

```python
# wan_wrapper.py:231-255
def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
    # 用 float64 提高精度
    flow_pred, xt, sigmas, timesteps = map(lambda x: x.double(), ...)
    
    # 根据 timestep 查找对应的 sigma
    timestep_id = torch.argmin((timesteps - timestep).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    
    x0_pred = xt - sigma_t * flow_pred   # 核心公式
    return x0_pred.to(original_dtype)
```

**为什么用 float64？** flow_pred 和 xt 都是 bfloat16，精度有限。当 σ_t 很小（t 接近 0，接近干净帧）时，`xt - σ_t × flow_pred` 的精度损失会很大，切换到 float64 避免数值问题。

### 4.2 `_convert_x0_to_flow_pred`（静态方法）

```python
# wan_wrapper.py:257-278（训练时用）
flow_pred = (xt - x0_pred) / sigma_t
```

---

## 五、FlowMatchScheduler

文件：[utils/scheduler.py](../utils/scheduler.py)

```python
class FlowMatchScheduler:
    def __init__(self, shift=8.0, sigma_min=0.0, extra_one_step=True):
        # shift: Wan 官方用 8.0/5.0，控制 sigma 曲线的偏斜程度
        # 较大的 shift → 更多步骤在高噪声区域，有利于细节
```

**与 DDPM 的区别**：Flow Matching 的 `add_noise` 不需要 alphas_cumprod，直接用线性插值：

```python
# 推理时的去噪步（causal_inference.py:173-178）
noisy_input = scheduler.add_noise(
    denoised_pred.flatten(0,1),     # x_0 预测
    torch.randn_like(...),           # 新随机噪声
    next_timestep × ones([B*F])     # 下一步的 timestep
).unflatten(0, denoised_pred.shape[:2])
```

`timestep_shift=5.0`（推理配置）vs `8.0`（Wan 原始）：LongLive 推理时把偏移调小，让去噪曲线更"靠前"，提升单步质量。

---

## 六、WanVAEWrapper

文件：[wan_wrapper.py:60-169](../utils/wan_wrapper.py)

### 6.1 编解码尺寸关系

```
像素空间: [B, 3, F, H=480, W=832]（原始视频）
                 ↓ encode（VAE）
latent 空间: [B, F, 16, H/8=60, W/8=104]
                 ↑ decode

每帧压缩比: 3×480×832 → 16×60×104
  = 1,198,080 → 99,840
  ≈ 12倍压缩
```

### 6.2 归一化参数

```python
# wan_wrapper.py:63-73（Wan 官方均值/标准差，16个通道各一个）
mean = [-0.7571, -0.7089, ...]  # 16 个值
std  = [ 2.8184,  1.4541, ...]  # 16 个值

scale = [mean, 1/std]   # 编解码时传给 VAE 做归一化
```

### 6.3 `decode_to_pixel_chunk`（无限长视频支持）

```python
def decode_to_pixel_chunk(self, latent, chunk_size=120):
    # 超过 chunk_size 帧时分批解码，避免 OOM
    for start_idx in range(0, num_frames, chunk_size):
        chunk = latent[:, start_idx:end_idx]
        self.model.clear_cache()    # 清空 VAE 内部缓存
        decoded_chunk = decode(chunk)
        decoded_chunks.append(decoded_chunk.cpu())  # 移到 CPU 节省显存
```

**为什么要 `clear_cache()`？** VAE 可能有内部的卷积缓存（用于流式解码），每个 chunk 开始前清除，避免跨 chunk 的边界效应。

---

## 七、LoRA 加载（推理时）

```python
# inference.py:98-131
from utils.lora_utils import configure_lora_for_model

# 1. 对 generator.model（CausalWanModel）应用 LoRA 包装
pipeline.generator.model = configure_lora_for_model(
    pipeline.generator.model,
    lora_config=config.adapter  # rank=256, alpha=256
)

# 2. 加载 LoRA 权重
lora_checkpoint = torch.load(lora_ckpt_path)
peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
```

LoRA 叠加在**基础模型权重**上，推理时合并为一体（`rank=256` 相当于大 LoRA，接近全参数微调的效果）。

---

## 八、关键流程：从 noise 到 video

```
noise [B, F, 16, 60, 104]  (纯随机噪声)
    │
    │ [逐帧块，4步去噪]
    ▼
WanDiffusionWrapper.forward(noisy, cond, t=1000)
    → flow_pred → pred_x0
    → add_noise(pred_x0, t=750) → noisy_input_next
    │
WanDiffusionWrapper.forward(noisy_input_next, cond, t=750)
    → pred_x0
    → add_noise(pred_x0, t=500) → ...
    │
WanDiffusionWrapper.forward(..., t=250)
    → final denoised_pred  [B, F, 16, 60, 104]  (latent)
    │
    │ [context pass，t=0]
WanDiffusionWrapper.forward(denoised_pred, cond, t=0, kv_cache)
    → 只更新 KV Cache，不记录输出
    │
    ▼
WanVAEWrapper.decode_to_pixel(latent)
    → video [B, F, 3, 480, 832]  ∈ [0,1]
```
