# CausalInferencePipeline：推理主循环·时序去噪·Context Pass

> 核心文件：[pipeline/causal_inference.py](../pipeline/causal_inference.py)

---

## 一、Pipeline 架构

```
CausalInferencePipeline(torch.nn.Module)
│
├── self.generator      : WanDiffusionWrapper  (CausalWanModel)
├── self.text_encoder   : WanTextEncoder       (UMT5-XXL)
├── self.vae            : WanVAEWrapper        (Video VAE)
├── self.scheduler      : FlowMatchScheduler
├── self.kv_cache1      : List[dict] × 30      (推理用 KV Cache)
├── self.crossattn_cache: List[dict] × 30      (文本 KV Cache)
├── self.denoising_step_list: [1000,750,500,250]
├── self.num_frame_per_block: 3                (每块生成帧数)
└── self.local_attn_size: 12                  (滑动窗口大小)
```

---

## 二、推理流程全图

```
inference(noise, text_prompts)
│
├── 1. 文本编码
│   WanTextEncoder(text_prompts) → conditional_dict
│
├── 2. 初始化 Cache
│   _initialize_kv_cache(batch, dtype, device, kv_cache_size=12×1560)
│   _initialize_crossattn_cache(batch, dtype, device)
│   _set_all_modules_max_attention_size(local_attn_size=12)
│
└── 3. 时序去噪循环 (外层：逐帧块)
    for current_num_frames in [3, 3, 3, ..., 3]:   # num_blocks = num_output_frames / 3
    │
    ├── 3.1 取当前帧块的初始噪声
    │   noisy_input = noise[:, frame_start:frame_start+3]
    │
    ├── 3.2 空间去噪循环 (内层：多步去噪)
    │   for t in [1000, 750, 500, 250]:
    │   │
    │   ├── [非最后步] generator(noisy_input, cond, t, kv_cache) → pred_x0
    │   │   add_noise(pred_x0, t_next) → noisy_input  (加噪准备下步)
    │   │
    │   └── [最后步]  generator(noisy_input, cond, t=250, kv_cache) → denoised_pred
    │
    ├── 3.3 记录输出
    │   output[:, frame_start:frame_start+3] = denoised_pred
    │
    ├── 3.4 Context Pass（更新干净帧的 KV Cache）
    │   generator(denoised_pred, cond, t=0, kv_cache)  ← 输出丢弃
    │
    └── 3.5 更新帧指针
        current_start_frame += current_num_frames
    │
    ▼
    4. VAE 解码
    vae.decode_to_pixel(output) → video ∈ [0,1]
```

---

## 三、时序与空间去噪的理解

LongLive 的去噪有两层嵌套循环，容易混淆：

```
"时序"去噪 = 外层循环，决定生成第几帧
  → 每次处理 num_frame_per_block 帧（通常 3 帧）
  → 必须顺序执行（第 N 帧要读第 0~N-1 帧的 KV Cache）

"空间"去噪 = 内层循环，对单帧执行多步去噪
  → 每帧执行 4 步（t=1000→750→500→250）
  → 每步都调用 generator，KV Cache 被重复读但不重复写
    （只有 context pass 才最终写入干净帧的 K/V）
```

**为什么内层去噪时 KV Cache 可以安全重用？**

第 N 帧在去噪时，第 0~N-1 帧已经完成（context pass 后 cache 是干净帧的 K/V）。第 N 帧的去噪是用带噪声的帧去查询历史干净帧的 K/V，这没有问题——历史 K/V 不会被中间去噪步骤修改，直到 context pass。

---

## 四、Context Pass 详解

文件：[causal_inference.py:192-200](../pipeline/causal_inference.py)

```python
# 去噪循环结束后，denoised_pred 是最终干净帧 (latent)
context_timestep = torch.ones_like(timestep) * self.args.context_noise  # = 0

self.generator(
    noisy_image_or_video=denoised_pred,   # ← 干净帧（t=0）
    conditional_dict=conditional_dict,
    timestep=context_timestep,             # t=0，σ≈0
    kv_cache=self.kv_cache1,              # ← 会更新 cache
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * self.frame_seq_length,
)
# 返回值被丢弃 —— 只要 cache 更新这个副作用
```

**为什么需要 Context Pass？**

```
去噪步骤中 KV Cache 的状态演变：

步骤1 (t=1000): generator(noisy1) → 写入"重噪声帧"的 K/V
步骤2 (t=750):  generator(noisy2) → 覆盖为"中等噪声帧"的 K/V
步骤3 (t=500):  generator(noisy3) → 覆盖为"轻微噪声帧"的 K/V
步骤4 (t=250):  generator(noisy4) → 覆盖为"接近干净帧"的 K/V
                                       ↑ 但还不是真正的干净帧！

Context Pass (t=0): generator(clean) → 覆盖为"完全干净帧"的 K/V ✓
```

如果没有 Context Pass，下一帧的注意力看到的历史 K/V 对应带噪声的特征表示，随着帧数增加，累积误差会让视频质量越来越差。

**t=0 意味着什么？**

Flow Matching 公式：`x_t = (1-σ_t)×x_0 + σ_t×ε`

当 `t=0` 时，`σ_0 = 0`，所以 `x_0 = x_0`（干净帧）。模型的 timestep embedding 也传入 0，"知道"当前输入是干净数据，生成的 K/V 代表干净帧的特征。

---

## 五、KV Cache 初始化

文件：[causal_inference.py:245-283](../pipeline/causal_inference.py)

### 5.1 `_initialize_kv_cache`

```python
def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override=None):
    kv_cache_size = kv_cache_size_override or (local_attn_size * frame_seq_length)
    
    for _ in range(30):  # 30 个 block
        kv_cache1.append({
            "k": torch.zeros([B, kv_cache_size, 12, 128], dtype, device),
            "v": torch.zeros([B, kv_cache_size, 12, 128], dtype, device),
            "global_end_index": torch.tensor([0], torch.long, device),
            "local_end_index":  torch.tensor([0], torch.long, device)
        })
    self.kv_cache1 = kv_cache1
```

- 初始全零（之后 direct_insert 会逐渐填充）
- `global_end_index = 0` 标志"尚未生成任何帧"

### 5.2 `_initialize_crossattn_cache`

```python
for _ in range(30):
    crossattn_cache.append({
        "k": torch.zeros([B, 512, 12, 128], dtype, device),
        "v": torch.zeros([B, 512, 12, 128], dtype, device),
        "is_init": False    # 首次 forward 时计算，之后复用
    })
```

`is_init=False` 时，`cross_attn` 层会计算并存储文本的 K/V，之后改为 `True` 直接复用。

### 5.3 `_set_all_modules_max_attention_size`

```python
# causal_inference.py:285-318
# 遍历 generator.model 所有子模块，把 max_attention_size 统一设置
for name, module in self.generator.model.named_modules():
    if hasattr(module, "max_attention_size"):
        setattr(module, "max_attention_size", target_size)
```

`max_attention_size = local_attn_size × 1560`（sink+window 的总上限）。

---

## 六、low_memory 模式

```python
# inference.py:62-63
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True  # 强制开启（代码里硬编码了）
```

low_memory=True 的效果：
1. **文本编码器**：通过 `DynamicSwapInstaller` 管理，编码时移到 GPU，完成后移回 CPU
2. **输出 buffer**：`output = zeros(..., device='cpu')` 存在 CPU，每帧生成后 `.to(output.device)` 拷回

```python
output_device = torch.device('cpu') if low_memory else noise.device
output = torch.zeros([B, F, C, H, W], device=output_device, ...)

# 每帧生成后：
output[:, frame_start:end] = denoised_pred.to(output.device)  # GPU→CPU 拷贝
```

---

## 七、分布式推理支持

```python
# inference.py:30-58
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", ...)
    
# 数据并行：每个 GPU 处理不同的 prompt
if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False)
else:
    sampler = SequentialSampler(dataset)
```

注意：LongLive 推理是**数据并行**（每卡处理不同 prompt），不是张量并行（单视频不跨卡）。

---

## 八、输出视频后处理

```python
# causal_inference.py:223-224
video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
video = (video * 0.5 + 0.5).clamp(0, 1)   # [-1,1] → [0,1]
```

VAE 解码输出范围 `[-1, 1]`，`×0.5+0.5` 映射到 `[0,1]`，再 clamp 防止异常值。

保存时乘以 255（inference.py）：

```python
video = 255.0 * torch.cat(all_video, dim=1)   # [0,1] → [0,255]
write_video(output_path, video[seed_idx], fps=16)
```

---

## 九、profiling 支持

Pipeline 内置了 CUDA event 计时，通过 `profile=True` 开启：

```python
video, latents = pipeline.inference(noise, text_prompts, profile=True)
# 输出：
# Profiling results:
#   - Initialization/caching time: 150 ms (0.2%)
#   - Diffusion generation time: 72000 ms (99.5%)
#     - Block 0 generation time: 600 ms
#     - Block 1 generation time: 590 ms
#   - VAE decoding time: 200 ms (0.3%)
#   - Total time: 72350 ms
```
