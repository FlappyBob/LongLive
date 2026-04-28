# 14 推理流程全程导读（带代码注释 + Shape 追踪）

> 以 `configs/longlive_inference.yaml` 为具体配置示例：
> - B=1, F_total=120 帧, F_block=3 帧/块, n_blocks=40
> - 分辨率 480×832 → 潜变量 60×104
> - local_attn_size=12, sink_size=3, context_noise=0
> - denoising_step_list=[1000, 750, 500, 250]（4步去噪）

---

## 完整调用链概览

```
inference.py
└── CausalInferencePipeline.inference()        pipeline/causal_inference.py:56
    ├── WanTextEncoder.forward()                utils/wan_wrapper.py:43
    ├── _initialize_kv_cache()                  pipeline/causal_inference.py:245
    ├── _initialize_crossattn_cache()           pipeline/causal_inference.py:271
    └── [外层循环 × 40块]
        └── [内层循环 × 4个去噪步]
            └── WanDiffusionWrapper.forward()   utils/wan_wrapper.py:280
                └── CausalWanModel._forward_inference()   wan/modules/causal_model.py:891
                    ├── patch_embedding                    causal_model.py:943
                    ├── time_embedding                     causal_model.py:960
                    ├── text_embedding                     causal_model.py:968
                    └── [× 30层] CausalWanAttentionBlock.forward()  causal_model.py:401
                        ├── CausalWanSelfAttention.forward()        causal_model.py:97
                        │   ├── Q/K/V 线性投影
                        │   ├── causal_rope_apply()                 causal_model.py:32
                        │   ├── KV Cache 管理（direct_insert/roll_and_insert）
                        │   └── attention()                         wan/modules/attention.py:139
                        ├── cross_attn（交叉注意力）
                        └── FFN
                    ├── CausalHead.forward()                causal_model.py:485
                    └── unpatchify()                        causal_model.py:1222
                └── _convert_flow_pred_to_x0()              wan_wrapper.py:231
        ├── scheduler.add_noise()（非最终步）   utils/scheduler.py:159
        └── Context Pass（t=0再跑一次）
    └── WanVAEWrapper.decode_to_pixel()         utils/wan_wrapper.py:96
```

---

## 第一步：入口 `inference.py`

```python
# inference.py:194-213

# 采样纯噪声作为起始输入
# sampled_noise.shape = [B=1, F=120, C=16, H_lat=60, W_lat=104]
sampled_noise = torch.randn(
    [config.num_samples, config.num_output_frames, 16, 60, 104],
    device=device, dtype=torch.bfloat16
)
# 为什么是 16? VAE 的潜空间维数 = 16
# 为什么是 60, 104? 480/8=60, 832/8=104（VAE 空间下采样 8x）

# 调用推理，返回视频像素和潜变量
# video.shape = [1, 120, 3, 480, 832]（像素空间，值域 [0,1]）
# latents.shape = [1, 120, 16, 60, 104]
video, latents = pipeline.inference(
    noise=sampled_noise,          # [1, 120, 16, 60, 104]
    text_prompts=prompts,         # ["a dog running..." ] × 1
    return_latents=True,
    low_memory=low_memory,        # True（强制开启 CPU 卸载文本编码器）
    profile=False,
)

# 转换为 [B, T, H, W, C] 格式用于保存
# current_video.shape = [1, 120, 480, 832, 3]
current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()

# video 乘以 255 得到 uint8
video = 255.0 * torch.cat(all_video, dim=1)  # [1, 120, 480, 832, 3]
write_video(output_path, video[seed_idx], fps=16)
```

---

## 第二步：`CausalInferencePipeline.inference()` 主函数

**文件**: `pipeline/causal_inference.py:56`

### 2.1 解析维度

```python
# causal_inference.py:76-78
batch_size, num_output_frames, num_channels, height, width = noise.shape
# = 1, 120, 16, 60, 104

assert num_output_frames % self.num_frame_per_block == 0
# 120 % 3 == 0 ✓

num_blocks = num_output_frames // self.num_frame_per_block
# = 120 // 3 = 40 块
```

### 2.2 文本编码

```python
# causal_inference.py:80-82
conditional_dict = self.text_encoder(
    text_prompts=text_prompts   # ["a dog running..."]
)
# 返回 {"prompt_embeds": [1, 512, 4096]}
# 512 = 最大 token 数, 4096 = UMT5-xxl 维度
```

**文本编码器内部** (`utils/wan_wrapper.py:43`)：

```python
# wan_wrapper.py:44-57
# Step 1: tokenize
ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
# ids.shape = [1, 512]  ← 固定序列长度 512，不足补 pad
# mask.shape = [1, 512]

# Step 2: 通过 UMT5-xxl 编码
seq_lens = mask.gt(0).sum(dim=1).long()  # 每个 prompt 的实际 token 数，如 [87]
context = self.text_encoder(ids, mask)   # [1, 512, 4096]

# Step 3: 把 padding 位置清零（避免 padding 影响后续注意力）
for u, v in zip(context, seq_lens):
    u[v:] = 0.0  # 真实 token 后面全置 0
# context = [1, 512, 4096], 实际内容在前 87 位
```

### 2.3 KV Cache 初始化

```python
# causal_inference.py:110-127
local_attn_cfg = 12  # from config.model_kwargs.local_attn_size

# kv_cache_size = 12 × 1560 = 18720 个 token 槽
# （12 帧容量，每帧 30×52=1560 个 token）
kv_cache_size = local_attn_cfg * self.frame_seq_length
# = 12 × 1560 = 18720

self._initialize_kv_cache(
    batch_size=1,
    dtype=torch.bfloat16,
    device=cuda:0,
    kv_cache_size_override=18720
)
```

**KV Cache 初始化内部** (`causal_inference.py:245`)：

```python
# causal_inference.py:261-269
kv_cache1 = []
for _ in range(30):   # 30 个 Transformer 块各有独立的缓存
    kv_cache1.append({
        # K 缓存: [B=1, cache_slots=18720, heads=12, head_dim=128]
        "k": torch.zeros([1, 18720, 12, 128], dtype=bfloat16, device=cuda),
        # V 缓存: 同形状
        "v": torch.zeros([1, 18720, 12, 128], dtype=bfloat16, device=cuda),
        # global_end_index: 当前已填入的绝对 token 位置（用于 RoPE 位置编码）
        # 初始=0，单调递增，永不倒退
        "global_end_index": torch.tensor([0], dtype=torch.long, device=cuda),
        # local_end_index: 当前 cache 数组中实际填入的位置（用于数组索引）
        # 初始=0，填满后 roll 后保持在 18720
        "local_end_index":  torch.tensor([0], dtype=torch.long, device=cuda)
    })

# 为何 12 heads × 128 head_dim?
# Wan2.1-T2V-1.3B: dim=1536, num_heads=12, head_dim=1536/12=128

self.kv_cache1 = kv_cache1  # 30 个字典组成的列表
```

**交叉注意力缓存** (`causal_inference.py:271`)：

```python
crossattn_cache = []
for _ in range(30):
    crossattn_cache.append({
        # 文本侧 K/V: [1, 512, 12, 128]
        # 512 = 文本 token 长度，第一次 forward 后缓存，后续直接复用
        "k": torch.zeros([1, 512, 12, 128], dtype=bfloat16, device=cuda),
        "v": torch.zeros([1, 512, 12, 128], dtype=bfloat16, device=cuda),
        "is_init": False  # 首次 forward 后变为 True，之后跳过文本编码
    })
```

### 2.4 外层帧块循环（40 次）

```python
# causal_inference.py:145-209
all_num_frames = [3] * 40   # 每块 3 帧，共 40 块

current_start_frame = 0     # 当前块起始帧号（以帧为单位）

for current_num_frames in all_num_frames:  # 每次 current_num_frames=3

    # 切出当前块的噪声
    # noisy_input.shape = [1, 3, 16, 60, 104]
    noisy_input = noise[:, current_start_frame : current_start_frame + 3]
    # noise 是全部 120 帧的随机噪声，每次切 3 帧

    # ─── 内层去噪循环（4步）───
    for index, current_timestep in enumerate([1000, 750, 500, 250]):
        # ...（见下节）

    # 保存本块结果到 output buffer
    # output.shape = [1, 120, 16, 60, 104]（整个视频的输出缓冲）
    output[:, current_start_frame:current_start_frame+3] = denoised_pred.to(output.device)

    # Context Pass：用干净帧 t=0 再跑一次，把干净 K/V 写入缓存
    # （见第九步）
    context_timestep = torch.ones_like(timestep) * 0  # context_noise=0
    self.generator(
        noisy_image_or_video=denoised_pred,  # [1, 3, 16, 60, 104] 已去噪的帧
        conditional_dict=conditional_dict,
        timestep=context_timestep,           # t=0，相当于"干净帧"
        kv_cache=self.kv_cache1,
        crossattn_cache=self.crossattn_cache,
        current_start=current_start_frame * 1560,  # 以 token 为单位的起始位置
    )

    current_start_frame += 3   # 移动到下一块
```

---

## 第三步：内层去噪循环（每块 4 次）

```python
# causal_inference.py:154-188

for index, current_timestep in enumerate([1000, 750, 500, 250]):

    # 构造 timestep 张量：每帧独立记录时间步（causal 模式）
    # timestep.shape = [B=1, F=3]，每个值都是 current_timestep
    timestep = torch.ones([1, 3], device=cuda, dtype=torch.int64) * current_timestep

    # 非最后一步（index < 3）：去噪后重新加噪
    if index < 3:
        _, denoised_pred = self.generator(
            noisy_image_or_video=noisy_input,    # [1, 3, 16, 60, 104]
            conditional_dict=conditional_dict,
            timestep=timestep,                   # [1, 3]
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * 1560
            # current_start 是绝对 token 偏移量
            # 第0块=0, 第1块=4680, ..., 第k块=k*4680
        )
        # denoised_pred.shape = [1, 3, 16, 60, 104]（x0 预测）

        # 重新加噪到下一个时间步
        next_timestep = [750, 500, 250][index]

        # flatten(0,1): [1,3,16,60,104] → [3, 16, 60, 104]
        # scheduler.add_noise: 在时间步 next_timestep 加噪
        # unflatten: [3, 16, 60, 104] → [1, 3, 16, 60, 104]
        noisy_input = self.scheduler.add_noise(
            denoised_pred.flatten(0, 1),                    # [3, 16, 60, 104]
            torch.randn_like(denoised_pred.flatten(0, 1)), # [3, 16, 60, 104] 新噪声
            next_timestep * torch.ones([3], dtype=torch.long)
        ).unflatten(0, (1, 3))
        # noisy_input.shape = [1, 3, 16, 60, 104]

    # 最后一步（index=3）：获取最终 x0
    else:
        _, denoised_pred = self.generator(...)
        # denoised_pred.shape = [1, 3, 16, 60, 104] ← 最终干净帧
```

**FlowMatchScheduler.add_noise 内部** (`utils/scheduler.py:159`)：

```python
# scheduler.py:159-176
def add_noise(self, original_samples, noise, timestep):
    # original_samples: [3, 16, 60, 104] ← 干净帧 x0
    # noise: [3, 16, 60, 104] ← 随机噪声 ε
    # timestep: [3] ← 下一步的时间步，如 750

    # 找到 timestep 在预定义时间表中的位置
    timestep_id = torch.argmin(
        (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    # timestep_id.shape = [3]，每帧对应时间表中的位置

    sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
    # sigma.shape = [3, 1, 1, 1]，广播覆盖 [C, H, W]
    # sigma 是噪声强度，shift=5 导致高时间步 sigma 更大

    # Flow Matching 前向过程公式：x_t = (1-σ) × x0 + σ × ε
    sample = (1 - sigma) * original_samples + sigma * noise
    # sample.shape = [3, 16, 60, 104]
    return sample.type_as(noise)
```

---

## 第四步：`WanDiffusionWrapper.forward()` 包装层

**文件**: `utils/wan_wrapper.py:280`

```python
# wan_wrapper.py:293-356
def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, ...):
    prompt_embeds = conditional_dict["prompt_embeds"]
    # prompt_embeds.shape = [1, 512, 4096]

    # causal 模式下每帧有独立时间步，不取平均
    input_timestep = timestep          # [1, 3]

    # ─── 调用核心 DiT 模型 ───
    # 关键：permute 把 [B, F, C, H, W] → [B, C, F, H, W]
    # noisy_image_or_video: [1, 3, 16, 60, 104]
    # 经 permute(0,2,1,3,4): [1, 16, 3, 60, 104]
    flow_pred = self.model(
        noisy_image_or_video.permute(0, 2, 1, 3, 4),  # [1, 16, 3, 60, 104]
        t=input_timestep,                              # [1, 3]
        context=prompt_embeds,                         # [1, 512, 4096]
        seq_len=self.seq_len,                          # 32760（历史原因保留的最大值）
        kv_cache=kv_cache,                             # 30个字典的列表
        crossattn_cache=crossattn_cache,
        current_start=current_start,                   # 当前块的 token 起始偏移
    ).permute(0, 2, 1, 3, 4)
    # 模型输出: [1, 16, 3, 60, 104]
    # 经 permute(0,2,1,3,4): [1, 3, 16, 60, 104]
    # flow_pred.shape = [1, 3, 16, 60, 104]

    # ─── flow_pred → x0 转换 ───
    # flatten(0,1): [1, 3, 16, 60, 104] → [3, 16, 60, 104]
    pred_x0 = self._convert_flow_pred_to_x0(
        flow_pred=flow_pred.flatten(0, 1),         # [3, 16, 60, 104]
        xt=noisy_image_or_video.flatten(0, 1),     # [3, 16, 60, 104]
        timestep=timestep.flatten(0, 1)            # [3]
    ).unflatten(0, flow_pred.shape[:2])
    # pred_x0.shape = [1, 3, 16, 60, 104]

    return flow_pred, pred_x0
    # 调用方用 _, denoised_pred = self.generator(...)
    # 所以实际用的是 pred_x0（x0 预测）
```

**flow_pred → x0 转换公式** (`wan_wrapper.py:231`)：

```python
# wan_wrapper.py:231-255
def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
    # flow_pred: [3, 16, 60, 104]  模型输出的速度场 v = ε - x0
    # xt: [3, 16, 60, 104]         当前含噪帧 x_t
    # timestep: [3]                 每帧时间步

    # 找到 sigma_t（此时间步的噪声强度）
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)   # [3, 1, 1, 1]

    # Flow Matching 逆推公式：x0 = x_t - σ_t × v
    # 其中 v = ε - x0（模型预测的速度方向）
    x0_pred = xt - sigma_t * flow_pred   # [3, 16, 60, 104]
    return x0_pred
```

---

## 第五步：`CausalWanModel._forward_inference()` 核心模型

**文件**: `wan/modules/causal_model.py:891`

### 5.1 Patch Embedding

```python
# causal_model.py:943-950
# 输入 x 为 [B, C, F, H, W] 格式的列表，每个元素对应一个 batch item
# x = [Tensor[16, 3, 60, 104]] （B=1，所以只有一个元素）

x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
# 对每个 u: shape = [16, 3, 60, 104]
# u.unsqueeze(0): [1, 16, 3, 60, 104]
# patch_embedding = Conv3d(in=16, out=1536, kernel=(1,2,2), stride=(1,2,2))
# 时间方向 stride=1（不合并帧），空间方向 stride=2（2×2 合并）
# 输出: [1, 1536, 3, 30, 52]
#   - 1536: 模型维度（dim）
#   - 3: 帧数不变（temporal stride=1）
#   - 30: 60/2=30（空间下采样 2x）
#   - 52: 104/2=52（空间下采样 2x）

# 记录 grid_sizes 用于 RoPE
grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
# grid_sizes.shape = [1, 3] → [[3, 30, 52]]
# 3D 网格：F=3 帧, H=30, W=52

x = [u.flatten(2).transpose(1, 2) for u in x]
# u: [1, 1536, 3, 30, 52]
# .flatten(2): [1, 1536, 3*30*52] = [1, 1536, 4680]
# .transpose(1, 2): [1, 4680, 1536]
# 4680 = 3 × 1560（每帧 1560 个 token，3 帧共 4680 个）

seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
# seq_lens = [4680]

x = torch.cat(x)
# x.shape = [1, 4680, 1536]  ← 正式进入 Transformer 处理的 token 序列
```

### 5.2 时间步嵌入（Time Embedding）

```python
# causal_model.py:960-963
# t.shape = [1, 3]，推理时 3 个值完全相同（均为 current_timestep）
# shape 设计成 [B,F] 是为了训练时支持每帧不同噪声级别，推理时不使用此能力
e = self.time_embedding(
    sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
# t.flatten(): [3]，值 = [t, t, t]（三帧同一个时间步）
# sinusoidal_embedding_1d(256, [3]): [3, 256]（正弦位置编码，freq_dim=256）
# time_embedding = Linear(256→1536) → SiLU → Linear(1536→1536)
# e.shape = [3, 1536]

e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
# self.time_projection = SiLU + Linear(1536 → 1536×6 = 9216)
# 经 time_projection: [3, 9216]
# .unflatten(1, (6, 1536)): [3, 6, 1536]
# .unflatten(0, (1, 3)): [1, 3, 6, 1536]
# e0.shape = [1, 3, 6, 1536]
# 6个通道：对应 Transformer 块内 AdaLN 的6个调制系数
```

### 5.3 文本嵌入（Text Embedding）

```python
# causal_model.py:968-973
# context = prompt_embeds = [1, 512, 4096]（来自文本编码器）
context = self.text_embedding(
    torch.stack([
        torch.cat(
            [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))
# 对每个 u: [512, 4096]（已是完整512长度，zero pad这里不起作用）
# text_embedding = Linear(4096→1536) → GELU → Linear(1536→1536)
# 映射: [1, 512, 4096] → [1, 512, 1536]
# context.shape = [1, 512, 1536]
```

### 5.4 30 层 Transformer 块循环

```python
# causal_model.py:997-1044
kwargs = dict(
    e=e0,                # [1, 3, 6, 1536] 时间调制
    seq_lens=seq_lens,   # [4680]
    grid_sizes=grid_sizes, # [[3, 30, 52]]
    freqs=self.freqs,    # [1024, 64]（复数 RoPE 频率表）
    context=context,     # [1, 512, 1536] 文本
    context_lens=None,
    block_mask=None,     # 推理时不用 FlexAttention
)

cache_update_infos = []  # 收集所有块的缓存更新计划（延迟写入）

for block_index, block in enumerate(self.blocks):  # 30 次
    kwargs.update({
        "kv_cache": kv_cache[block_index],         # 当前层的缓存字典
        "crossattn_cache": crossattn_cache[block_index],
        "current_start": current_start,            # 当前块起始 token 偏移
    })

    result = block(x, **kwargs)
    # result = (x_new, (current_end, local_end_index, cache_update_info))
    # 注意：此时 cache 还没有写入！只是收集了 update_info

    x, block_cache_update_info = result
    # x.shape = [1, 4680, 1536]（每层保持不变）
    cache_update_infos.append((block_index, block_cache_update_info))

# ─── 所有块处理完后，统一写入 KV Cache ───
# 这样设计是为了避免多 GPU 时的数据竞争
if kv_cache is not None:
    self._apply_cache_updates(kv_cache, cache_update_infos)
# 执行后，kv_cache[0..29] 中的 k, v 更新完毕
```

---

## 第六步：`CausalWanAttentionBlock.forward()`

**文件**: `wan/modules/causal_model.py:401`

```python
# causal_model.py:401-465
def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
            block_mask, kv_cache, crossattn_cache, current_start, ...):
    # x.shape = [1, 4680, 1536]
    # e.shape = [1, 3, 6, 1536]（时间调制向量，F=3 帧）

    num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
    # num_frames = 3, frame_seqlen = 4680 // 3 = 1560

    # ─── AdaLN 调制系数计算 ───
    # self.modulation: [1, 6, 1536] 可学习参数
    # self.modulation.unsqueeze(1): [1, 1, 6, 1536]
    # e: [1, 3, 6, 1536]
    # 相加后 chunk(6, dim=2) → 6个 [1, 3, 1, 1536] 张量
    e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    # e[0]: shift for self-attn  [1, 3, 1, 1536]
    # e[1]: scale for self-attn  [1, 3, 1, 1536]
    # e[2]: gate for self-attn   [1, 3, 1, 1536]
    # e[3]: shift for FFN
    # e[4]: scale for FFN
    # e[5]: gate for FFN

    # ─── 自注意力前的层归一化 + 调制 ───
    # self.norm1(x): [1, 4680, 1536]
    # .unflatten(1, (3, 1560)): [1, 3, 1560, 1536]
    # * (1 + e[1]): 对每帧 1560 个 token 施加不同的缩放
    #               e[1]=[1,3,1,1536] 广播到 [1,3,1560,1536]
    # + e[0]: 加上偏移
    # .flatten(1, 2): [1, 4680, 1536]
    self_attn_result = self.self_attn(
        (self.norm1(x).unflatten(dim=1, sizes=(3, 1560)) * (1 + e[1]) + e[0]).flatten(1, 2),
        # 输入形状: [1, 4680, 1536]
        seq_lens, grid_sizes, freqs, block_mask,
        kv_cache, current_start, cache_start, sink_recache_after_switch)

    y, cache_update_info = self_attn_result
    # y.shape = [1, 4680, 1536]（自注意力输出）

    # 残差连接 + 门控
    # y.unflatten(1, (3, 1560)): [1, 3, 1560, 1536]
    # * e[2]: 门控（gate）缩放，e[2]=[1,3,1,1536] 广播
    # .flatten(1, 2): [1, 4680, 1536]
    x = x + (y.unflatten(dim=1, sizes=(3, 1560)) * e[2]).flatten(1, 2)
    # x.shape = [1, 4680, 1536]

    # ─── 交叉注意力 + FFN ───
    def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
        # 交叉注意力：x 作 Q，context（文本）作 K/V
        x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache)
        # x.shape 不变 [1, 4680, 1536]
        # crossattn_cache 在第一次 forward 后会缓存文本的 K/V

        # FFN 前的 AdaLN
        # .unflatten(1,(3,1560)): [1, 3, 1560, 1536]
        # * (1 + e[4]) + e[3]: 调制
        # .flatten(1,2): [1, 4680, 1536]
        y = self.ffn(
            (self.norm2(x).unflatten(1, (3, 1560)) * (1 + e[4]) + e[3]).flatten(1, 2)
        )
        # FFN = Linear(1536→6144) → GELU → Linear(6144→1536)
        # y.shape = [1, 4680, 1536]

        x = x + (y.unflatten(1, (3, 1560)) * e[5]).flatten(1, 2)
        return x

    x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
    # x.shape = [1, 4680, 1536]

    # 返回: (更新后的 x, 缓存更新计划)
    return x, cache_update_info
```

---

## 第七步：`CausalWanSelfAttention.forward()` — 核心 KV Cache 机制

**文件**: `wan/modules/causal_model.py:97`

### 7.1 QKV 计算

```python
# causal_model.py:117-128
# x.shape = [1, 4680, 1536]
b, s, n, d = 1, 4680, 12, 128
# n = num_heads = 12, d = head_dim = 128

def qkv_fn(x):
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    # self.q = Linear(1536, 1536)
    # self.q(x): [1, 4680, 1536]
    # .view(1, 4680, 12, 128): [1, 4680, 12, 128]
    # norm_q: RMSNorm 对最后一维归一化

    k = self.norm_k(self.k(x)).view(b, s, n, d)  # [1, 4680, 12, 128]
    v = self.v(x).view(b, s, n, d)               # [1, 4680, 12, 128]（V 不归一化）
    return q, k, v

q, k, v = qkv_fn(x)
# q.shape = k.shape = v.shape = [1, 4680, 12, 128]
```

### 7.2 `causal_rope_apply()` — 旋转位置编码

**文件**: `wan/modules/causal_model.py:32`

```python
# causal_model.py:206-211
frame_seqlen = 30 * 52   # = 1560（每帧的 patch 数）
current_start_frame = current_start // frame_seqlen
# current_start=0 → current_start_frame=0（第0块）
# current_start=4680 → current_start_frame=3（第1块，帧号从3开始）

roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
roped_key   = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)
```

**`causal_rope_apply` 内部** (`causal_model.py:32`)：

```python
# causal_model.py:32-60
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    # x.shape = [1, 4680, 12, 128]（Q 或 K）
    # grid_sizes = [[3, 30, 52]]
    # freqs.shape = [1024, 64]（复数频率表，64=128/2个复数维度）
    # start_frame: 当前块的帧起始号（第k块=k*3）

    n, c = x.size(2), x.size(3) // 2
    # n = 12, c = 64（每个头的复数维度数）

    # 把 freqs 按时间/高度/宽度三个维度切分
    # c=64, c//3=21
    # [22, 21, 21] 分别对应时间、高度、宽度的频率分量
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # freqs[0]: [1024, 22]  时间维度频率（最多 1024 帧）
    # freqs[1]: [1024, 21]  高度维度频率（最多 1024 个 h patch）
    # freqs[2]: [1024, 21]  宽度维度频率（最多 1024 个 w patch）

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        # f=3, h=30, w=52（当前块的 3D patch 网格）
        seq_len = f * h * w  # = 3 × 30 × 52 = 4680

        # 把 Q/K 转为复数（相邻两个实数 → 一个复数）
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        # x[i, :4680]: [4680, 12, 128]
        # .reshape(4680, 12, 64, 2): [4680, 12, 64, 2]
        # view_as_complex: [4680, 12, 64] 复数

        # 构造 3D 频率：时间帧偏移 start_frame，高度从0，宽度从0
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            # freqs[0][0:3]: [3, 22] → .view(3,1,1,22) → .expand(3,30,52,22)
            # 关键：start_frame 让不同块的时间频率不同！
            # 第0块 start_frame=0，第1块 start_frame=3，第k块 start_frame=3k

            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            # freqs[1][:30]: [30, 21] → .view(1,30,1,21) → .expand(3,30,52,21)

            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            # freqs[2][:52]: [52, 21] → .view(1,1,52,21) → .expand(3,30,52,21)
        ], dim=-1).reshape(seq_len, 1, -1)
        # cat 结果: [3, 30, 52, 64]
        # .reshape(4680, 1, 64)

        # 旋转：每个向量元素乘以对应频率的复数（等价于旋转）
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i * freqs_i: [4680, 12, 64]复数 × [4680, 1, 64]复数 = [4680, 12, 64]复数
        # view_as_real: [4680, 12, 64, 2]
        # .flatten(2): [4680, 12, 128]

        output.append(x_i)

    return torch.stack(output).type_as(x)
    # output.shape = [1, 4680, 12, 128]（与输入形状相同，值已旋转）
```

**为什么 start_frame 是关键？**

RoPE 的核心是让相对位置编码正确：
- 第0块帧号 0-2 → `start_frame=0` → 时间频率取 `freqs[0][0:3]`
- 第1块帧号 3-5 → `start_frame=3` → 时间频率取 `freqs[0][3:6]`
- KV Cache 中保存的是旋转后的 K，位置编码已编码进去
- 后续帧的 Q 和历史帧的 K 点积时，自然得到正确的相对位置注意力分数

### 7.3 KV Cache 管理

每次 forward 分两种情况：

**情况 A：直接插入（前 12 块，cache 未满）**

```python
# causal_model.py:286-315

current_end = current_start + roped_query.shape[1]
# current_start=0, roped_query.shape[1]=4680
# current_end = 0 + 4680 = 4680（绝对 token 位置）

is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
# 第一次 forward: global_end_index=0, current_end=4680 → is_recompute=False

# 判断是否需要 roll（cache 是否溢出）
# 条件：local_end_index + num_new_tokens > kv_cache_size
# 初始: 0 + 4680 <= 18720 → 不溢出，走直接插入

local_end_index = kv_cache["local_end_index"].item() + (current_end - kv_cache["global_end_index"].item())
# = 0 + (4680 - 0) = 4680
local_start_index = local_end_index - num_new_tokens
# = 4680 - 4680 = 0

# 构建临时 K/V（不直接写入 cache）
temp_k = kv_cache["k"].clone()  # [1, 18720, 12, 128]
temp_v = kv_cache["v"].clone()

# 写入新的 K/V
temp_k[:, 0:4680] = roped_key    # [1, 4680, 12, 128]
temp_v[:, 0:4680] = v            # [1, 4680, 12, 128]
# 注意：写的是 roped_key（已经旋转编码过的），V 不旋转

cache_update_info = {
    "action": "direct_insert",
    "local_start_index": 0,
    "local_end_index": 4680,
    "write_start_index": 0,
    "write_end_index": 4680,
    "new_k": roped_key,    # [1, 4680, 12, 128]
    "new_v": v,            # [1, 4680, 12, 128]
    "current_end": 4680,
    "is_recompute": False
}
```

**情况 B：滚动插入（从第 13 块开始，cache 满后）**

```python
# causal_model.py:231-282
# 假设处理第 13 块（帧 36-38），此时 global_end=18720（12帧已满）
# current_start=18720, current_end=18720+4680=23400
# local_end_index=18720, num_new_tokens=4680

# 条件: 4680 + 18720 = 23400 > 18720=kv_cache_size → 需要 roll！

num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
# = 4680 + 18720 - 18720 = 4680（被驱逐的 token 数 = 1块=3帧）

sink_tokens = 3 * 1560  # = 4680（前3帧作为 sink，永久保留）

num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
# = 18720 - 4680 - 4680 = 9360（需要平移的 token 数 = 6帧）

# 临时 cache 中执行 roll
temp_k = kv_cache["k"].clone()  # [1, 18720, 12, 128]
temp_v = kv_cache["v"].clone()

# roll 操作：把 [sink+evicted : sink+evicted+rolled] 移到 [sink : sink+rolled]
# 即：丢弃帧 3-5（紧跟 sink 的那 3 帧），保留帧 6-11 的 K/V
temp_k[:, sink_tokens:sink_tokens+num_rolled_tokens] = \
    temp_k[:, sink_tokens+num_evicted_tokens:sink_tokens+num_evicted_tokens+num_rolled_tokens].clone()
# 等效：cache[4680:14040] ← cache[9360:18720]

temp_v[:, sink_tokens:sink_tokens+num_rolled_tokens] = \
    temp_v[:, sink_tokens+num_evicted_tokens:sink_tokens+num_evicted_tokens+num_rolled_tokens].clone()

# 插入新 K/V 到 [14040:18720]
local_start_index = 14040
local_end_index   = 18720
temp_k[:, 14040:18720] = roped_key   # 帧 36-38 的新 K
temp_v[:, 14040:18720] = v           # 帧 36-38 的新 V

# roll 后 cache 布局：
# [0    : 4680 ] = 帧 0-2  (sink，永久保留)
# [4680 : 14040] = 帧 6-11 (窗口历史，平移过来)
# [14040: 18720] = 帧 36-38 (刚写入的最新帧)
```

### 7.4 Attention 计算

```python
# causal_model.py:321-348

# sink 模式下（sink_size=3 → sink_tokens=4680）
sink_tokens = 4680

# 提取 sink K/V（帧 0-2，永久保留在最前面）
k_sink = temp_k[:, :4680]   # [1, 4680, 12, 128]
v_sink = temp_v[:, :4680]   # [1, 4680, 12, 128]

# 滑动窗口预算：允许在 sink 之外的 token 数
local_budget = self.max_attention_size - sink_tokens
# max_attention_size = local_attn_size × frame_seqlen = 12 × 1560 = 18720
# local_budget = 18720 - 4680 = 14040（9帧的 token 数）

# 提取窗口 K/V（最新的 local_budget 个 token）
local_start_for_window = max(sink_tokens, local_end_index - local_budget)
# = max(4680, 18720 - 14040) = max(4680, 4680) = 4680
k_local = temp_k[:, 4680:18720]   # [1, 14040, 12, 128]
v_local = temp_v[:, 4680:18720]   # [1, 14040, 12, 128]

# 拼接 sink + 窗口（形成完整的历史 K/V）
k_cat = torch.cat([k_sink, k_local], dim=1)  # [1, 18720, 12, 128]
v_cat = torch.cat([v_sink, v_local], dim=1)  # [1, 18720, 12, 128]

# FlashAttention
x = attention(
    roped_query,  # [1, 4680, 12, 128] ← Query（当前帧的 token）
    k_cat,        # [1, 18720, 12, 128] ← Key（sink + 窗口历史帧）
    v_cat         # [1, 18720, 12, 128] ← Value
)
# x.shape = [1, 4680, 12, 128]
```

**`attention()` 内部** (`wan/modules/attention.py:139`)：

```python
# attention.py:139-185
# 优先使用 FlashAttention 2/3，否则退回 scaled_dot_product_attention
# FlashAttention varlen_func 接受展平格式：
#   q.flatten(0,1): [4680, 12, 128]
#   k.flatten(0,1): [18720, 12, 128]
# 输出展平: [4680, 12, 128] → unflatten: [1, 4680, 12, 128]

# 不需要 causal=True！
# 因为 KV Cache 本身只保存历史帧，当前帧 Query 不会"看到未来"
# KV Cache 机制天然保证了因果性
```

### 7.5 输出

```python
# causal_model.py:351-358
x = x.flatten(2)   # [1, 4680, 12, 128] → [1, 4680, 1536]
x = self.o(x)      # Linear(1536, 1536) → [1, 4680, 1536]

# 返回给 CausalWanAttentionBlock
return x, (current_end, local_end_index, cache_update_info)
# x.shape = [1, 4680, 1536]
```

---

## 第八步：CausalHead + Unpatchify

```python
# causal_model.py:1046-1050（在 _forward_inference 末尾）

# e.shape = [3, 1536]（原始时间嵌入，不是 e0！）
# e.unflatten(0, t.shape): [1, 3, 1536]（t.shape=[1,3]）
# .unsqueeze(2): [1, 3, 1, 1536]
x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
```

**CausalHead 内部** (`causal_model.py:485`)：

```python
# causal_model.py:485-496
def forward(self, x, e):
    # x.shape = [1, 4680, 1536]
    # e.shape = [1, 3, 1, 1536]
    num_frames, frame_seqlen = 3, 1560

    e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
    # self.modulation: [1, 2, 1536]，.unsqueeze(1): [1, 1, 2, 1536]
    # e=[1, 3, 1, 1536]，相加后: [1, 3, 2, 1536]（广播）
    # .chunk(2, dim=2): 2个 [1, 3, 1, 1536]

    # 层归一化 + AdaLN 调制
    # self.norm(x): [1, 4680, 1536]
    # .unflatten(1, (3, 1560)): [1, 3, 1560, 1536]
    # * (1 + e[1]): [1, 3, 1, 1536] 广播 → [1, 3, 1560, 1536]
    # + e[0]: [1, 3, 1560, 1536]
    # self.head = Linear(1536 → 1×2×2×16=64)
    x = self.head(
        self.norm(x).unflatten(dim=1, sizes=(3, 1560)) * (1 + e[1]) + e[0]
    )
    # 输入: [1, 3, 1560, 1536]
    # 输出: [1, 3, 1560, 64]
    # 为什么是 64？ patch_size=(1,2,2), out_dim=16 → 1×2×2×16=64 个值/patch
    return x   # [1, 3, 1560, 64]
```

**Unpatchify** (`causal_model.py:1222`)：

```python
# causal_model.py:1222-1245
def unpatchify(self, x, grid_sizes):
    # x.shape = [1, 3, 1560, 64]
    # grid_sizes = [[3, 30, 52]]
    c = self.out_dim  # = 16（输出通道数，即 VAE 潜变量通道数）

    out = []
    for u, v in zip(x, grid_sizes.tolist()):
        # u = x[0] = [3, 1560, 64]（单个 batch item，迭代 batch 维度）
        # v = [3, 30, 52]
        # math.prod([3,30,52]) = 4680
        # u[:4680]: 取前 4680 个元素（3*1560=4680，等同于取全部）

        u = u[:math.prod(v)].view(*v, *self.patch_size, c)
        # .view(3, 30, 52, 1, 2, 2, 16): [3, 30, 52, 1, 2, 2, 16]
        # 含义：[F_patch, H_patch, W_patch, t_patch, h_patch, w_patch, C]
        # 1×2×2 是每个 patch 的时空尺寸，16 是通道

        u = torch.einsum('fhwpqrc->cfphqwr', u)
        # 把 patch 维度和空间维度交织
        # 输入: [f=3, h=30, w=52, p=1, q=2, r=2, c=16]
        # 输出: [c=16, f=3, p=1, h=30, q=2, w=52, r=2]

        u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
        # .reshape(16, 3×1, 30×2, 52×2) = .reshape(16, 3, 60, 104)
        # u.shape = [16, 3, 60, 104]  ← 重建为潜变量格式！

        out.append(u)

    return out   # [Tensor[16, 3, 60, 104]]
```

**回到 `_forward_inference`**：

```python
# causal_model.py:1049-1050
x = self.unpatchify(x, grid_sizes)
return torch.stack(x)
# torch.stack([Tensor[16, 3, 60, 104]])  → [1, 16, 3, 60, 104]
```

**回到 `WanDiffusionWrapper.forward()`**：

```python
# wan_wrapper.py:312-313
flow_pred = self.model(...).permute(0, 2, 1, 3, 4)
# [1, 16, 3, 60, 104].permute(0,2,1,3,4) = [1, 3, 16, 60, 104]
# flow_pred.shape = [1, 3, 16, 60, 104]
```

---

## 第九步：Context Pass（KV Cache 更新为干净帧）

**每块去噪完成后**，用干净帧再跑一次 forward，把干净的 K/V 写入缓存：

```python
# causal_inference.py:192-200
context_timestep = torch.ones_like(timestep) * 0   # t=0 表示"干净帧"
# context_timestep.shape = [1, 3]，全是 0

self.generator(
    noisy_image_or_video=denoised_pred,   # [1, 3, 16, 60, 104]（已去噪的干净帧）
    conditional_dict=conditional_dict,
    timestep=context_timestep,            # [1, 3]，全是 0
    kv_cache=self.kv_cache1,
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * 1560,
)
```

**为什么需要 Context Pass？**

在去噪循环中，每次 forward 写入 KV Cache 的是**含噪帧的 K/V**。
但下一块生成时，应该基于**干净帧的历史**。
Context Pass 用 t=0 重跑，覆盖掉 KV Cache 中的含噪 K/V，替换为干净帧的 K/V。

```
时间线（以 Block 0 为例）：
  t=1000: 含噪 K/V → cache（中间状态，会被覆盖）
  t=750:  含噪 K/V → cache（中间状态，会被覆盖）
  t=500:  含噪 K/V → cache（中间状态，会被覆盖）
  t=250:  含噪 K/V → cache（最后一步，最接近干净）
  t=0 (Context Pass): 干净帧 K/V → cache ← 最终写入，供 Block 1 读取
```

**注意**：去噪循环中每步都会写入 KV Cache（is_recompute 判断来保护 sink 区），
而 Context Pass 是最后一次写入，把 "最干净的" K/V 固化到 cache 中。

---

## 第十步：VAE 解码

**文件**: `utils/wan_wrapper.py:96`

```python
# causal_inference.py:222-224
# output.shape = [1, 120, 16, 60, 104]（全部 120 帧的潜变量）
video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
video = (video * 0.5 + 0.5).clamp(0, 1)
# video.shape = [1, 120, 3, 480, 832]，值域 [0, 1]
```

**VAE 解码内部** (`wan_wrapper.py:96`)：

```python
# wan_wrapper.py:97-117
def decode_to_pixel(self, latent, use_cache=False):
    # latent.shape = [1, 120, 16, 60, 104]

    # 转换为 VAE 期望的 [B, C, F, H, W] 格式
    zs = latent.permute(0, 2, 1, 3, 4)
    # zs.shape = [1, 16, 120, 60, 104]

    device, dtype = latent.device, latent.dtype
    scale = [self.mean.to(device, dtype), 1.0 / self.std.to(device, dtype)]
    # scale 用于反归一化（每个通道独立的均值和标准差）

    output = []
    for u in zs:
        # u.shape = [16, 120, 60, 104]（单个 batch item）
        # u.unsqueeze(0): [1, 16, 120, 60, 104]

        output.append(
            self.model.decode(u.unsqueeze(0), scale)
            # VAE decode: [1, 16, 120, 60, 104] → [1, 3, 120, 480, 832]
            # 时间上采样 ×1（latent 1帧对应像素 1帧）
            # 空间上采样 ×8（60→480, 104→832）
            .float().clamp_(-1, 1).squeeze(0)
            # .squeeze(0): [3, 120, 480, 832]，值域 [-1, 1]
        )

    output = torch.stack(output, dim=0)
    # [1, 3, 120, 480, 832]

    output = output.permute(0, 2, 1, 3, 4)
    # [1, 120, 3, 480, 832]  ← [B, T, C, H, W] 格式
    return output
```

**回到 `inference()`**：
```python
video = (video * 0.5 + 0.5).clamp(0, 1)
# [-1, 1] → [0, 1]：将 VAE 输出归一化到合法像素范围
# video.shape = [1, 120, 3, 480, 832]
```

---

## KV Cache 状态演进（随帧块变化）

```
帧块    global_end  local_end  cache 布局（以帧为单位描述）
──────  ──────────  ─────────  ──────────────────────────────────────
初始         0           0     全0
Block 0   4680        4680     [槽0:3  ] = 帧 0-2   (Context Pass后为干净帧K/V)
Block 1   9360        9360     [槽0:3  ] = 帧 0-2
                                [槽3:6  ] = 帧 3-5
Block 2   14040       14040    [槽0:3  ] = 帧 0-2
                                [槽3:9  ] = 帧 3-8
Block 3   18720       18720    [槽0:3  ] = 帧 0-2   (sink，以下是窗口满了)
                                [槽3:12 ] = 帧 3-11
Block 4   23400       18720    ROLL！驱逐帧 3-5
                                [槽0:3  ] = 帧 0-2   (sink 不动)
                                [槽3:9  ] = 帧 6-11  (平移过来)
                                [槽9:12 ] = 帧 12-14 (新写入)
Block 5   28080       18720    ROLL！驱逐帧 6-8
                                [槽0:3  ] = 帧 0-2   (sink 不动)
                                [槽3:9  ] = 帧 9-14  (平移)
                                [槽9:12 ] = 帧 15-17 (新写入)
...
Block 39  187200      18720    最终状态：
                                [槽0:3  ] = 帧 0-2   (sink，始终保留)
                                [槽3:12 ] = 帧 111-119 (最近9帧)
```

注意：`global_end_index` 反映绝对位置（用于 RoPE），`local_end_index` 反映 cache 数组索引（用于读写）。一旦 cache 满，local_end 稳定在 18720，global_end 继续增长。

---

## 完整 Shape 变换速查表

```
阶段                              形状                         注释
───────────────────────────────────────────────────────────────────────
采样噪声 (inference.py)            [1, 120, 16, 60, 104]        全视频 120 帧噪声
当前块噪声切片                      [1,   3, 16, 60, 104]        当前去噪的 3 帧
permute → 模型输入                 [1,  16,  3, 60, 104]        [B,C,F,H,W] 格式
patch_embedding (Conv3d)          [1,1536,  3, 30,  52]        stride=(1,2,2)
flatten + transpose               [1,4680,1536]                 4680=3×30×52
Q/K/V 线性投影                     [1,4680, 12, 128]             12头, 每头128维
causal_rope_apply 后              [1,4680, 12, 128]             形状不变，值已旋转
attention (Q vs 历史K/V)          Q:[1,4680,12,128]
                                  K:[1,18720,12,128]            18720=12帧×1560
                                  输出:[1,4680,12,128]
自注意力输出 flatten               [1,4680,1536]                 12×128=1536
经过30层 Transformer               [1,4680,1536]                 形状全程不变
CausalHead unflatten + modulation [1,3,1560,1536]              按帧调制
CausalHead linear (1536→64)       [1,3,1560,64]                64=1×2×2×16
unpatchify reshape                [1,3,1560,64] → [16,3,60,104] 逆 patch 操作
torch.stack                       [1,16,3,60,104]              batch 维度
permute → flow_pred               [1,3,16,60,104]              [B,F,C,H,W]
_convert_flow_pred_to_x0          [1,3,16,60,104]              x0 预测
output 缓冲（120帧合计）            [1,120,16,60,104]            全视频潜变量
VAE permute                       [1,16,120,60,104]            [B,C,F,H,W]
VAE decode                        [1,3,120,480,832]            像素空间 RGB
permute → video                   [1,120,3,480,832]            [B,T,C,H,W]
× 0.5 + 0.5                      [1,120,3,480,832]            值域 [0, 1]
× 255 → write_video               [120, 480, 832, 3]           uint8 MP4
```

---

## 与训练的核心区别

| 特性 | 推理（本文） | 训练（参考） |
|------|-------------|-------------|
| KV Cache | 有（保存历史帧 K/V）| 无（用 FlexAttention BlockMask） |
| 注意力实现 | FlashAttention + sink/window concat | FlexAttention (compiled) |
| 因果性保证 | KV Cache 本身不含未来帧 | BlockMask 显式屏蔽未来 |
| 去噪步数 | 4步（多步迭代） | 1步（Self-Forcing） |
| Context Pass | 有（用干净帧更新 cache）| 无 |
| RoPE 函数 | causal_rope_apply（start_frame 偏移）| rope_apply（从0开始）|
| 帧块处理 | 顺序（一块块生成）| 并行（全部帧同时训练）|
