# LongLive 代码库全貌

> 本文档从名词、动词、引擎、点火钥匙四个角度对 LongLive 进行系统梳理，附模块结构与核心调用图。

---

## 一、名词表（概念词汇）

| 名词 | 含义 |
|------|------|
| **Latent Frame（潜帧）** | 经 VAE 编码后的视频帧，形状 `[C=16, H/8, W/8]`。1 帧 = 1560 个 token（60×104/patch(2,2) = 30×52 = 1560）。 |
| **KV Cache（键值缓存）** | 每个 Transformer Block 独立维护的 `{"k": [B, cache_size, 12, 128], "v": ..., "global_end_index", "local_end_index"}` 字典，用于避免历史帧重复计算。 |
| **Frame Sink（帧锚点）** | KV Cache 中永久保留的前 `sink_size` 帧（默认 3），无论 cache 满多少次都不会被驱逐，保持全局语义一致性。 |
| **Local Attention Window** | `local_attn_size`（默认 12 帧）控制的滑动窗口，每帧只关注最近 12 帧而非全部历史，内存复杂度从 O(n²) 降为 O(n·w)。 |
| **Flow Matching** | Wan2.1 使用的生成范式：`flow_pred = noise - x0`，对应速度场。`xt = (1-σt)·x0 + σt·noise`。 |
| **x0 prediction（clean 预测）** | 从噪声帧 `xt` 和 `flow_pred` 反推出的 clean latent：`x0 = xt - σt · flow_pred`。 |
| **Denoising Step List** | 推理时的多步去噪序列，例如 `[1000, 750, 500, 250]`，每帧依次经过多次 denoise 迭代。 |
| **Context Noise** | 帧生成完毕后，以极小噪声（`t ≈ 0`）重跑 transformer，将 **干净帧** 写入 KV Cache，供后续帧作为上下文参考。 |
| **Block Mask** | FlexAttention 编译的块级因果掩码：同 chunk 内全局可见，跨 chunk 只看过去。 |
| **RoPE（旋转位置编码）** | 3D RoPE，分别对时间 (T)、高度 (H)、宽度 (W) 维度独立编码频率，`causal_rope_apply` 支持从任意 `start_frame` 开始。 |
| **Generator（生成器）** | 被训练的 `CausalWanModel`（1.3B），使用因果注意力逐帧生成，参数梯度开启。 |
| **Real Score / Teacher（真实分布）** | 冻结的 `Wan2.1-T2V-14B`，提供真实数据分布的梯度指导，`requires_grad_(False)`。 |
| **Fake Score / Critic（假分布）** | 与 Generator 同架构的辅助模型，拟合生成器产生的分布，`requires_grad_(True)`。 |
| **DMD（分布匹配蒸馏）** | 核心训练损失，通过对比 fake_score 与 real_score 的预测差来计算 KL 梯度，使生成器向真实分布靠拢。论文：arxiv/2311.18828。 |
| **Streaming Training（流式训练）** | 将长视频（240 帧）切成 21 帧的 chunk，逐 chunk 生成并蒸馏，KV Cache 跨 chunk 持续保留，避免一次性 OOM。 |
| **FSDP** | PyTorch Fully Sharded Data Parallel，Generator/Critic/TextEncoder 均用 FSDP 包裹分片。 |
| **LoRA** | Low-Rank Adaptation，训练阶段插入秩 256 的低秩适配器，仅训练 LoRA 权重，冻结基础模型。 |

---

## 二、动词表（核心操作）

| 动词 | 发生在哪里 | 做什么 |
|------|-----------|--------|
| **encode** | `WanVAEWrapper.encode_to_latent` | 像素帧 `[B,C,F,H,W]` → latent `[B,F,16,H/8,W/8]` |
| **decode** | `WanVAEWrapper.decode_to_pixel` | latent → 像素帧，支持分块解码防 OOM |
| **tokenize/embed** | `WanTextEncoder.forward` | 文本 → UMT5-XXL → 4096-dim context embeddings |
| **add_noise** | `FlowMatchScheduler.add_noise` | `xt = (1-σt)·x0 + σt·noise`，在 latent 上按 timestep 加噪 |
| **denoise** | `WanDiffusionWrapper.forward` | 对 `noisy_input` 运行 Transformer，预测 `flow_pred` 和 `x0` |
| **cache_update** | `CausalWanModel._apply_cache_updates` | 把新帧的 K/V 写入 KV Cache；满时先 roll（左移驱逐旧帧），再插入 |
| **recache** | `InteractiveCausalInferencePipeline._recache_after_switch` | prompt 切换后，用新 prompt 重跑历史帧，刷新 KV Cache |
| **unroll** | `SelfForcingTrainingPipeline.inference_with_trajectory` | 从纯噪声自回归展开生成整段视频，逐帧 denoise + cache_update |
| **distill** | `DMD.generator_loss` / `DMD.critic_loss` | 先 unroll，再用 DMD 损失对 Generator/Critic 反向传播 |
| **roll & evict** | `CausalWanSelfAttention.forward` | KV Cache 满时将 sink 之后的旧 tokens 左移（roll），腾出位置写新帧 |

---

## 三、引擎（核心执行单元）

```
┌─────────────────────────────────────────────────────────────────┐
│                    LongLive 引擎层级                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  CausalWanModel  (wan/modules/causal_model.py)           │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  [×30] CausalWanAttentionBlock                  │    │    │
│  │  │  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ │    │    │
│  │  │  │ CausalWan   │ │WanCrossAttn  │ │  FFN     │ │    │    │
│  │  │  │ SelfAttn    │ │(text context)│ │          │ │    │    │
│  │  │  │ + KV Cache  │ │              │ │          │ │    │    │
│  │  │  └─────────────┘ └──────────────┘ └──────────┘ │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  CausalHead  (输出 unpatchify 到 latent)         │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ▲                                       │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │  WanDiffusionWrapper  (utils/wan_wrapper.py)            │    │
│  │  flow_pred → x0 转换 │ FlowMatchScheduler               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ▲                                       │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │  Pipeline 层  (pipeline/)                               │    │
│  │  CausalInferencePipeline                                │    │
│  │  ├── InteractiveCausalInferencePipeline  (多提示词)      │    │
│  │  ├── SwitchCausalInferencePipeline  (单次切换)           │    │
│  │  └── SelfForcingTrainingPipeline  (训练展开)             │    │
│  │       └── StreamingTrainingPipeline  (流式训练展开)       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ▲                                       │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │  Model 层  (model/)                                     │    │
│  │  BaseModel  →  SelfForcingModel  →  DMD  →  DMDSwitch   │    │
│  │  generator + real_score + fake_score + text_encoder + vae│    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ▲                                       │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │  Trainer  (trainer/distillation.py)                     │    │
│  │  分布式初始化、FSDP、LoRA、优化器、checkpoint、日志       │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、点火钥匙（入口文件）

| 入口 | 文件 | 触发的核心链路 |
|------|------|----------------|
| **单提示词推理** | `inference.py` → `inference.sh` | `CausalInferencePipeline.inference()` |
| **多提示词交互推理** | `interactive_inference.py` → `interactive_inference.sh` | `InteractiveCausalInferencePipeline.inference()` |
| **阶段一训练（初始化）** | `train.py` → `train_init.sh` | `Trainer.__init__` + `Trainer.train()` + `DMD.generator_loss/critic_loss` |
| **阶段二训练（长调优）** | `train.py` → `train_long.sh` | 同上，但启用 `streaming_training=True`，使用 `StreamingTrainingModel` |

---

## 五、主要模块 Call Graph（推理路径）

```
inference.py
└── CausalInferencePipeline.inference(noise, text_prompts)
    ├── WanTextEncoder.forward(text_prompts)
    │   └── UMT5-XXL encoder → prompt_embeds [B, 512, 4096]
    ├── _initialize_kv_cache()          # 初始化 30 块 kv_cache
    ├── _initialize_crossattn_cache()   # 初始化 30 块 crossattn_cache
    └── [loop per frame block]
        ├── [loop per denoising step]
        │   └── WanDiffusionWrapper.forward(noisy_input, cond, timestep, kv_cache)
        │       └── CausalWanModel._forward_inference(x, t, context, ...)
        │           ├── patch_embedding(x)         # Conv3d patchify
        │           ├── time_embedding(t)           # sinusoidal + MLP
        │           ├── text_embedding(context)     # Linear proj
        │           └── [×30 blocks] CausalWanAttentionBlock.forward
        │               ├── CausalWanSelfAttention.forward  ← KV Cache 读写
        │               │   ├── causal_rope_apply(q/k, start_frame)
        │               │   ├── [roll_and_insert / direct_insert]
        │               │   └── attention(q, k_cat, v_cat)  ← FA2/FA3/SDPA
        │               ├── WanCrossAttention.forward(context)
        │               └── FFN
        │           ├── _apply_cache_updates(kv_cache, updates)
        │           └── CausalHead → unpatchify → latent
        │       └── _convert_flow_pred_to_x0 → pred_x0
        └── [context pass] re-run with t≈0 to write clean KV Cache
    └── WanVAEWrapper.decode_to_pixel(output_latents)
        └── _video_vae.decode → 像素视频 [B,T,C,H,W]
```

```
trainer/distillation.py
└── Trainer.train()  [无限循环]
    ├── fwdbwd_one_step / fwdbwd_one_step_streaming
    │   ├── WanTextEncoder → conditional_dict
    │   ├── [train_generator=True]
    │   │   └── DMD.generator_loss
    │   │       ├── _run_generator → SelfForcingTrainingPipeline.inference_with_trajectory
    │   │       │   └── [逐帧 unroll + context KV update]
    │   │       └── compute_distribution_matching_loss
    │   │           ├── fake_score(noisy_latent)  ← Critic 预测
    │   │           ├── real_score(noisy_latent)  ← Teacher 预测
    │   │           └── grad = fake - real; loss = MSE(x0, x0 - grad)
    │   └── [train_generator=False]
    │       └── DMD.critic_loss
    │           ├── generator unroll (no grad)
    │           └── fake_score denoising loss (grad)
    ├── loss.backward()
    ├── clip_grad_norm_
    └── optimizer.step()
```

---

## 六、关键超参数速查

| 参数 | 训练默认值 | 推理默认值 | 含义 |
|------|-----------|-----------|------|
| `local_attn_size` | 12 | 12 | 滑动窗口帧数，-1 为全局 |
| `sink_size` | 3 | 3 | 锚定帧数，永不驱逐 |
| `num_frame_per_block` | 3 | 3 | 每次生成的帧数块 |
| `denoising_step_list` | [1000,750,500,250] | [1000,750,500,250] | 每帧的去噪步骤 |
| `context_noise` | 0 | 0 | KV Cache 更新时的噪声强度 |
| `streaming_chunk_size` | 21 | N/A | 流式训练每个 chunk 的帧数 |
| `frame_seq_length` | 1560 | 1560 | 每帧的 token 数（硬编码）|

---

## 七、数据形状备忘

```
pixel video:   [B, T, C=3, H=480, W=832]
latent video:  [B, T, C=16, H=60, W=104]
1 latent frame token count: 60/2 × 104/2 = 30 × 52 = 1560
KV cache per block: [B, cache_tokens, n_heads=12, head_dim=128]
text context: [B, seq_len=512, text_dim=4096] → [B, 512, dim=2048]
```
