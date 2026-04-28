# LongLive 总览：名词·动词·引擎·点火钥匙 & Call Graph

> 本文档是重写前的"地图"，帮助你在代码海洋中定位方向。

---

## 一、名词表（Things）

| 名词 | 含义 | 代码位置 |
|------|------|---------|
| **Latent Frame（潜帧）** | VAE 压缩后的视频帧，shape `[C=16, H=60, W=104]`，1帧对应 1560 个 token | `utils/wan_wrapper.py` |
| **Token** | Latent 经 patch_embedding 后的最小单元，每帧 `60/2 × 104/2 = 1560` 个 | `wan/modules/causal_model.py:946` |
| **KV Cache** | 存储历史帧 Key/Value 的张量字典列表，每个 block 一份 | `pipeline/causal_inference.py:261` |
| **KV Cache (local_end_index)** | Cache 数组里当前有效数据的末尾位置 | `wan/modules/causal_model.py:244` |
| **KV Cache (global_end_index)** | 全局已生成的 token 数（绝对坐标），只增不减 | `wan/modules/causal_model.py:288` |
| **Frame Sink** | 前 N 帧永久占据 cache 开头，不被驱逐 | `wan/modules/causal_model.py:214` |
| **Context Pass** | 帧生成后用干净帧（t=0）再跑一次 forward，仅为更新 KV Cache | `pipeline/causal_inference.py:192` |
| **KV-Recache** | prompt 切换后，用新 prompt 重写历史帧的 KV Cache | `pipeline/interactive_causal_inference.py:34` |
| **BlockMask** | FlexAttention 的稀疏注意力掩码，实现因果 + 滑动窗口约束 | `wan/modules/causal_model.py:636` |
| **denoising_step_list** | 每帧的多步去噪 timestep 序列，如 `[1000,750,500,250]` | `pipeline/causal_inference.py:33` |
| **flow_pred** | 模型输出的"噪声方向"向量（Flow Matching 的速度场） | `utils/wan_wrapper.py:231` |
| **pred_x0** | 从 flow_pred 换算出的"干净帧"预测 | `utils/wan_wrapper.py:347` |
| **sigma_t** | 对应 timestep 的噪声强度，Flow Matching 公式 `x_t = (1-σ)x_0 + σ·ε` | `utils/wan_wrapper.py:254` |
| **CausalWanModel** | 魔改后的 Wan DiT backbone，支持 KV Cache 和因果注意力 | `wan/modules/causal_model.py:499` |
| **WanDiffusionWrapper** | 封装 CausalWanModel，负责 flow↔x0 转换、scheduler 绑定 | `utils/wan_wrapper.py:171` |
| **WanTextEncoder** | UMT5-XXL 文本编码器，输出 `[B, 512, 4096]` 的 prompt embedding | `utils/wan_wrapper.py:16` |
| **WanVAEWrapper** | 视频 VAE，像素↔潜变量互转 | `utils/wan_wrapper.py:60` |
| **DMD** | Distribution Matching Distillation，训练用的蒸馏损失模块 | `model/dmd.py:14` |
| **LoRA** | 低秩适配器，阶段二训练的参数高效微调 | `utils/lora_utils.py` |
| **FSDP** | Fully Sharded Data Parallel，多卡训练时的显存优化 | `utils/distributed.py` |

---

## 二、动词表（Actions）

| 动词/操作 | 含义 | 代码位置 |
|-----------|------|---------|
| **encode** | 像素帧 → latent（VAE 编码） | `wan_wrapper.py:80` |
| **decode_to_pixel** | latent → 像素帧（VAE 解码） | `wan_wrapper.py:96` |
| **patch_embedding** | latent → token 序列，Conv3d(16,2048,kernel=(1,2,2)) | `causal_model.py:587` |
| **unpatchify** | token 序列 → latent（patch_embedding 的逆操作） | `causal_model.py:1222` |
| **causal_rope_apply** | 给 Q/K 加上带绝对位置偏移的旋转位置编码 | `causal_model.py:32` |
| **roll_and_insert** | KV Cache 满时左移驱逐旧帧、插入新帧 | `causal_model.py:253` |
| **context_pass** | 用 t=0 干净帧更新 KV Cache（副作用 only） | `causal_inference.py:192` |
| **_recache_after_switch** | prompt 切换后重算历史帧 K/V | `interactive_causal_inference.py:34` |
| **add_noise** | 根据 scheduler 给 x0 加噪，用于下一去噪步 | `causal_inference.py:173` |
| **_convert_flow_pred_to_x0** | flow_pred + x_t + σ → x0 | `wan_wrapper.py:231` |
| **_apply_cache_updates** | 所有 block forward 结束后批量写回 KV Cache | `causal_model.py:837` |
| **generate_chunk_with_cache** | 流式训练中逐 chunk 生成并累积 KV Cache | `pipeline/streaming_training.py:73` |

---

## 三、引擎（Core Engines）

LongLive 有三个主要"引擎"，各司其职：

```
┌─────────────────────────────────────────────────────────┐
│                    引擎一：生成引擎                        │
│  CausalWanModel (_forward_inference)                     │
│  - 30层 CausalWanAttentionBlock                          │
│  - 每层：因果自注意力(KV Cache) + 交叉注意力(文本) + FFN  │
│  - FlexAttention (BlockMask 驱动)                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    引擎二：调度引擎                        │
│  FlowMatchScheduler                                      │
│  - 管理 1000步 sigma 曲线                                 │
│  - add_noise: x_0 → x_t                                 │
│  - 推理时按 denoising_step_list 跳步执行                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    引擎三：蒸馏引擎（训练）                 │
│  DMD (Distribution Matching Distillation)               │
│  - generator: 学生模型（产生视频）                         │
│  - fake_score: 评判模型（判断真实性）                      │
│  - KL散度梯度驱动 generator 逼近真实分布                   │
└─────────────────────────────────────────────────────────┘
```

---

## 四、点火钥匙（Entry Points）

| 入口 | 用途 | 关键参数 |
|------|------|---------|
| `inference.py` | 单 prompt 长视频生成 | `--config_path configs/longlive_inference.yaml` |
| `interactive_inference.py` | 多 prompt 交互式生成 | `--config_path configs/longlive_interactive_inference.yaml` |
| `train.py` | 训练（需要 torchrun 多卡） | `--config_path configs/longlive_train_init.yaml` |
| `inference.sh` | 封装 inference.py 的 shell 脚本 | 读取 YAML 配置 |
| `interactive_inference.sh` | 封装 interactive_inference.py | 读取 YAML 配置 |

---

## 五、主要模块结构

```
LongLive/
├── inference.py              ← 推理入口（点火钥匙）
├── interactive_inference.py  ← 交互推理入口
├── train.py                  ← 训练入口
│
├── pipeline/                 ← 推理/训练流程协调器
│   ├── causal_inference.py   ← 单 prompt 推理 Pipeline
│   ├── interactive_causal_inference.py  ← 多 prompt 推理 Pipeline
│   ├── self_forcing_training.py         ← SelfForcing 训练 Pipeline
│   ├── streaming_training.py            ← 流式长视频训练 Pipeline
│   └── streaming_switch_training.py     ← Switch 版流式训练
│
├── model/                    ← 模型定义（训练用）
│   ├── base.py              ← SelfForcingModel 基类
│   ├── dmd.py               ← DMD 蒸馏损失
│   ├── dmd_switch.py        ← DMDSwitch（含 prompt 切换）
│   └── streaming_training.py ← 流式训练模型封装
│
├── trainer/
│   └── distillation.py      ← ScoreDistillationTrainer（训练主循环）
│
├── wan/                      ← Wan 基础模型（修改版）
│   └── modules/
│       ├── causal_model.py   ← ★ 核心：CausalWanModel（因果注意力+KV Cache）
│       ├── model.py          ← 原版 WanModel（非因果，训练初始化用）
│       ├── attention.py      ← FlashAttention 封装
│       ├── vae.py            ← 视频 VAE
│       └── t5.py             ← UMT5-XXL 文本编码器
│
└── utils/
    ├── wan_wrapper.py        ← ★ WanDiffusionWrapper/TextEncoder/VAE 封装
    ├── scheduler.py          ← FlowMatchScheduler
    ├── lora_utils.py         ← LoRA 配置与加载
    ├── memory.py             ← 显存管理、DynamicSwap
    └── distributed.py        ← FSDP、EMA 工具
```

---

## 六、完整 Call Graph（推理路径）

```
inference.py
│
├── CausalInferencePipeline.__init__
│   ├── WanDiffusionWrapper.__init__          # 加载 CausalWanModel
│   ├── WanTextEncoder.__init__               # 加载 UMT5-XXL
│   └── WanVAEWrapper.__init__               # 加载 VAE
│
└── CausalInferencePipeline.inference(noise, text_prompts)
    │
    ├── WanTextEncoder.forward(text_prompts)
    │   └── UMT5.forward → prompt_embeds [B,512,4096]
    │
    ├── _initialize_kv_cache()                # 分配 30×{k,v,indices} 张量
    ├── _initialize_crossattn_cache()         # 分配 30×{k,v,is_init} 张量
    │
    └── for each frame_block:                 # 外层：逐帧块循环
        │
        ├── for each timestep in denoising_step_list:   # 内层：多步去噪
        │   │
        │   └── WanDiffusionWrapper.forward(noisy, cond, t, kv_cache)
        │       ├── CausalWanModel._forward_inference(x, t, context, ...)
        │       │   ├── patch_embedding(x)              # latent → token
        │       │   ├── time_embedding(t) → e           # 时间条件
        │       │   ├── text_embedding(context) → ctx   # 文本条件
        │       │   └── for each block in 30 blocks:
        │       │       └── CausalWanAttentionBlock.forward(x, e, ctx, kv_cache[i])
        │       │           ├── CausalWanSelfAttention.forward(x, kv_cache[i])
        │       │           │   ├── q,k,v = qkv_fn(x)
        │       │           │   ├── causal_rope_apply(q, start_frame=N)
        │       │           │   ├── causal_rope_apply(k, start_frame=N)
        │       │           │   ├── [roll_and_insert 或 direct_insert] → temp_k,v
        │       │           │   └── attention(q, cat[sink,window]) → x
        │       │           ├── cross_attn(x, context, crossattn_cache)
        │       │           └── ffn(x)
        │       │   └── _apply_cache_updates(kv_cache, infos)   # 批量写回
        │       │   └── head(x) → unpatchify → [C,F,H,W]
        │       └── _convert_flow_pred_to_x0 → pred_x0
        │
        ├── [非最后步] add_noise(pred_x0) → noisy_input (下一步输入)
        │
        ├── output[frame] = denoised_pred      # 记录干净帧
        │
        └── [context pass] WanDiffusionWrapper.forward(clean, t=0, kv_cache)
            └── 只为写入干净帧的 K/V，输出丢弃
    │
    └── WanVAEWrapper.decode_to_pixel(latents) → video [0,1]
```

---

## 七、完整 Call Graph（交互推理路径 - prompt 切换时）

```
interactive_causal_inference.py
│
└── InteractiveCausalInferencePipeline.inference(noise, text_prompts_list, switch_frame_indices)
    │
    ├── WanTextEncoder(prompts) × N_segments → cond_list
    │
    ├── 初始化 KV Cache / CrossAttn Cache
    │
    └── for each frame_block:
        │
        ├── [if 到达切换点] _recache_after_switch(output, frame_idx, new_cond)
        │   ├── CrossAttn Cache 清零 (is_init=False)
        │   ├── [if not global_sink] KV Cache 清零
        │   └── WanDiffusionWrapper.forward(历史干净帧, new_cond, t=0, kv_cache)
        │       └── 用新 prompt 的语境重写历史帧的 K/V
        │
        ├── [去噪循环，同单 prompt 推理]
        │
        └── [context pass，同单 prompt 推理]
```

---

## 八、完整 Call Graph（流式训练路径）

```
train.py → ScoreDistillationTrainer.train()
│
└── for each training step:
    │
    └── DMD.forward(noise, conditional_dict)
        │
        ├── [warmup] StreamingTrainingPipeline.generate_chunk_with_cache(
        │              noise=chunk_noise, requires_grad=False)
        │   └── 生成 chunk0，写入 KV Cache，梯度关闭
        │
        ├── [training] StreamingTrainingPipeline.generate_chunk_with_cache(
        │               noise=chunk_noise, requires_grad=True)
        │   └── 生成 chunk1+，梯度开启，计算 DMD loss
        │
        ├── DMD._compute_kl_grad(fake_score, estimated_x0, ...)
        │   ├── fake_score.forward(x_t, t, cond)      # 判别器评分
        │   └── kl_grad = fake_score - real_score      # KL 散度梯度
        │
        └── optimizer.step()
```

---

## 九、关键超参数速查

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `local_attn_size` | 滑动窗口（帧数），-1=全局 | 12 |
| `sink_size` | Frame Sink 帧数 | 3 |
| `num_frame_per_block` | 每块生成的帧数 | 3 |
| `denoising_step_list` | 每帧去噪 timestep 序列 | [1000,750,500,250] |
| `context_noise` | Context Pass 的噪声级别 | 0 |
| `num_output_frames` | 总生成帧数 | 120（8s@15fps） |
| `frame_seq_length` | 每帧的 token 数（硬编码） | 1560 |
| `num_transformer_blocks` | Wan 1.3B 的 DiT 层数（硬编码） | 30 |
