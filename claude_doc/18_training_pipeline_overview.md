# 训练流程总览：从 Wan2.1-1.3B 到推理 ckpt

> 本文配套 deepwiki 总览图，自顶向下讲清"推理时加载的那个 ckpt 是怎么训出来的"。
>
> 关键文件：
> [train.py](../train.py) · [trainer/distillation.py](../trainer/distillation.py)
> [model/dmd.py](../model/dmd.py) · [model/dmd_switch.py](../model/dmd_switch.py) · [model/streaming_training.py](../model/streaming_training.py)
> [pipeline/self_forcing_training.py](../pipeline/self_forcing_training.py) · [pipeline/streaming_switch_training.py](../pipeline/streaming_switch_training.py)
> [configs/longlive_train_init.yaml](../configs/longlive_train_init.yaml) · [configs/longlive_train_long.yaml](../configs/longlive_train_long.yaml)

---

## 一、ckpt 谱系（一图流）

```
                         Wan2.1-T2V-1.3B (HF 预训练，Non-causal)
                                      │
                                      │  ① ODE distill / causal 化
                                      │     （仓库不含此步骤，外部产物）
                                      ▼
                       checkpoints/ode_init.pt
                                      │
              ┌───────────────────────┴────────────────────────┐
              │                                                │
              │  ② Phase-1: Self-Forcing Init (DMD, 全参微调)  │
              │     train_init.sh + longlive_train_init.yaml   │
              │     21 frames · lr=2e-6 · 700 iters · EMA=0.99 │
              │                                                │
              │   teachers/critics:                             │
              │     real_score  = Wan2.1-T2V-14B  (frozen)      │
              │     fake_score  = Wan2.1-T2V-1.3B (trainable)   │
              ▼                                                │
       checkpoints/longlive_init.pt        ◀── 阶段 1 输出 / 阶段 2 输入
              │                                                │
              │  ③ Phase-2: Streaming Long Tuning              │
              │     train_long.sh + longlive_train_long.yaml   │
              │     240 frames · lr=1e-5 · 3000 iters          │
              │     LoRA r=256/α=256 · No EMA · Streaming      │
              │     distribution_loss = dmd_switch (含 prompt 切换)
              ▼
            longlive_lora.safetensors  (≈数十 MB，仅 LoRA adapter)
              │
              ▼
   推理时 = longlive_init.pt (full)  ⊕  longlive_lora.safetensors (PEFT 注入)
```

入口脚本：

| 阶段 | 脚本 | Config | 起点 ckpt |
|---|---|---|---|
| 1 | [train_init.sh](../train_init.sh) | [longlive_train_init.yaml](../configs/longlive_train_init.yaml) | `checkpoints/ode_init.pt` |
| 2 | [train_long.sh](../train_long.sh) | [longlive_train_long.yaml](../configs/longlive_train_long.yaml) | `checkpoints/longlive_init.pt` |

两个脚本都跑同一份 [train.py](../train.py)，差别全在 yaml。

---

## 二、共用骨架：三模型 + Score Distillation

[train.py:40-42](../train.py#L40-L42) 唯一支持 `trainer=score_distillation`：

```
ScoreDistillationTrainer.__init__         ← trainer/distillation.py:45
   ├─ launch_distributed_job() 8×GPU torchrun
   ├─ if distribution_loss == "dmd"       ← model/dmd.py:14
   │      self.model = DMD(...)
   ├─ elif distribution_loss == "dmd_switch"  ← model/dmd_switch.py:18
   │      self.model = DMDSwitch(...)
   ├─ (Phase-2) _configure_lora_for_model(generator) / (fake_score) ← :208-212
   ├─ FSDP 包 generator + fake_score (sharding=hybrid_full)
   ├─ AdamW 优化器（generator / critic 各一）
   ├─ EMA_FSDP（仅 Phase-1，ema_start_step=200, decay=0.99）
   ├─ TextDataset / TwoTextDataset                ← :391-394
   └─ (Phase-2) StreamingTrainingModel 包一层    ← :570-576
```

[model/base.py:29-46](../model/base.py#L29-L46) 起三个 WanDiffusionWrapper：

| 角色 | 实例 | 状态 | 作用 |
|---|---|---|---|
| `generator` | Wan2.1-1.3B, `is_causal=True` | 训 | 学生（推理时用的就是它） |
| `real_score` | Wan2.1-14B, `is_causal=False` | 冻结 | Teacher，提供 real distribution score |
| `fake_score` | Wan2.1-1.3B, `is_causal=False` | 训 | Critic，逼近生成器分布 |

**Score distillation 节奏**（[distillation.py:1177](../trainer/distillation.py#L1177)）：

```
TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)   # 5
                  → 每 5 步：1 次 generator + 1 次 critic
                  → 其余 4 步：只更新 critic
```

---

## 三、DMD 损失（两阶段共用）

[model/dmd.py:60-200](../model/dmd.py#L60-L200) 完整实现 DMD2 论文 eq.7：

```
1. 采样时间步 t（带 timestep_shift=5.0 偏向高噪），加噪到 noisy_x
2. pred_real = real_score(noisy_x, t)  + CFG=3 的 cfg
3. pred_fake = fake_score(noisy_x, t)
4. grad      = pred_fake - pred_real
5. grad     /= |x - pred_real|.mean()           （eq.8 归一化）
6. dmd_loss  = 0.5 * MSE(x, (x - grad).detach())  （把 grad 挂到 x 反传）
```

`gradient_mask` 决定哪些帧计入 loss——首 chunk（含 image latent）或 overlap 帧屏蔽掉。

**Critic loss**（[dmd.py:272-392](../model/dmd.py#L272-L392)）：对 generator 输出加噪后，让 fake_score 做 flow / x0 去噪 loss，使其追踪生成器分布。

---

## 四、Phase-1：Self-Forcing 初始化

**目的**：把非 causal 的 Wan1.3B 通过 DMD 蒸成 4 步 causal 生成器，建立"用自己的输出当下一步条件"的能力。

**配置要点**（[longlive_train_init.yaml](../configs/longlive_train_init.yaml)）：

```yaml
generator_ckpt: checkpoints/ode_init.pt        # 起点
real_name: Wan2.1-T2V-14B                      # 14B teacher
fake_name: Wan2.1-T2V-1.3B                     # 1.3B critic
denoising_step_list: [1000, 750, 500, 250]     # 4 步蒸馏
guidance_scale: 3.0                            # CFG on real_score
distribution_loss: dmd
num_training_frames: 21                        # 固定 21 帧
slice_last_frames: 21
lr: 2.0e-6  /  lr_critic: 4.0e-7
max_iters: 700
ema_weight: 0.99   ema_start_step: 200
dfake_gen_update_ratio: 5
model_kwargs: {timestep_shift: 5.0, local_attn_size: 12, sink_size: 3}
```

**一步训练**（非 streaming 分支 [distillation.py:1259-1298](../trainer/distillation.py#L1259-L1298)）：

```
batch = next(dataloader)       # text-only prompt
fwdbwd_one_step(batch, train_generator)
    └─ DMD.generator_loss / DMD.critic_loss
            └─ _run_generator                      ← model/base.py:106
                 └─ SelfForcingTrainingPipeline
                       .inference_with_trajectory  ← pipeline/self_forcing_training.py:296
                       (从纯噪声开始 4 步去噪 21 帧)
            └─ compute_distribution_matching_loss  （前述 DMD 公式）
backward → clip_grad → optimizer.step → EMA.update
```

关键："训练时也走推理同款"——`inference_with_trajectory` 内部维护 KV cache、按 `denoising_step_list` 多步去噪，每个 block 只在 `exit_flags` 随机选中的一个时间步上开 grad（其余 `torch.no_grad`），完美对齐 train/inference 分布。

**输出**：[distillation.py:758-794](../trainer/distillation.py#L758-L794) 存 full state dict：

```python
{
    "generator":           full FSDP state_dict,
    "critic":              full FSDP state_dict (fake_score),
    "generator_ema":       EMA params,
    "generator_optimizer": ...,
    "critic_optimizer":    ...,
    "step": 700,
}
```

最终 `checkpoint_model_000700/model.pt` → 重命名为 `longlive_init.pt`。

---

## 五、Phase-2：Streaming Long Tuning

**目的**：让 21 帧的 causal 模型扩展到 240 帧长视频，且支持中途切 prompt；用 LoRA 做轻量增量微调，避免破坏 Phase-1 已学到的能力。

**配置要点**（[longlive_train_long.yaml](../configs/longlive_train_long.yaml)）：

```yaml
generator_ckpt: checkpoints/longlive_init.pt   # 起点 = Phase-1 输出
distribution_loss: dmd_switch
streaming_training: true
streaming_chunk_size: 21    streaming_max_length: 240
streaming_min_new_frame: 18 train_first_chunk: true

switch_mode: random_choice
switch_choices: [21, 39, 57, 75, 93, 111, 129, 147, 165, 183, 201]
global_sink: false

lr: 1.0e-5  / lr_critic: 2.0e-6   # 5× higher
max_iters: 3000

adapter:
  type: "lora"
  rank: 256   alpha: 256   dropout: 0.0   dtype: "bfloat16"
  apply_to_critic: true
```

**模型差异**：

- `DMDSwitch` ([model/dmd_switch.py:18-32](../model/dmd_switch.py#L18-L32)) 仅覆盖 inference_pipeline → `StreamingSwitchTrainingPipeline`。
- LoRA 在 FSDP 包之前注入 ([distillation.py:208-212](../trainer/distillation.py#L208-L212))，target = 注意力 q/k/v/o；rank=alpha=256 → scale=1，效果接近全参。
- LoRA 模式 EMA 关闭（[:1310-1312](../trainer/distillation.py#L1310-L1312)）。
- `StreamingTrainingModel` 把 DMDSwitch 包一层，提供 chunk-by-chunk 状态机（[model/streaming_training.py:21](../model/streaming_training.py#L21)）。

**一步训练 = 一个 chunk** （[distillation.py:1056-1170](../trainer/distillation.py#L1056-L1170) `fwdbwd_one_step_streaming`）：

```
if not streaming_active:
    start_new_sequence()                              ← :948
        ├─ batch=next(TwoTextDataset)                 # (prompt, switch_prompt)
        ├─ switch_frame_index = _get_switch_frame_index()  ← :690
        │     从 switch_choices 里 dist.broadcast 一致地随机选
        └─ streaming_model.setup_sequence(...)        ← model/streaming_training.py:282
              └─ reset KV cache + 保存 cond / switch_cond

generated_chunk, info = streaming_model.generate_next_chunk(requires_grad=True)
                                                       ← model/streaming_training.py:407
   ├─ 随机决定 new_frames ∈ range(18, max, 3)
   ├─ overlap = 21 - new_frames，复用上次 previous_frames
   ├─ 首帧 VAE decode→encode 模拟推理时 image-latent ([:79-119])
   ├─ 若 chunk 跨 switch_frame_index：
   │     StreamingSwitchTrainingPipeline._recache_after_switch
   │            ← pipeline/streaming_switch_training.py:244
   │     · 清 cross-attn cache
   │     · global_sink=false → 同时清 self-attn KV
   │     · 用最近 21 帧在新 prompt 下重写 cache
   ├─ generate_chunk_with_cache：4 步去噪 + 跨 chunk KV 持久
   └─ gradient_mask：仅新帧 True

generator_loss = DMD.compute_distribution_matching_loss(chunk, mask=gradient_mask)
backward → 累 grad → optimizer.step on LoRA params only
```

**关键机制**：

| 机制 | 作用 | 代码位置 |
|---|---|---|
| KV cache 跨 chunk 持久 | 模拟推理时长视频的状态延续 | `kv_cache_size=(12+21)*1560` [pipeline/streaming_training.py:50](../pipeline/streaming_training.py#L50) |
| Overlap + 首帧 re-encode | overlap 帧不算梯度，但提供视觉上下文 | [streaming_training.py:497-524](../model/streaming_training.py#L497-L524) |
| `train_first_chunk=true` | 首 chunk 也参与 loss | [distillation.py:1088-1098](../trainer/distillation.py#L1088-L1098) |
| Critic detach 防泄漏 | critic 前 `chunk.detach()` + `_clear_cache_gradients()` | [streaming_training.py:601-626, 654-658](../model/streaming_training.py#L601-L626) |
| `min_new_frame=18` 终止 | `current_length + 18 > 240` 时序列结束，重启 | [:397-406](../model/streaming_training.py#L397-L406) |

**输出**：[distillation.py:746-756](../trainer/distillation.py#L746-L756) 走 LoRA 分支，仅 dump：

```python
{
    "generator_lora": LoRA adapter state_dict,
    "critic_lora":    LoRA adapter state_dict,
    "step": 3000,
}
```

体积从全参的 ~2.5 GB 降到 ~数十 MB。

---

## 六、共用基础设施

| 组件 | 配置 | 说明 |
|---|---|---|
| **FSDP** | `sharding_strategy: hybrid_full` | 节点内 FULL_SHARD + 节点间 NO_SHARD，bf16 mixed precision |
| **EMA** | decay=0.99，仅 Phase-1（Phase-2 LoRA 模式禁用） | [distillation.py:1304-1312](../trainer/distillation.py#L1304-L1312) |
| **梯度累积** | `batch_size=1, accumulation_steps=1, total_batch_size=64` | 即 64 卡或 8 卡×8 累积 |
| **梯度裁剪** | `clip_grad_norm_` | generator/critic 分别 |
| **Auto resume** | `find_latest_checkpoint(logdir)` | [distillation.py:605](../trainer/distillation.py#L605) |
| **数据集** | TextDataset / TwoTextDataset (text only) | 完全无视频数据，靠 self-forcing 生成 |

---

## 七、推理端如何 load

[inference.py](../inference.py) 路径：

1. 起 `WanDiffusionWrapper(is_causal=True)` 1.3B causal 模型；
2. load `longlive_init.pt["generator"]`（或仓库直接发布的合并 base ckpt）→ 灌进 generator；
3. 通过 PEFT 注入 `longlive_lora`（与训练时同 target_modules）；
4. 起 `CausalInferencePipeline` / `InteractiveCausalInferencePipeline`；
5. 走 `denoising_step_list=[1000,750,500,250]` 4 步生成。

`real_score`、`fake_score`、`generator_ema`、optimizer state 在推理全部不需要。

---

## 八、一句话总结

> **推理 ckpt = (Wan1.3B → ode_init → Phase-1 全参 DMD self-forcing 700 步 ⇒ longlive_init.pt) ⊕ (Phase-2 DMDSwitch streaming + LoRA r256 3000 步 ⇒ adapter)**。
> 两阶段同用 DMD 框架（学生 vs 14B teacher 与自家 1.3B critic），区别仅在于：序列长度（21→240）、是否 streaming chunk 化、是否含 prompt switching、是否走 LoRA、是否开 EMA。
