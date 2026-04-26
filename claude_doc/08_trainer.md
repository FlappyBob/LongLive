# Trainer — 分布式训练调度器

**文件**：[trainer/distillation.py](../trainer/distillation.py)

---

## 一、定位

`Trainer` 是**整个训练系统的总调度器**，负责：
1. 分布式环境初始化（FSDP、rank、seed）
2. 模型初始化（Generator、Critic、TextEncoder、VAE）
3. LoRA 配置与加载
4. Optimizer、EMA、DataLoader 初始化
5. Checkpoint 保存/恢复
6. 训练主循环（generator_loss + critic_loss）
7. 可视化（定期生成样本视频）

---

## 二、初始化流程

```
Trainer.__init__(config)
│
├── Step 1: 分布式初始化
│   ├── launch_distributed_job()          # torchrun 入口
│   ├── dist.get_rank(), get_world_size()
│   └── set_seed(config.seed + rank)
│
├── Step 2: W&B / OneLogger 日志初始化
│
├── Step 3: 模型初始化
│   ├── 根据 distribution_loss 选择模型类:
│   │   ├── "dmd"        → DMD(config)
│   │   ├── "dmd_switch" → DMDSwitch(config)
│   │   └── "causvid"    → CausVid(config)
│   ├── LoRA 配置（如果 config.adapter 存在）
│   │   ├── 加载 base checkpoint
│   │   ├── _configure_lora_for_model(generator)  ← PEFT LoraConfig rank=256
│   │   ├── _configure_lora_for_model(fake_score)
│   │   └── 加载 LoRA checkpoint（如有）
│   ├── fsdp_wrap(generator)       # FSDP 分片
│   ├── fsdp_wrap(real_score)
│   ├── fsdp_wrap(fake_score)
│   └── fsdp_wrap(text_encoder)
│
├── Step 4: EMA 初始化（ema_weight=0.99）
│
├── Step 5: Optimizer 初始化
│   ├── generator_optimizer = AdamW(lr=1e-5, β=(0,0.999))
│   └── critic_optimizer    = AdamW(lr=2e-6, β=(0,0.999))
│
├── Step 6: DataLoader 初始化
│   ├── TextDataset(data_path)  ← 纯文本提示词文件
│   └── DistributedSampler
│
├── Step 7: Checkpoint 恢复（auto_resume）
│
└── Step 8: StreamingTrainingModel 初始化（如果 streaming_training=True）
```

---

## 三、训练主循环

```python
def train():
    while True:
        TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)  # 每 5 步更新一次 Generator

        if streaming_training:
            # 流式训练路径
            if not streaming_active:
                start_new_sequence()  # 获取新 batch，设置序列
            if not streaming_model.can_generate_more():
                start_new_sequence()  # 当前序列生成完了

            fwdbwd_one_step_streaming(train_generator=TRAIN_GENERATOR)

        else:
            # 标准训练路径（阶段一）
            batch = next(dataloader)
            fwdbwd_one_step(batch, train_generator=TRAIN_GENERATOR)

        optimizer.step()
        step += 1

        if step % log_iters == 0:
            save()
        if step % vis_interval == 0:
            _visualize()
        if step > max_iters:
            break
```

---

## 四、fwdbwd_one_step() — 标准训练一步

```
1. text_encoder(prompts) → conditional_dict
2. text_encoder([negative_prompt]) → unconditional_dict  (第一次后缓存)

if train_generator:
    3. model.generator_loss(...)
       → SelfForcingTrainingPipeline.inference_with_trajectory  (unroll)
       → compute_distribution_matching_loss (DMD grad)
    4. loss.backward()
    → 返回 generator_log_dict

else:
    5. model.critic_loss(...)
       → generator unroll (no_grad)
       → fake_score denoising loss
    6. loss.backward()
    → 返回 critic_log_dict
```

---

## 五、fwdbwd_one_step_streaming() — 流式训练一步

```
if train_generator:
    if current_seq_length == 0:
        generate_next_chunk(requires_grad=False)  # 先无梯度跑一个 warmup chunk
    generated_chunk, chunk_info = generate_next_chunk(requires_grad=True)
    generator_loss, _ = streaming_model.compute_generator_loss(chunk, chunk_info)
    loss.backward()

else:
    if current_seq_length == 0:
        generate_next_chunk(requires_grad=False)
    generated_chunk, chunk_info = generate_next_chunk(requires_grad=False)
    critic_loss, _ = streaming_model.compute_critic_loss(chunk, chunk_info)
    loss.backward()
```

**为什么第一个 chunk 不带梯度？**  
第一个 chunk 的历史上下文是空的（KV Cache 全 0），对应的损失不具有代表性，跳过可以稳定训练。

---

## 六、LoRA 配置

```python
def _configure_lora_for_model(transformer, model_name):
    # 找所有 CausalWanAttentionBlock 下的 Linear 层
    target_linear_modules = [所有 attn + ffn 的 Linear]

    peft_config = LoraConfig(
        r=256,           # LoRA rank
        lora_alpha=256,  # 缩放系数（通常 = rank）
        lora_dropout=0.0,
        target_modules=target_linear_modules
    )
    return peft.get_peft_model(transformer, peft_config)
```

LoRA 插入到 Generator 的所有 attention + FFN linear 层，冻结基础权重，只训练 A、B 矩阵。
参数量：rank=256 的 LoRA 对 1.3B 模型大约增加 ~100M 可训练参数。

---

## 七、Checkpoint 管理

```python
def save():
    # LoRA 模式: 只保存 LoRA 权重（几十 MB）
    state_dict = {
        "generator_lora": get_peft_model_state_dict(generator),
        "critic_lora": get_peft_model_state_dict(fake_score),
        "step": self.step
    }

    # 非 LoRA 模式: 保存完整权重
    state_dict = {
        "generator": FSDP.full_state_dict(generator),
        "critic": FSDP.full_state_dict(fake_score),
        "generator_ema": ema.state_dict(),
        "generator_optimizer": FSDP.optim_state_dict(...),
        "critic_optimizer": ...,
        "step": self.step
    }

    torch.save(state_dict, f"checkpoint_model_{step:06d}/model.pt")
    cleanup_old_checkpoints(max_checkpoints=3)  # 只保留最新 3 个
```

---

## 八、关键配置参数（longlive_train_long.yaml）

```yaml
distribution_loss: dmd_switch     # 使用 DMDSwitch（支持 prompt 切换训练）
real_name: Wan2.1-T2V-14B         # 教师模型（14B）
fake_name: Wan2.1-T2V-1.3B        # Critic 模型（1.3B）
model_kwargs:
  local_attn_size: 12             # 12 帧滑动窗口
  sink_size: 3                    # 3 帧 Frame Sink
  timestep_shift: 5.0

streaming_training: true
streaming_chunk_size: 21          # 每 chunk 21 帧
streaming_max_length: 240         # 最大序列 240 帧

dfake_gen_update_ratio: 5         # Generator:Critic = 1:5 更新比率
lr: 1e-5                          # Generator 学习率
lr_critic: 2e-6                   # Critic 学习率（更小）
adapter:
  rank: 256                       # LoRA rank
max_iters: 3000                   # 训练步数
```
