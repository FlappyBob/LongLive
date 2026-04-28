# Trainer：训练主循环·LoRA 配置·Checkpoint 管理

> 核心文件：[trainer/distillation.py](../trainer/distillation.py)

---

## 一、Trainer 类层级

```
Trainer（基类）
    └── ScoreDistillationTrainer（实际使用）
            ├── 初始化模型（DMD / DMDSwitch）
            ├── 配置 FSDP 分布式训练
            ├── 配置 LoRA（阶段二）
            ├── 配置优化器 & 学习率调度
            └── train() 主循环
```

---

## 二、`Trainer.__init__` 初始化顺序

```python
def __init__(self, config):
    # 1. 初始化分布式环境
    launch_distributed_job()          # 设置 rank, world_size, device
    
    # 2. 初始化日志（wandb / one_logger）
    wandb.init(...)
    
    # 3. 加载基础模型权重（Wan 预训练）
    generator_ckpt = torch.load(config.generator_ckpt)
    generator.load_state_dict(generator_ckpt)
    
    # 4. FSDP 包装（多卡分布式）
    self.generator = fsdp_wrap(generator)
    self.fake_score = fsdp_wrap(fake_score)
    
    # 5. 初始化推理 Pipeline
    self.inference_pipeline = StreamingTrainingPipeline(
        generator=self.generator, ...
    )
    
    # 6. 配置 LoRA（如果是阶段二）
    if config.use_lora:
        configure_lora_for_model(generator.model, rank=256, alpha=256)
    
    # 7. 优化器
    self.optimizer = torch.optim.AdamW(
        trainable_params, lr=config.lr, weight_decay=0.01
    )
    
    # 8. 学习率调度
    self.scheduler = get_cosine_schedule_with_warmup(
        self.optimizer, num_warmup_steps=config.warmup_steps
    )
    
    # 9. 自动 Resume（如果 logdir 有 checkpoint）
    if config.auto_resume:
        self._maybe_load_checkpoint()
```

---

## 三、训练主循环 `train()`

```python
def train(self):
    for self.step in range(start_step, total_steps):
        
        # Step 1: 读取一个 batch（text prompt）
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        
        # Step 2: 文本编码
        conditional_dict = text_encoder(text_prompts)
        
        # Step 3: 采样随机噪声（latent 空间）
        noise = torch.randn([B, F, 16, 60, 104], device=device)
        
        # Step 4: 初始化 KV Cache（每个 step 重新初始化）
        pipeline.initialize_kv_cache(batch_size, ...)
        
        # Step 5: Warmup chunk（no_grad）
        output_warmup = pipeline.generate_chunk_with_cache(
            noise[:, :warmup_frames], cond, requires_grad=False
        )
        
        # Step 6: Training chunk（with grad）
        output_train = pipeline.generate_chunk_with_cache(
            noise[:, warmup_frames:], cond,
            current_start_frame=warmup_frames, requires_grad=True
        )
        
        # Step 7: 计算损失
        loss, log_dict = model.compute_loss(output_train, cond, ...)
        
        # Step 8: 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        
        # Step 9: EMA 更新（可选）
        if use_ema:
            ema.update()
        
        # Step 10: 日志 & 可视化
        if step % log_every == 0:
            wandb.log(log_dict)
        if step % vis_every == 0:
            visualize(output_train)
        
        # Step 11: Checkpoint 保存
        if step % save_every == 0:
            self._save_checkpoint()
```

---

## 四、LoRA 配置

文件：[utils/lora_utils.py](../utils/lora_utils.py)

```python
def configure_lora_for_model(model, model_name, lora_config, is_main_process=True):
    from peft import LoraConfig, get_peft_model
    
    peft_config = LoraConfig(
        r=lora_config.rank,          # 256（接近全参数微调）
        lora_alpha=lora_config.alpha, # 256（缩放比 = alpha/r = 1.0）
        lora_dropout=lora_config.dropout,  # 0.0
        target_modules=["q", "k", "v", "o"],  # 注意力的 QKV + 输出层
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    return model
```

**为什么 rank=256 这么大？**

标准 LoRA 的 rank 通常是 4~64。LongLive 用 256 是因为：
- 阶段一训练了完整参数（2.5GB）
- 阶段二需要学习的"长视频一致性"是个复杂的能力，小 rank 学不了
- `alpha=rank=256` 使缩放比 = 1，实际效果接近全参数微调

---

## 五、Checkpoint 管理

### 5.1 保存格式

```python
# trainer/distillation.py 的 _save_checkpoint
checkpoint = {
    "step": self.step,
    "generator": fsdp_state_dict(self.generator),     # 完整参数
    "generator_ema": fsdp_state_dict(self.ema),       # EMA 参数（可选）
    "fake_score": fsdp_state_dict(self.fake_score),
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
}

# LoRA 单独保存（体积小，方便分享）
lora_state_dict = get_peft_model_state_dict(generator.model)
torch.save({"generator_lora": lora_state_dict}, lora_ckpt_path)

torch.save(checkpoint, checkpoint_path)
```

### 5.2 自动 Resume

```python
def _maybe_load_checkpoint(self):
    # 扫描 logdir 下的所有 checkpoint，加载最新的
    checkpoints = sorted(glob(f"{self.output_path}/checkpoint_*.pt"))
    if len(checkpoints) > 0:
        latest = checkpoints[-1]
        checkpoint = torch.load(latest)
        generator.load_state_dict(checkpoint["generator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
```

---

## 六、EMA（Exponential Moving Average）

```python
# 训练时：
ema_decay = 0.9999
ema.update():
    for ema_p, p in zip(ema_params, model_params):
        ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data

# 推理时（use_ema=True）：
raw_gen_state_dict = state_dict["generator_ema"]  # 用 EMA 权重推理
```

EMA 权重比普通权重更"平滑"，通常推理质量略好（去除训练过程中的噪声波动）。

---

## 七、分布式训练配置（FSDP）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# fsdp_wrap 函数（utils/distributed.py）
def fsdp_wrap(model, ...):
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 全分片
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls={CausalWanAttentionBlock}
        ),
        device_id=torch.cuda.current_device()
    )
```

FSDP 把模型参数切分到所有 GPU，每个 GPU 只保存 1/N 的参数。前向时 all-gather 收集完整参数，后向时 reduce-scatter 聚合梯度。

---

## 八、`launch_distributed_job`

```python
# utils/distributed.py
def launch_distributed_job():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    # 设置 NCCL 超时（大模型通信可能很慢）
    os.environ["NCCL_TIMEOUT"] = "1800"
```

训练用 `torchrun --nproc_per_node=8 train.py` 启动 8 进程，每进程占一张 GPU。

---

## 九、One Logger 集成

```python
# trainer/distillation.py:88-100
from one_logger_utils import OneLoggerUtils

# One Logger 是 NVIDIA 内部的实验追踪工具（类似 wandb）
# 外部复现时用 --no-one-logger 跳过
if self.use_one_logger and not self.disable_wandb:
    OneLoggerUtils.on_train_start(config, app_tag, ...)
```

本 repo 提供了 `one_logger_utils.py` 的空 no-op 实现，外部用户不受影响。
