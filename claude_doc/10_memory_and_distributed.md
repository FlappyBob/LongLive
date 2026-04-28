# 显存管理·FSDP·梯度检查点·EMA

> 核心文件：[utils/memory.py](../utils/memory.py)，[utils/distributed.py](../utils/distributed.py)

---

## 一、推理显存预算

以 Wan 1.3B，120 帧生成，bfloat16 为例：

```
┌─────────────────────────────────────────────────────────────┐
│                    推理显存分布                               │
├──────────────────────────────┬──────────────────────────────┤
│  组件                         │  显存估算                    │
├──────────────────────────────┼──────────────────────────────┤
│  CausalWanModel 参数          │  ~2.5 GB                     │
│  UMT5-XXL 文本编码器          │  ~11 GB（low_memory=True时   │
│                              │  用 DynamicSwap 换出到 CPU）  │
│  Video VAE                   │  ~3 GB                       │
│  KV Cache（30块×2×15帧）      │  ~4.3 GB                     │
│  Cross-Attn Cache（30块）     │  ~0.4 GB                     │
│  中间激活（单帧去噪）           │  ~2 GB                       │
│  Output buffer（low_mem=CPU）│  在 CPU（不占 GPU）           │
├──────────────────────────────┼──────────────────────────────┤
│  合计（文本编码在 GPU 时）      │  ~23 GB                      │
│  合计（low_memory=True）      │  ~12 GB（文本编码器在 CPU）   │
└──────────────────────────────┴──────────────────────────────┘

H100 80GB：绰绰有余
A100 40GB：需要 low_memory=True
RTX 3090 24GB：可能 OOM（显存边界）
```

---

## 二、DynamicSwapInstaller（低显存文本编码器）

文件：[utils/memory.py:13-58](../utils/memory.py)

来源自 lllyasviel/FramePack，是一个 hack：

```python
class DynamicSwapInstaller:
    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        # 对 model 的每个子模块做 class monkey-patch
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, device=gpu)
```

原理：把每个模块的 `__getattr__` 重写，使得访问参数时**自动把它移到目标设备**：

```python
def hacked_get_attr(self, name):
    if name in self._parameters:
        p = self._parameters[name]
        return p.to(**kwargs)   # 参数还在 CPU，访问时临时移到 GPU
    ...
```

**效果**：
- 模型参数物理上存在 CPU
- 每次 forward 时，参数按需移到 GPU（自动）
- forward 完成后参数仍在 CPU（不改变 `.data` 位置）
- 节省约 11 GB 的 GPU 显存（代价：每次 forward 有 CPU→GPU 传输开销）

---

## 三、`move_model_to_device_with_memory_preservation`

```python
def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb):
    # 把模型移到 target_device，但保留 preserved_memory_gb 的可用空间
    # 如果剩余空间不够，把模型"换出"到 CPU
    available = get_cuda_free_memory_gb(target_device)
    if available < preserved_memory_gb:
        model.to('cpu')  # 腾出空间给其他模型
    else:
        model.to(target_device)
```

用于文本编码器在推理时的动态调度：

```python
# causal_inference.py:84-86
if low_memory:
    gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
    move_model_to_device_with_memory_preservation(
        self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
    )
```

---

## 四、KV Cache 显存精确计算

```python
# 参数（Wan 1.3B 默认）
num_blocks = 30
num_heads = 12
head_dim = 128
sink_size = 3
local_attn_size = 12
frame_seqlen = 1560

kv_cache_frames = sink_size + local_attn_size  # = 15 帧
kv_cache_size = kv_cache_frames * frame_seqlen  # = 23400 tokens

# 单块 K 或 V 的显存：
# [B=1, 23400, 12, 128] × 2 bytes (bf16)
single_kv = 1 * 23400 * 12 * 128 * 2 / (1024**3)  # ≈ 0.072 GB

# 所有 30 块的 K+V：
total_kv = single_kv * 2 * 30  # ≈ 4.3 GB

# Cross-Attn Cache（文本，固定 512 tokens）：
# [1, 512, 12, 128] × 2 bytes × 30 块 × K+V
cross_attn = 1 * 512 * 12 * 128 * 2 * 2 * 30 / (1024**3)  # ≈ 0.09 GB
```

---

## 五、FSDP 配置

文件：[utils/distributed.py:23-70](../utils/distributed.py)

### 5.1 分片策略

| 策略 | 含义 | 用途 |
|------|------|------|
| `FULL_SHARD` | 参数、梯度、优化器状态全部分片 | 最省显存，但通信开销大 |
| `HYBRID_SHARD` | 机器内 full shard，机器间 all-reduce | 多机训练的折中 |
| `NO_SHARD` | DDP，不分片 | 单机调试 |

LongLive 训练配置：`sharding_strategy: hybrid_full`（多机 8×8卡 部署）。

### 5.2 自动包装策略

```python
fsdp_wrap(generator, wrap_strategy="size", min_num_params=5e7)
# size 策略：参数量 > 5000万 的子模块单独分片
# 对 CausalWanAttentionBlock（每层约 1.3亿参数）自然分片
```

### 5.3 保存 FSDP 模型

```python
def fsdp_state_dict(model):
    # FSDP 下每个 rank 只有部分参数
    # FullStateDictConfig + rank0_only=True：只在 rank 0 收集完整 state dict
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True,  # 收集到 CPU（参数太大）
        rank0_only=True       # 只有 rank 0 有完整副本，其他 rank 返回 {}
    )
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, ...):
        checkpoint = model.state_dict()
    return checkpoint
```

---

## 六、梯度检查点（Gradient Checkpointing）

```python
# trainer/distillation.py（config.gradient_checkpointing=True 时）
if args.gradient_checkpointing:
    self.generator.enable_gradient_checkpointing()
    self.fake_score.enable_gradient_checkpointing()

# WanDiffusionWrapper.enable_gradient_checkpointing:
def enable_gradient_checkpointing(self):
    self.model.gradient_checkpointing = True
```

### 6.1 与 KV Cache 的兼容性问题

梯度检查点会**重新运行 forward**（recompute）以重新计算激活值，而不是保存它们。

这对 KV Cache 是个挑战：

```
正常 forward:
  block 0 forward → 写入 kv_cache[0]
  block 1 forward → 写入 kv_cache[1]
  ...
  backward: 直接用保存的激活值

梯度检查点 forward:
  block 0 forward（梯度阶段）:
    → 先 forward（不保存激活）
    → 遇到需要梯度的点：丢弃激活，标记重算
    → 后向时：重新 forward（recompute）
```

**如果 recompute 时 KV Cache 已经被更新，recompute 的结果会不同！**

LongLive 的解决方案（见 `causal_model.py`）：

```python
# 判断是否是 recompute pass
is_recompute = (current_end <= kv_cache["global_end_index"].item()) 
               and (current_start > 0)

# recompute 时：
# 1. 不更新 global/local 指针
# 2. 保护 sink 区域（不覆盖已写入的 sink K/V）
# 3. 用 temp_k/v 副本做 attention（不改原 cache）
```

这样 recompute 的 attention 结果与第一次 forward 一致（都用同一个 cache 快照），梯度正确。

---

## 七、EMA（指数移动平均）

文件：[utils/distributed.py](../utils/distributed.py)（`EMA_FSDP` 类）

```python
class EMA_FSDP:
    def __init__(self, model, decay=0.9999, start_step=200):
        self.model = model        # 原模型（训练中更新）
        self.ema_model = copy.deepcopy(model)  # EMA 副本
        self.decay = decay
        self.start_step = start_step
    
    def update(self, step):
        if step < self.start_step:
            return  # 前 200 步不更新 EMA（让模型先"预热"）
        
        # EMA 更新公式
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1-self.decay)
```

**EMA 的作用**：
- 平均掉训练过程中的噪声波动
- 最终推理质量通常优于最后一步的 checkpoint
- LongLive 默认 `use_ema=False`（推理时），但如果 `generator_ema` 存在于 checkpoint，可以用 `use_ema=True`

---

## 八、`log_gpu_memory` 调试工具

```python
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[GPU Memory] {tag}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
```

在代码多处可以看到被注释掉的 `log_gpu_memory(...)` 调用，这是调试显存的工具，正式运行时关闭。

---

## 九、显存优化策略总结

| 策略 | 节省量 | 代价 |
|------|--------|------|
| `low_memory=True`（文本编码器 DynamicSwap） | ~11 GB | 文本编码慢一些 |
| Output buffer 在 CPU | ~0.3 GB | GPU→CPU 拷贝 |
| 局部注意力（local_attn_size=12） | ~30 GB（vs 全局） | 轻微质量下降 |
| 梯度检查点 | ~50% 显存（激活值） | 约 30% 速度损失 |
| FSDP FULL_SHARD | 1/N 参数显存 | 通信开销 |
| LoRA 微调（阶段二） | ~90% 参数显存 | 需要阶段一基础 |
