# 内存管理与分布式训练

**文件**：[utils/memory.py](../utils/memory.py)、[utils/distributed.py](../utils/distributed.py)、[wan/distributed/fsdp.py](../wan/distributed/fsdp.py)

---

## 一、内存管理策略总览

```
显存压力来源:
1. 三个大模型（Generator 1.3B + Critic 1.3B + Teacher 14B）
2. KV Cache（30块 × [B, cache_tokens, 12, 128]）
3. 梯度 + 优化器状态
4. 激活值（gradient checkpointing 用时间换空间）

缓解策略:
┌────────────────────────────────────────────────────────────────┐
│ FSDP              所有模型分片到多 GPU，每 GPU 只存 1/N 参数    │
│ LoRA              只训练低秩矩阵，冻结基础模型                  │
│ gradient_checkpointing  不存激活值，反向传播时重新计算          │
│ local_attn        KV Cache 大小从 O(T) 降为 O(window)          │
│ frame_sink        维持语义一致性的同时不增加注意力计算量         │
│ DynamicSwap       CPU offload：不用时移到 CPU，需要时再移回 GPU │
│ low_memory mode   VAE decode 时 text encoder offload 到 CPU    │
│ chunk decode      超长视频分块解码，防止 VAE OOM               │
└────────────────────────────────────────────────────────────────┘
```

---

## 二、FSDP 配置

```python
fsdp_wrap(module, sharding_strategy, mixed_precision, wrap_strategy)
```

**Sharding Strategy**：

| 策略 | 含义 | 适用场景 |
|------|------|---------|
| `hybrid_full` | 节点内全分片，节点间复制 | 多节点训练（默认）|
| `full` | 所有 GPU 全分片 | 单节点多 GPU |
| `no_shard` | 不分片 | 小模型或 debug |

**Mixed Precision**：使用 `bfloat16`，计算高效且数值稳定。

**Wrap Strategy**（决定哪些 submodule 单独分片）：

```python
"size" → 按参数量自动分片，超过阈值的 module 独立分片
"block" → 按 transformer block 分片（每个 block 是一个分片单元）
```

---

## 三、DynamicSwapInstaller（按需 CPU offload）

```python
DynamicSwapInstaller.install_dynamic_swap(model, device='cpu')
```

这个工具将模型的 forward 替换为"按需加载"版本：
- 调用前：从 CPU 加载权重到 GPU
- 调用后：卸载回 CPU

适用于推理时 GPU 内存不足（如同时需要 Generator + Teacher + VAE）的场景。

---

## 四、KV Cache 内存估算

```
1 帧 KV Cache（单个 Block）：
  k: [1, 1560, 12, 128] = 1560×12×128 = 2,396,160 float16 = 4.57 MB

30 个 Block 的 KV Cache（local_attn=12 帧）：
  total_cache_frames = 12
  per_block = 12 × 1560 × 12 × 128 × 2 bytes = ~54.7 MB (k+v)
  30 blocks = 30 × 54.7 MB ≈ 1.64 GB

全局注意力（21 帧）：
  30 × (21 × 1560 × 12 × 128 × 2) ≈ 2.87 GB

Cross-Attn Cache（文本，所有 Block）：
  30 × 512 × 12 × 128 × 2 × 2 bytes ≈ 94 MB（可忽略）
```

这就是为什么 `local_attn_size=12` 比全局注意力（-1）节省约 43% KV Cache 显存。

---

## 五、梯度检查点（Gradient Checkpointing）

```python
self.generator.enable_gradient_checkpointing()
# → CausalWanModel._set_gradient_checkpointing(True)
```

启用后，`torch.utils.checkpoint.checkpoint` 包裹每个 Block 的 forward：
- 前向传播：不保存中间激活值（节省 ~30% 激活显存）
- 反向传播：重新计算激活值（增加约 33% 计算量）

**与 KV Cache 的兼容性问题**：
gradient checkpointing 重新计算 forward 时，`current_end <= global_end_index` 触发 `is_recompute=True`，此时 sink 区域受保护，不会被重计算覆盖。这是专门为解决这个问题设计的。

---

## 六、log_gpu_memory 调试工具

```python
log_gpu_memory(f"After text encoding", device=device, rank=rank)
# 输出: [rank 0] After text encoding: 35.2GB / 80.0GB
```

只在 `LOG_GPU_MEMORY=True`（`DEBUG_OPTION` 环境变量）时启用，方便定位显存泄漏和峰值。

---

## 七、EMA（指数移动平均）

```python
EMA_FSDP(model.generator, decay=0.99)
```

EMA 在 step >= `ema_start_step`（默认 200）后启用：
- 每步训练后更新：`ema_params = 0.99 × ema_params + 0.01 × current_params`
- 推理时可选用 EMA 权重（`use_ema=true`）获得更稳定的输出
- LoRA 模式下 EMA 禁用（LoRA 本身已足够轻量）
