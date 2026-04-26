# CausalWanModel — 因果扩散 Transformer 主干

**文件**：[wan/modules/causal_model.py](../wan/modules/causal_model.py)

---

## 一、定位

`CausalWanModel` 是整个系统的**神经网络核心**。它是在 Wan2.1-T2V-1.3B 原版 `WanModel` 基础上改造的因果（autoregressive）版本，能够：

1. 逐帧生成（每次处理 1 帧或若干帧），无需重复计算历史帧
2. 通过 KV Cache 存储历史 K/V，实现 O(n) 内存增量
3. 通过 Frame Sink + 滑动窗口控制注意力范围，防止内存爆炸

原版 `WanModel` 是一次性处理整段视频的非因果模型；`CausalWanModel` 改掉了 `WanSelfAttention` → `CausalWanSelfAttention`，其余结构（交叉注意力、FFN、head）保持不变。

---

## 二、类结构总览

```
CausalWanModel (ModelMixin, ConfigMixin)
├── patch_embedding     Conv3d(in_dim=16, dim=2048, kernel=(1,2,2))
├── text_embedding      Linear(4096→2048) → GELU → Linear(2048→2048)
├── time_embedding      Linear(256→2048) → SiLU → Linear(2048→2048)
├── time_projection     SiLU → Linear(2048→2048*6)
├── blocks [×30]        CausalWanAttentionBlock
│   ├── norm1           WanLayerNorm
│   ├── self_attn       CausalWanSelfAttention  ← 核心改造点
│   ├── norm3           WanLayerNorm (可选)
│   ├── cross_attn      WanCrossAttention (文本交叉注意力)
│   ├── norm2           WanLayerNorm
│   ├── ffn             Linear → GELU → Linear
│   └── modulation      Parameter [1, 6, 2048]  (AdaLN-Zero 条件调制)
├── head                CausalHead (LayerNorm + Linear → unpatchify)
└── freqs               RoPE 频率参数 [1024, head_dim/2]
```

---

## 三、核心组件详解

### 3.1 CausalWanSelfAttention

```
输入:  x [B, S, dim]  (S = current_num_frames × frame_seqlen)
输出:  x [B, S, dim]  + cache_update_info
```

**关键逻辑**（有 kv_cache 时）：

```
1. qkv_fn(x) → q, k, v
2. causal_rope_apply(q/k, start_frame=current_start_frame)   # 带偏移的 RoPE
3. 判断 cache 状态:
   ├── direct_insert: cache 未满，直接在 local_end_index 位置写入新 K/V
   └── roll_and_insert: cache 已满
       ├── 计算 num_evicted_tokens (要丢弃的最旧 token 数)
       ├── 将 sink_tokens 之后的内容向左滚动 (roll)
       └── 在腾出的位置写入新 K/V
4. attention(q, k_cat, v_cat)    # k_cat = sink_tokens + local_window
5. 返回 output + cache_update_info（延迟更新，由 _apply_cache_updates 统一写入）
```

**为什么延迟更新 cache？**
梯度检查点（gradient checkpointing）会重新计算 forward pass，如果在 forward 中直接写 cache，重计算时会覆盖已写入的正确数据。因此先收集 `cache_update_info`，所有 block 都跑完后再统一写入。

### 3.2 Frame Sink 机制

```
KV Cache 布局:
┌──────────────────────────────────────────────────────┐
│  Sink Frames (永不驱逐)    │  Local Window (滑动)    │
│  [0 : sink_size×1560]     │  [...: local_end_index] │
└──────────────────────────────────────────────────────┘

当 cache 满时:
old: [SINK | old1 | old2 | old3 | old4]   ← old1 被驱逐
new: [SINK | old2 | old3 | old4 | new  ]  ← left-shift + insert
```

`sink_size=3` 时，前 3 帧（4680 个 token）永远不动，保障视频长程一致性。

### 3.3 BlockMask 的三种模式

| 方法 | 场景 | 掩码逻辑 |
|------|------|----------|
| `_prepare_blockwise_causal_attn_mask` | 训练/标准推理 | 每帧只看过去帧，同 chunk 内全局可见 |
| `_prepare_teacher_forcing_mask` | Teacher Forcing 训练 | clean 帧按因果，noisy 帧可见对应 clean 帧 + 自身 |
| `_prepare_blockwise_causal_attn_mask_i2v` | 图生视频 | 第一帧独立，后续帧按 chunk 因果 |

FlexAttention 的 `create_block_mask` 将掩码函数 **编译为 CUDA kernel**，避免了实例化完整的注意力矩阵。

### 3.4 forward 分发

```python
def forward(self, *args, **kwargs):
    if kwargs.get('kv_cache') is not None:
        return self._forward_inference(...)   # 推理路径：逐帧 + KV Cache
    else:
        return self._forward_train(...)        # 训练路径：全序列 FlexAttention
```

`_forward_train` 目前有 `raise NotImplementedError()`（pass 之后），实际训练通过 `_forward_inference` + 梯度启用实现。

---

## 四、forward_inference 流程

```
输入: x [B, F, C, H/8, W/8] latent 帧（1~few 帧）

1. patch_embedding(x)         # Conv3d: [B, dim, F, H', W'] → flatten → [B, S, dim]
2. time_embedding(t)          # sinusoidal + MLP → e [B, F, dim]
3. time_projection(e)         # → e0 [B, F, 6, dim]  (AdaLN-Zero 6个调制参数)
4. text_embedding(context)    # [B, 512, 4096] → [B, 512, dim]

[循环 30 个 Block]:
   block(x, e=e0, context=context, kv_cache=kv_cache[i], current_start=...)
   → 返回 (x, cache_update_info)，收集 cache_update_infos

5. _apply_cache_updates(kv_cache, cache_update_infos)  # 统一写 cache

6. head(x, e)                 # LayerNorm + Linear → patch output
7. unpatchify(x, grid_sizes)  # einsum 重排 → [C, F, H, W] latent
```

---

## 五、RoPE 细节

`causal_rope_apply` 与原版 `rope_apply` 的关键区别：

```python
# 原版: 从 frame 0 开始算频率
freqs_i = freqs[0:F]

# causal 版本: 从 start_frame 开始，保证位置编码连续
freqs_i = freqs[start_frame : start_frame + F]
```

这确保了在生成第 20 帧时，它的 RoPE 位置是 20，而不是重置为 0，从而与 KV Cache 中历史帧的位置编码正确对齐。

---

## 六、关键超参数

| 参数 | 1.3B 值 | 含义 |
|------|---------|------|
| `dim` | 2048 | 隐藏层维度 |
| `num_layers` | 30 | Transformer 层数 |
| `num_heads` | 12 | 注意力头数（注意：不是 16！） |
| `head_dim` | 2048/12 ≈ 170 → 实为 128（dim=1536 for 1.3B） | 每头维度 |
| `ffn_dim` | 8192 | FFN 中间层维度 |
| `patch_size` | (1, 2, 2) | 时间不压缩，空间 2×2 压缩 |
| `text_len` | 512 | 文本 token 最大长度 |

> **注意**：Wan 1.3B 实际 dim=1536，num_heads=12，head_dim=128。代码中写的 `dim=2048` 是默认参数，从 pretrained 权重加载后会被覆盖。FlexAttention 的 `max-autotune-no-cudagraphs` 模式正是为了解决 12 头这个非标准配置的 issue。
