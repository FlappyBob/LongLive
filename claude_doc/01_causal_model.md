# CausalWanModel：因果注意力 DiT 骨干

> 核心文件：[wan/modules/causal_model.py](../wan/modules/causal_model.py)

---

## 一、整体架构

`CausalWanModel` 是 LongLive 对 Wan 的核心改造，把双向 Diffusion Transformer 变成支持逐帧流式生成的因果模型。

```
输入: noisy video latent [B, C=16, F, H, W]
           │
           ▼
     patch_embedding        Conv3d(16→2048, kernel=(1,2,2))
           │
           ▼  [B, F×30×52, 2048] = [B, F×1560, 2048] tokens
           │
     time_embedding         sinusoidal → Linear → dim=2048
     time_projection        Linear → dim=2048×6（调制参数）
           │
           ▼
     text_embedding         [B,512,4096] → Linear → [B,512,2048]
           │
           ▼
     ┌─────────────────────────────────────────────────────┐
     │  CausalWanAttentionBlock × 30 层                    │
     │  每层执行：                                           │
     │  1. LayerNorm + 调制 → CausalWanSelfAttention       │
     │     （因果注意力 + KV Cache 读写）                    │
     │  2. LayerNorm + 调制 → CrossAttention（文本）        │
     │  3. LayerNorm + 调制 → FFN                          │
     └─────────────────────────────────────────────────────┘
           │
           ▼
     CausalHead             LayerNorm + Linear → unpatchify
           │
           ▼
输出: flow_pred [B, C=16, F, H, W]  （速度场，非直接输出视频）
```

---

## 二、CausalWanModel 初始化参数（Wan 1.3B）

文件：[causal_model.py:511-527](../wan/modules/causal_model.py)

| 参数 | 值（1.3B） | 含义 |
|------|-----------|------|
| `dim` | 2048 | Transformer 隐藏维度 |
| `ffn_dim` | 8192 | FFN 中间维度（dim×4） |
| `freq_dim` | 256 | 时间嵌入的正弦维度 |
| `text_dim` | 4096 | UMT5 输出的文本维度 |
| `num_heads` | 16 | 注意力头数 |
| `num_layers` | 32 | Transformer 块数 |
| `head_dim` | 128 | 每头维度（dim/num_heads） |
| `patch_size` | (1,2,2) | 时间不分割，空间 2×2 分块 |
| `local_attn_size` | 12（可配） | 滑动窗口帧数，-1=全局 |
| `sink_size` | 3（可配） | Frame Sink 帧数 |

> 注意：代码注释中说 `num_heads=16`，但 Wan 1.3B 实际是 12 头（`kv_cache["k"]` 的 shape 是 `[B, N, 12, 128]`）。这是因为代码里 `CausalWanSelfAttention` 接收 `dim=1536`（1.3B的实际dim），`num_heads=12`。

---

## 三、`_prepare_blockwise_causal_attn_mask`：BlockMask 构造

文件：[causal_model.py:636-689](../wan/modules/causal_model.py)

这是训练时使用的 FlexAttention 掩码，**推理时不用**（推理用 KV Cache 替代）。

```
例：3帧，frame_seqlen=4，num_frame_per_block=1

token 位置：[0 1 2 3 | 4 5 6 7 | 8 9 10 11]
             帧0       帧1        帧2

ends 数组（attention 的"可见上界"）：
  ends[0:4] = 4    ← 帧0的token只能看到帧0自身（结束位置是4）
  ends[4:8] = 8    ← 帧1的token只能看到帧0+帧1
  ends[8:12] = 12  ← 帧2的token只能看到全部

attention_mask(b, h, q_idx, kv_idx):
    return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
           ↑ kv在q的"可见范围内"   ↑ 自注意力（对角线）
```

**滑动窗口版本**（local_attn_size != -1）：

```python
return ((kv_idx < ends[q_idx]) 
        & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) \
       | (q_idx == kv_idx)
```

同时限制"上界"和"下界"，形成一个局部窗口。

---

## 四、CausalWanSelfAttention：因果自注意力

文件：[causal_model.py:63-358](../wan/modules/causal_model.py)

### 4.1 初始化

```python
class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, ...):
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        self.q = nn.Linear(dim, dim)   # Query 投影
        self.k = nn.Linear(dim, dim)   # Key 投影
        self.v = nn.Linear(dim, dim)   # Value 投影
        self.o = nn.Linear(dim, dim)   # Output 投影
        self.norm_q = WanRMSNorm(dim)  # QK 归一化（防止数值不稳定）
        self.norm_k = WanRMSNorm(dim)
```

### 4.2 前向传播的两条路径

```
forward()
    │
    ├── [kv_cache is None] ── 训练路径（FlexAttention + BlockMask）
    │   │
    │   ├── [teacher forcing] is_tf = (s == seq_lens * 2)
    │   │   └── 分别对 clean/noisy 部分做 rope_apply，再拼接后 flex_attention
    │   │
    │   └── [普通训练] rope_apply + flex_attention(block_mask)
    │
    └── [kv_cache is not None] ── 推理路径（KV Cache 管理）
        │
        ├── causal_rope_apply(q, start_frame=N)
        ├── causal_rope_apply(k, start_frame=N)
        ├── [判断是否需要 roll]
        │   ├── [需要 roll] → 构造 temp_k/v，左移，插入新帧
        │   └── [直接插入] → 构造 temp_k/v，直接插入新帧
        │
        ├── [sink_tokens > 0] → attention(q, cat[k_sink, k_local])
        └── [无 sink]         → attention(q, temp_k[window_start:end])
```

### 4.3 is_recompute 判断

```python
is_recompute = (current_end <= kv_cache["global_end_index"].item()) 
               and (current_start > 0)
```

这个条件在**梯度检查点**重算时为 True。重算时 global_end_index 已经是最终值，但 current_end 还是历史位置，所以 `current_end <= global_end_index`。

`is_recompute=True` 时：
- **不更新** global/local 指针（`_apply_cache_updates` 跳过）
- **保护 sink 区域**不被覆盖（`write_start_index = max(local_start_index, sink_tokens)`）

---

## 五、`causal_rope_apply`：带绝对偏移的旋转位置编码

文件：[causal_model.py:32-60](../wan/modules/causal_model.py)

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    # freqs 是预计算好的频率表，shape [1024, head_dim//2]
    # 按 T/H/W 三个方向拆分
    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)
    
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame+f],  # 时间频率：从第N帧开始
            freqs[1][:h],                          # 高度频率
            freqs[2][:w]                           # 宽度频率
        ], dim=-1)
        x_i = view_as_complex(x[i]) * freqs_i    # 复数乘法 = 旋转
        x_i = view_as_real(x_i).flatten(2)
```

**与普通 `rope_apply` 的区别**：

| | `rope_apply`（训练） | `causal_rope_apply`（推理） |
|-|---------------------|----------------------------|
| 时间频率起点 | `freqs[0][0:f]`（总从第 0 帧） | `freqs[0][start_frame:start_frame+f]`（从当前帧） |
| 使用场景 | 一次处理完整序列 | 逐帧生成，需要绝对位置 |
| start_frame | 无此参数 | 必须传入当前帧的全局位置 |

---

## 六、CausalWanAttentionBlock：完整注意力块

文件：[causal_model.py:361-465](../wan/modules/causal_model.py)

```
输入 x [B, F×1560, 2048]
  │
  ├── [调制] e = (modulation + e).chunk(6)  ← 6个调制参数
  │
  ├── [Self-Attn] norm1(x) × (1+e[1]) + e[0]
  │   → CausalWanSelfAttention → y
  │   → x = x + y × e[2]
  │
  ├── [Cross-Attn] norm3(x)
  │   → WanCrossAttention(x, context, crossattn_cache) → y
  │   → x = x + y
  │
  └── [FFN] norm2(x) × (1+e[4]) + e[3]
      → Linear(8192) → GELU → Linear(2048) → y
      → x = x + y × e[5]
```

**调制机制**（DiT 的 adaptive layer norm）：

`modulation` 是一个学习参数，shape `[1, 6, dim]`，加上时间嵌入 `e`（shape `[B, F, 6, dim]`）后分成 6 块，分别控制 Self-Attn 的 shift/scale/gate、FFN 的 shift/scale/gate。

---

## 七、`_apply_cache_updates`：批量写回 Cache

文件：[causal_model.py:837-889](../wan/modules/causal_model.py)

这个函数在**所有 block 的 forward 完成后**才统一写回 KV Cache，而不是每个 block forward 时立刻写。

**为什么要延迟写回？**

如果 block N 正在用 KV Cache 做 attention，而 block N-1 已经把 cache 更新了，那 block N 看到的是"更新后"的 cache，破坏了数据一致性。延迟写回确保所有 block 都从相同的"快照"读取。

```python
def _apply_cache_updates(self, kv_cache, cache_update_infos):
    for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
        if update_info["action"] == "roll_and_insert":
            # 左移 + 插入
            cache["k"][:, sink:sink+rolled] = cache["k"][:, sink+evicted:...]
            cache["k"][:, write_start:write_end] = new_k
        elif update_info["action"] == "direct_insert":
            # 直接插入
            cache["k"][:, write_start:write_end] = new_k
        
        # 只有非 recompute 时才更新指针
        if not is_recompute:
            cache["global_end_index"].fill_(current_end)
            cache["local_end_index"].fill_(local_end_index)
```

---

## 八、CausalHead：输出层

文件：[causal_model.py:468-496](../wan/modules/causal_model.py)

```python
class CausalHead(nn.Module):
    def forward(self, x, e):
        # x: [B, F×1560, 2048]
        # e: [B, F, 1, 2048]  ← 时间调制参数（仅2个：shift/scale）
        e = (self.modulation + e).chunk(2)  # 2个调制参数
        x = self.head(self.norm(x) * (1+e[1]) + e[0])
        # x: [B, F×1560, patch_vol × out_dim]
        # patch_vol = 1×2×2=4, out_dim=16 → 64
```

`unpatchify` 负责把 `[B, F×1560, 64]` 还原为 `[B, 16, F, 60, 104]`（latent）。

---

## 九、`from_pretrained` 加载流程

```python
CausalWanModel.from_pretrained("wan_models/Wan2.1-T2V-1.3B/", 
                                local_attn_size=12, sink_size=3)
```

1. 读取 `config.json`（ModelMixin 功能）
2. 初始化 `CausalWanModel(local_attn_size=12, sink_size=3, ...)`
3. 加载 `diffusion_pytorch_model.safetensors`（标准 Wan 权重）
4. 权重直接适配：`CausalWanSelfAttention` 的线性层名称与原版 `WanSelfAttention` 相同

**关键区别**：原版 `WanModel` 用标准 FlexAttention + 全序列 BlockMask；`CausalWanModel` 用 KV Cache + 因果注意力，推理时完全绕过 FlexAttention。
