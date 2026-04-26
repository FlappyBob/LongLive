# KV Cache 与注意力机制

**文件**：[wan/modules/attention.py](../wan/modules/attention.py)  
**核心逻辑**：[wan/modules/causal_model.py:63-358](../wan/modules/causal_model.py)（`CausalWanSelfAttention`）

---

## 一、注意力后端选择

```
attention(q, k, v)
    ├── FLASH_ATTN_3_AVAILABLE  →  flash_attn_interface.flash_attn_varlen_func  (Hopper H100)
    ├── FLASH_ATTN_2_AVAILABLE  →  flash_attn.flash_attn_varlen_func            (Ampere A100)
    └── fallback                →  torch.nn.functional.scaled_dot_product_attention
```

**为什么用 varlen（变长）版本？** 因为 batch 内不同样本的序列长度可能不同，varlen 版本通过 `cu_seqlens`（累积序列长度）直接打包，避免 padding 的无效计算。

**训练时的 FlexAttention**：`CausalWanSelfAttention` 在训练路径（无 kv_cache）下使用 `flex_attention`，配合预编译的 `BlockMask` 实现任意形状的因果掩码，性能优于手动 mask + FA。

---

## 二、KV Cache 数据结构

每个 Transformer Block 有一个独立的 cache 字典：

```python
{
    "k": Tensor[B, cache_size, n_heads=12, head_dim=128],  # 历史 key
    "v": Tensor[B, cache_size, n_heads=12, head_dim=128],  # 历史 value
    "global_end_index": Tensor[1],  # 全局视角：已生成的 token 数（永远递增）
    "local_end_index":  Tensor[1],  # 本地视角：cache 中有效 token 的实际末尾位置
}
```

**global vs local 的区别**：

```
生成前 8 帧 (12480 tokens)，cache_size = 6×1560 = 9360:

global_end_index = 12480   ← 全局已生成了多少 token
local_end_index  = 9360    ← cache 实际存了多少 token（受 roll 影响）

当 global > cache_size 时，local 会被 roll 操作"压缩"，
但 global 始终记录真实的生成进度，用于计算 RoPE 偏移。
```

---

## 三、两种 Cache 更新模式

### 3.1 直接插入（direct_insert）

条件：`new_tokens + local_end_index <= cache_size`（cache 还有空间）

```
before: [SINK | ---- | ---- | ---- | (空) ]
                                    ↑ local_end_index
after:  [SINK | ---- | ---- | ---- | NEW  ]
                                          ↑ local_end_index (+=new)
```

### 3.2 滚动插入（roll_and_insert）

条件：`new_tokens + local_end_index > cache_size`（cache 已满）

```
before: [SINK | old1 | old2 | old3 | old4 ]  ← old1 是最旧的非 sink 帧
         ←sink→

evict = new_tokens + local_end - cache_size = 要丢弃的 token 数
roll  = local_end - evict - sink_tokens      = 要向左移的 token 数

操作:
  k[:,sink:sink+roll] = k[:,sink+evict:sink+evict+roll].clone()  # 左移
  v[ 同上 ]
  k[:,写入位置] = new_k                                            # 插入新帧
```

**ASCII 示意**（sink=3帧, window=4帧, 新来2帧）：

```
cache (7 slots):
[S1 S2 S3 | A  B  C  D ]     ← 满了，新来 E F（2帧）
 ← sink →   ← window →

evict = 2，roll = 4-2 = 2
步骤1 左移: [S1 S2 S3 | C  D  _  _ ]    （A、B 被覆盖丢弃）
步骤2 插入: [S1 S2 S3 | C  D  E  F ]    ✓
```

---

## 四、注意力计算时的 token 拼接

有 `sink_tokens > 0` 时，不能只用 local window，还要加上 sink：

```python
k_sink  = cache["k"][:, :sink_tokens]                          # 前 sink 帧
k_local = cache["k"][:, local_start_for_window:local_end]      # 最近 window 帧
k_cat   = torch.cat([k_sink, k_local], dim=1)
```

这样 attention 同时关注"全局锚帧"和"近期窗口帧"，兼顾长程一致性和内存效率。

---

## 五、Recompute 保护

`is_recompute = current_end <= global_end_index and current_start > 0`

梯度检查点会在反向传播时重新执行 forward，这时 `current_end` 仍小于 `global_end_index`（因为它是重新计算历史帧）。此时：

```python
# 不允许新 K/V 覆盖 sink 区域
write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
```

重计算时 sink 区域被保护，不会被覆盖，同时 `global/local_end_index` 也不更新，避免 double-counting。

---

## 六、CrossAttention Cache

交叉注意力的 cache 结构更简单：

```python
{
    "k": Tensor[B, 512, 12, 128],  # 文本 key（512 = text token 数）
    "v": Tensor[B, 512, 12, 128],  # 文本 value
    "is_init": bool                # 是否已初始化
}
```

文本 embedding 不随帧变化，首次计算后直接缓存，后续帧直接复用，**节省大量重复的 cross-attn 计算**。

---

## 七、max_attention_size 动态控制

```python
# local_attn_size = 12 时:
max_attention_size = 12 * 1560 = 18720 tokens

# local_attn_size = -1 (全局) 时:
max_attention_size = 32760 tokens  (= 21帧×1560)
```

`_set_all_modules_max_attention_size` 会遍历所有 `CausalWanSelfAttention` 模块并统一设置，确保每个 block 的注意力上限一致。
