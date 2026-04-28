# KV Cache：数据结构·roll_and_insert·sink·recompute 保护

> 核心文件：[wan/modules/causal_model.py](../wan/modules/causal_model.py)，  
> [pipeline/causal_inference.py:245-283](../pipeline/causal_inference.py)

---

## 一、KV Cache 数据结构

### 1.1 Self-Attention Cache（kv_cache1）

每个 Transformer Block 独立维护一份 cache：

```python
# pipeline/causal_inference.py:261-267
for _ in range(30):  # 30 个 block，各一份
    kv_cache1.append({
        "k": torch.zeros([B, kv_cache_size, 12, 128], dtype, device),
        "v": torch.zeros([B, kv_cache_size, 12, 128], dtype, device),
        "global_end_index": torch.tensor([0], torch.long, device),
        "local_end_index":  torch.tensor([0], torch.long, device)
    })
```

| 字段 | shape | 含义 |
|------|-------|------|
| `k` | `[B, kv_cache_size, 12, 128]` | 历史帧的 Key 张量 |
| `v` | `[B, kv_cache_size, 12, 128]` | 历史帧的 Value 张量 |
| `global_end_index` | `[1]` | 全局已处理到哪个 token（绝对坐标） |
| `local_end_index` | `[1]` | cache 数组中当前有效数据的末尾 |

- `kv_cache_size = local_attn_size × 1560`（默认 `12 × 1560 = 18720`）
- `12` = 注意力头数（Wan 1.3B），`128` = 每头维度

### 1.2 Cross-Attention Cache（crossattn_cache）

```python
# pipeline/causal_inference.py:277-282
crossattn_cache.append({
    "k": torch.zeros([B, 512, 12, 128], dtype, device),  # 512 = 文本 token 数
    "v": torch.zeros([B, 512, 12, 128], dtype, device),
    "is_init": False    # 首次 forward 时计算并存储，之后直接复用
})
```

文本 embedding 在同一视频内不变，所以只计算一次后缓存复用。

---

## 二、双 index 的必要性

```
假设：local_attn_size=12帧, sink_size=3帧, 已生成20帧

global_end_index = 20 × 1560 = 31200
  ↑ 告诉 causal_rope_apply：当前帧的绝对位置是第 20 帧
  ↑ 用于 RoPE 位置编码（必须知道绝对位置）

local_end_index = (3+12) × 1560 = 23400（sink+window）
  ↑ 告诉注意力：cache 数组里有效数据到 23400
  ↑ 用于 cache 数组的读写下标（相对于 cache 数组）

为什么需要两个？
  ─ global 在 roll 后不回退（20 → 21 → 22...），记录绝对历史
  ─ local 在 roll 后不超过 kv_cache_size（≤18720），记录物理位置
```

---

## 三、Cache 更新的两条路径

### 3.1 直接插入（direct_insert）

**触发条件**：cache 还没满，或者是 recompute pass

```
before: [0  0  0 | A  B  C  D  _  _  _  _]   local_end=4帧×1560
               ↑ local_end_index

after (新来2帧 E, F):
        [0  0  0 | A  B  C  D  E  F  _  _]
                                    ↑ local_end_index
```

```python
# causal_model.py:286-314
local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
local_start_index = local_end_index - num_new_tokens

temp_k = kv_cache["k"].clone()
temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:]
```

### 3.2 滚动插入（roll_and_insert）

**触发条件**：`local_attn_size != -1` 且 `num_new_tokens + local_end_index > kv_cache_size`

```
before: [S1 S2 S3 | A  B  C  D  E  F  G  H  I  J  K  L]  ← cache 满
         ← sink →   ← window=12帧 ──────────────────────

新来 M, N（2帧），需要驱逐 A, B

num_evicted_tokens = 2帧 = 3120
num_rolled_tokens = (local_end-sink) - evicted = (12-2) 帧 × 1560

步骤 1 左移（C~L 整体左移 2 帧）:
        [S1 S2 S3 | C  D  E  F  G  H  I  J  K  L  _  _]

步骤 2 插入 M, N:
        [S1 S2 S3 | C  D  E  F  G  H  I  J  K  L  M  N]  ✓
```

对应代码（[causal_model.py:253-266](../wan/modules/causal_model.py)）：

```python
# 在 temp_k/v 副本上操作（不改原 cache）
temp_k = kv_cache["k"].clone()

# 左移：把 sink 之后的 C~L 移到 C 的位置
temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
    temp_k[:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

# 插入 M, N
temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:]
```

**为什么用 temp（副本）而不直接改 kv_cache？**

1. **计算正确性**：当前 forward 要用"插入前"的 cache 做 attention，插入后的 cache 要在 attention 结束后才写回
2. **梯度检查点兼容**：recompute pass 必须看到相同的 cache 快照，不能被当前 forward 污染

---

## 四、Frame Sink 机制

### 4.1 物理布局

```
KV Cache 内存布局（sink=3帧, window=12帧）：

索引: [   0   ][ 1 ][  2  ]|[   3   ]...[  14  ]
       帧 0       帧1   帧2  |  帧3            帧14
       ← Sink（永久保留）→   |  ← Sliding Window →
       4680 tokens           |  18720 tokens
```

### 4.2 注意力计算时的 Sink 拼接

```python
# causal_model.py:321-341
if sink_tokens > 0:
    local_budget = self.max_attention_size - sink_tokens
    
    k_sink  = temp_k[:, :sink_tokens]                          # 前3帧（永久）
    k_local = temp_k[:, local_start_for_window:local_end_index] # 最近N帧（滑动）
    
    k_cat = torch.cat([k_sink, k_local], dim=1)  # 拼接 → [B, sink+local, 12, 128]
    x = attention(roped_query, k_cat, v_cat)
```

**Sink 的作用**：Sink 帧（视频开头）包含主角、场景、风格等"全局锚"信息，让生成的第 200 帧依然与第 0 帧保持一致。

### 4.3 Sink 区域的写保护

```python
# causal_model.py:261
write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
```

- **正常生成**（`is_recompute=False`）：允许写入 sink 区域（初始化阶段，前 3 帧需要正常写入）
- **梯度重算**（`is_recompute=True`）：sink 区域被保护，从 `sink_tokens` 之后开始写

**例外**：KV-Recache 时（`sink_recache_after_switch=True`）强制绕过保护：

```python
# causal_model.py:296-298
if sink_recache_after_switch:
    write_start_index = local_start_index  # 绕过 sink 保护
```

原因：prompt 切换后需要用新 prompt 的语境重写 sink 帧的 K/V。

---

## 五、Cache 大小的显存影响

```
显存估算（Wan 1.3B，bfloat16）：

每个 block 的 KV Cache：
  k 张量: B × kv_cache_size × 12 × 128 × 2 bytes
  v 张量: 同上

kv_cache_size = (sink_size + local_attn_size) × frame_seqlen
             = (3 + 12) × 1560 = 23400 tokens

单个 block 的 KV：
  2 × 1 × 23400 × 12 × 128 × 2 bytes = 144 MB

全部 30 个 block：
  144 × 30 = 4.32 GB

相比之下，如果不用 KV Cache（全局注意力，120帧）：
  kv_cache_size = 120 × 1560 = 187200 tokens
  2 × 1 × 187200 × 12 × 128 × 2 × 30 bytes ≈ 34.5 GB  ← H100 也很紧张
```

---

## 六、`_apply_cache_updates`：延迟批量写回

所有 block 完成 forward 后，才统一写 KV Cache：

```python
# causal_model.py:1043-1044
if kv_cache is not None and cache_update_infos:
    self._apply_cache_updates(kv_cache, cache_update_infos)
```

这么做的原因：
1. **一致性**：block 0 的 forward 用 cache，block 1 的 forward 也用同一个 cache 快照，不会互相干扰
2. **梯度检查点**：recompute 时所有 block 要能重现相同的计算，cache 快照必须一致

---

## 七、总结：完整 Cache 生命周期

```
帧 0~2（初始化 sink）:
  direct_insert → [S0  S1  S2  _  _  _  _  _  _  _  _  _  _  _  _]
  global_end = 3×1560 = 4680
  local_end  = 4680

帧 3~14（填满 window）:
  direct_insert × 12 帧
  global_end = 15×1560 = 23400
  local_end  = 23400  (= kv_cache_size, cache 满)

帧 15（第一次 roll）:
  roll_and_insert: 驱逐帧3, 插入帧15
  [S0  S1  S2 | 4  5  6  7  8  9  10 11 12 13 14 15]
  global_end = 16×1560 = 24960
  local_end  = 23400  (保持不变，cache 已满但物理位置复用)

帧 200（长视频）:
  global_end = 201×1560
  local_end  = 23400  (cache 物理大小固定，只是内容在不断滚动)
  注意力仍可看到: Sink(0,1,2) + 最近12帧（188-200帧）
```
