# 15 CausalWanSelfAttention 推理路径逐行讲解

> 聚焦 `causal_model.py:206-289` 这段：3D RoPE 应用 → 计算 cache 写入位置 → 区分 recompute / 正常推进 / 滑动窗口 roll。
>
> 配置示例（`configs/longlive_inference.yaml`）：
> - B=1, F_block=3, h=30, w=52, frame_seqlen=1560
> - sink_size=3（前 3 帧永久保留）
> - local_attn_size=12（滑动窗口容量 12 帧）
> - kv_cache 总容量 = local_attn_size × frame_seqlen = 18720 token

---

## 一、整体定位

这段代码在 `CausalWanSelfAttention.forward()` 的推理分支（即 `kv_cache is not None`）。
它的任务是：

1. 对当前块的 Q/K 做 **3D RoPE**，并按当前块在视频中的绝对帧位置偏移
2. 计算这批新 token 在 KV Cache 全局坐标系中的范围
3. 判断本次是 **recompute（重算同一段）** 还是 **正向推进生成**
4. 如果 cache 容量已满，准备 **roll_and_insert**（左移驱逐最旧帧 + 插入新帧）的更新计划

注意：这段代码**不直接修改** `kv_cache`，只准备 `cache_update_info`。
真正的写入由 `_apply_cache_updates()` 在 attention 计算结束后统一执行（避免 attention 进行中破坏 cache 一致性）。

---

## 二、KV Cache 双索引设计回顾

| 字段 | 含义 | 单调性 | 用途 |
|------|------|--------|------|
| `global_end_index` | cache 已写到的最远**绝对** token 位置 | 永远递增 | 配合 `current_end` 判断是否 recompute |
| `local_end_index` | cache 已写到的**物理数组**位置 | 滚动后会被截断到 ≤ kv_cache_size | 实际索引 cache 数组 |

举例：视频已生成 15 帧（cache 容量 12 帧 + 3 sink）
- `global_end_index = 15 × 1560 = 23400`（一直递增）
- `local_end_index = 18720`（已经撑满 cache）

---

## 三、逐行讲解

### 3.1 算出当前块的绝对帧号

```python
# causal_model.py:206-207
frame_seqlen = math.prod(grid_sizes[0][1:]).item()
# grid_sizes[0] = (f, h, w) = (3, 30, 52)
# grid_sizes[0][1:] = (30, 52) → math.prod = 1560
# frame_seqlen：每帧 token 数 = h × w = 1560

current_start_frame = current_start // frame_seqlen
# current_start：外部传入，本批 token 在整个视频 token 序列中的起始下标
#   外部按 current_start_frame_idx × frame_seqlen 计算后传入
#   例：第 7 块开始 → current_start = 7 × 1560 = 10920
# 除以每帧 token 数 → current_start_frame = 当前块的起始绝对帧号 = 7
```

### 3.2 对 Q/K 做带绝对帧偏移的 3D RoPE

```python
# causal_model.py:208-211
roped_query = causal_rope_apply(
    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
# q.shape = [B, num_new_tokens, n_heads, head_dim] = [1, 4680, 12, 128]
# 内部 T 轴用 freqs[0][7:10] 旋转
# Q 携带的位置信息变成"我是视频的第 7~9 帧"
# .type_as(v)：causal_rope_apply 内部用 float64 复数运算保证精度，
#               这里转回 v 的 dtype（通常 bfloat16）以匹配后续 attention

roped_key = causal_rope_apply(
    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
# K 同样旋转。旋转后的 K 之后会被写入 kv_cache，
# 这样未来块的 Q 来 attend 时，cache 里的 K 已经带了正确的绝对帧位置
```

### 3.3 计算 cache 更新所需的位置变量

```python
# causal_model.py:213-217
current_end = current_start + roped_query.shape[1]
# roped_query.shape[1] = num_new_tokens = f × h × w = 4680
# current_end = 这批新 Q 在视频 token 序列中的结束下标（不含）
# 例：current_start=10920 → current_end=10920+4680=15600

sink_tokens = self.sink_size * frame_seqlen
# self.sink_size = 3（前 3 帧永久保留作为 attention sink）
# sink_tokens = 3 × 1560 = 4680
# 含义：KV Cache 数组前 4680 个位置永远存放视频开头 3 帧的 K/V，
#       不会被滑动窗口替换

kv_cache_size = kv_cache["k"].shape[1]
# kv_cache["k"].shape = [B, total_capacity, n_heads, head_dim]
# kv_cache_size = 18720（local_attn_size=12 帧 × 1560 token/帧）

num_new_tokens = roped_query.shape[1]
# = 4680
```

### 3.4 关键判断：recompute vs 推进生成

```python
# causal_model.py:229-230
cache_update_info = None
# 占位。稍后会填入"应该往 cache 哪里写、写什么"的信息
# 不直接动 kv_cache，等 attention 算完后统一更新

is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
```

`is_recompute` 两个条件并存：

| 条件 | 含义 |
|------|------|
| `current_end <= global_end_index` | 这批 token 范围在 cache 里**已经写过** |
| `current_start > 0` | 不是视频第一块（第一块时 cache 还空） |

何时触发 `is_recompute=True`？

| 场景 | 是否 recompute | 说明 |
|------|----------------|------|
| 4 步去噪的中间步骤 | ✓ | 同一帧用 t=1000, 750, 500, 250 反复前向，每步都重算同一段 |
| Context pass（t=0 干净帧） | ✓ | 去噪结束后用干净输入再前向一次，把"清版" K/V 覆写进 cache |
| 推进生成下一块新帧 | ✗ | 这批 token 是 cache 里没写过的新位置 |

后续逻辑会根据 `is_recompute` 决定：
- recompute 时**保护 sink 区不被覆盖**（避免 4 步去噪过程中污染 sink）
- 正常推进时允许写入 sink 区（视频开头的前 3 帧本就需要写入 sink）

---

## 四、完整数据流（以第 7 块为例）

```
输入参数：
    current_start = 10920（第 7 帧的起始 token）
    grid_sizes    = [[3, 30, 52]]
    q, k, v.shape = [1, 4680, 12, 128]

步骤 1：换算绝对帧号
    frame_seqlen        = 30 × 52 = 1560
    current_start_frame = 10920 // 1560 = 7

步骤 2：3D RoPE（带 start_frame=7 偏移）
    roped_query = causal_rope_apply(q, ..., start_frame=7) → [1, 4680, 12, 128]
    roped_key   = causal_rope_apply(k, ..., start_frame=7) → [1, 4680, 12, 128]

步骤 3：计算 cache 坐标
    current_end     = 10920 + 4680 = 15600
    sink_tokens     = 3 × 1560     = 4680
    kv_cache_size   = 18720
    num_new_tokens  = 4680

步骤 4：分支判断
    假设 kv_cache["global_end_index"] = 10920（cache 已写到第 7 帧前）
    is_recompute = (15600 <= 10920) and (10920 > 0)
                 = False
    → 这是正向推进生成，需要新分配位置写入

    若是 4 步去噪的第 2 步，cache 里第 7~9 帧已经被第 1 步写过了：
    假设 kv_cache["global_end_index"] = 15600
    is_recompute = (15600 <= 15600) and (10920 > 0)
                 = True
    → 这是 recompute，覆写同一段位置，但保护 sink 区
```

---

## 五、为什么不直接动 kv_cache？

```python
cache_update_info = None  # 先占位
# ... 计算各种位置变量
cache_update_info = {"action": "roll_and_insert", ...}  # 填入更新计划
# ↓ 后续：先做 attention 计算
# ↓ 计算结束后调用 _apply_cache_updates(cache_update_info) 才真正写入
```

原因：attention 计算时需要"完整的 cache 视图"（旧 K/V + 新 K/V）。
如果在计算前就动了原 cache，多卡通信、torch.compile 重入、梯度检查点等场景下容易出现：
- 不同 rank 看到的 cache 状态不一致
- 重新前向时 cache 状态已变，导致结果错位

把"准备更新计划"和"真正写入"分离，可以让 attention 始终基于一份**干净不变**的 cache + 临时拼接的新 K/V 计算，更新动作延迟到完整结束后统一执行。

---

## 六、与后续 roll_and_insert 的衔接

`is_recompute` 之后紧跟的代码（`causal_model.py:231-289`）会处理两种情况：

```
                    ┌─ cache 容量不够 → roll_and_insert
推进生成（非 recompute）┤
                    └─ cache 还有余量 → 直接 append 到末尾

recompute → 不需要 roll，原位覆写（但保护 sink 区前 4680 token 不被覆盖）
```

详见 [02_kv_cache_attention.md](02_kv_cache_attention.md) 和 [11_innovations_deep_dive.md](11_innovations_deep_dive.md) 中 Frame Sink 部分。

---

## 七、变量速查表

| 变量 | 类型 | 示例值 | 来源 | 作用 |
|------|------|--------|------|------|
| `current_start` | int | 10920 | 外部传入 | 本批 token 在视频序列中的起始下标 |
| `frame_seqlen` | int | 1560 | h × w | 每帧 token 数 |
| `current_start_frame` | int | 7 | 算出 | 当前块的起始绝对帧号 |
| `roped_query/key` | tensor | [1,4680,12,128] | RoPE 输出 | 已注入 3D 位置信息的 Q/K |
| `current_end` | int | 15600 | current_start + new_tokens | 本批结束位置 |
| `sink_tokens` | int | 4680 | sink_size × frame_seqlen | sink 区占用的 token 数 |
| `kv_cache_size` | int | 18720 | cache 数组容量 | 物理数组上限 |
| `num_new_tokens` | int | 4680 | 当前块 token 数 | 这次要写入的量 |
| `is_recompute` | bool | True/False | 比较 current_end 和 global_end_index | 决定是否覆写 + 保护 sink |
| `cache_update_info` | dict / None | 见下文 | 后续填充 | 延迟执行的 cache 更新计划 |

---
