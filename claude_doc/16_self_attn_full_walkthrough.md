# 16 CausalWanSelfAttention.forward() 全程逐行导读

> 涉及代码：[wan/modules/causal_model.py:97-358](../wan/modules/causal_model.py#L97)
>
> 配置示例（`configs/longlive_inference.yaml`）：
> - B=1, F_block=3, h=30, w=52, frame_seqlen=1560, head_dim=128, num_heads=12, dim=1536
> - sink_size=3, local_attn_size=12, kv_cache 容量=18720 token

---

## 一、入口签名（97-116 行）

```python
def forward(
    self,
    x,                         # 输入特征
    seq_lens,                  # 每个样本的序列长度
    grid_sizes,                # 视频潜空间网格 (F, H, W)
    freqs,                     # 预计算 RoPE 频率表 [1024, 64]
    block_mask,                # 训练时 flex_attention 用的 BlockMask
    kv_cache=None,             # 推理时传入；训练时为 None
    current_start=0,           # 推理：本批 token 在视频序列中的起始下标
    cache_start=None,          # 推理：sink-recache 时使用的另一个起始位置
    sink_recache_after_switch=False  # prompt 切换后是否需要重建 sink
):
```

| 参数 | 推理时 shape / 值 | 含义 |
|------|-------------------|------|
| `x` | `[1, 4680, 1536]` | 注意：注释写的是 `[B,L,n,d]` 但实际 b,s 直接从 `x.shape[:2]` 取，进入 qkv_fn 才会拆头 |
| `seq_lens` | `[1]`，例 `[4680]` | 每个样本有效 token 数 |
| `grid_sizes` | `[1, 3]`，例 `[[3,30,52]]` | 当前块的 (F, H, W) |
| `freqs` | `[1024, 64]` 复数 | RoPE 频率表 |
| `block_mask` | BlockMask 或 None | 训练用因果掩码 |
| `kv_cache` | dict 或 None | 推理 dict 含 k/v/global_end_index/local_end_index |
| `current_start` | int | 例 10920 = 第 7 帧起 |

---

## 二、读取头数和维度（117 行）

```python
b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
```

```
x.shape = [B, L, dim] = [1, 4680, 1536]
x.shape[:2] = (1, 4680)
*x.shape[:2] 解包 → b=1, s=4680
n = self.num_heads = 12
d = self.head_dim   = 128
```

`*` 是 Python 的解包运算符，把 tuple 展开成多个值。

---

## 三、cache_start 默认值（118-119 行）

```python
if cache_start is None:
    cache_start = current_start
```

`cache_start` 是 sink-recache 模式下使用的另一种起始位置。
正常推理时和 `current_start` 相等。本函数体内其实没有直接用 `cache_start`，它会在调用方判断 sink 重建时用到。

---

## 四、QKV 投影（121-128 行）

```python
def qkv_fn(x):
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    return q, k, v

q, k, v = qkv_fn(x)
```

shape 流：

```
x:                    [1, 4680, 1536]
self.q(x):            [1, 4680, 1536]   Linear(1536→1536)
self.norm_q(...):     [1, 4680, 1536]   RMSNorm，沿最后一维归一化
.view(1, 4680, 12, 128): [1, 4680, 12, 128]  把 1536 拆成 12 个头 × 128 维
```

`q, k, v` 都是 `[1, 4680, 12, 128]`，多头 attention 的标准布局。

| 子模块 | 作用 |
|--------|------|
| `self.q/k/v` | 三个独立的 Linear，把同一个 x 投影成 Q/K/V 三种不同视角 |
| `self.norm_q/k` | RMSNorm 仅作用于 Q/K（不动 V），稳定数值，提升训练收敛 |

V 不归一化，因为 V 直接进入加权求和，归一化会把数值范围压缩，破坏信息。

---

## 五、训练分支（130-204 行）：`kv_cache is None`

整个训练分支用 `flex_attention` 一次处理整个序列，靠 `block_mask` 控制因果性。

### 5.1 检测 Teacher Forcing（132 行）

```python
is_tf = (s == seq_lens[0].item() * 2)
```

含义：训练时如果输入序列长度是有效长度的 2 倍，说明是 **teacher forcing 模式**——同一段视频的"clean 版"和"noisy 版"被拼接在 dim=1 一起前向。
SelfForcing 训练用这种方式让 critic 同时对两个版本做对比。

### 5.2 TF 分支：分别 RoPE 再拼回（133-146 行）

```python
q_chunk = torch.chunk(q, 2, dim=1)   # 拆成 [clean, noisy]
k_chunk = torch.chunk(k, 2, dim=1)
roped_query, roped_key = [], []
for ii in range(2):
    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
    roped_query.append(rq)
    roped_key.append(rk)
roped_query = torch.cat(roped_query, dim=1)
roped_key   = torch.cat(roped_key, dim=1)
```

shape 流（假设 s=2L, L=4680）：

```
q:           [1, 9360, 12, 128]
chunk 后:    [1, 4680, 12, 128] × 2
rope_apply:  对每个 chunk 独立做 3D RoPE（位置都从 0 开始）
cat 后:      [1, 9360, 12, 128]
```

为什么要分别 RoPE？clean 部分和 noisy 部分代表"同一视频的两个去噪状态"，**位置应该一样**（都是第 0~F-1 帧）。如果整段一起 RoPE，noisy 部分会被错误地编码为第 F~2F-1 帧。

### 5.3 padding 到 128 的整数倍（148-166 行）

```python
padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
padded_roped_query = torch.cat(
    [roped_query,
     torch.zeros([..., padded_length, n, d], device=..., dtype=v.dtype)],
    dim=1
)
# 同理 padded_roped_key, padded_v
```

为什么 pad？`flex_attention` 对序列长度有"必须是 128 的倍数"的要求（FlashAttention 内部 tile 大小）。
不够就在末尾补零。

### 5.4 调用 flex_attention（168-173 行）

```python
x = flex_attention(
    query=padded_roped_query.transpose(2, 1),   # [B, n, L, d]
    key=padded_roped_key.transpose(2, 1),
    value=padded_v.transpose(2, 1),
    block_mask=block_mask
)[:, :, :-padded_length].transpose(2, 1)
```

shape 流：

```
padded_roped_query: [1, 9360+pad, 12, 128]
.transpose(2, 1):   [1, 12, 9360+pad, 128]   flex_attention 要求 (B, n, L, d) 布局
flex_attention 输出:[1, 12, 9360+pad, 128]
[:, :, :-padded_length]: 去掉末尾 padding → [1, 12, 9360, 128]
.transpose(2, 1):   [1, 9360, 12, 128]       恢复原始布局
```

`block_mask` 是 BlockMask 对象，编码了"哪些 token 可以看哪些 token"的因果约束（chunk 内全连接，跨 chunk 因果）。

### 5.5 普通训练分支（175-204 行）

`is_tf=False` 时不分 chunk，整段一次性 RoPE + flex_attention，逻辑和 TF 分支几乎一样，只少了 chunk/cat 步骤。

---

## 六、推理分支（205-348 行）：`kv_cache is not None`

### 6.1 算每帧 token 数和当前块绝对帧号（206-207 行）

```python
frame_seqlen = math.prod(grid_sizes[0][1:]).item()
# grid_sizes[0] = tensor([3, 30, 52])
# [1:] 切掉帧数 → tensor([30, 52])
# math.prod → 1560 = 每帧 token 数

current_start_frame = current_start // frame_seqlen
# current_start = 10920 → current_start_frame = 7（当前块从第 7 帧开始）
```

### 6.2 带绝对位置偏移的 3D RoPE（208-211 行）

```python
roped_query = causal_rope_apply(
    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
roped_key = causal_rope_apply(
    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
```

shape 不变：`[1, 4680, 12, 128]`。
内部 T 轴用 `freqs[0][7:10]` 旋转（详见 [13_rope_explained.md](13_rope_explained.md) 和 [15_self_attn_kv_cache_walkthrough.md](15_self_attn_kv_cache_walkthrough.md)）。

`.type_as(v)`：causal_rope_apply 内部用 float64 复数运算保证精度，这里转回 v 的 dtype（通常 bfloat16）。

### 6.3 计算 cache 更新所需位置（213-217 行）

```python
current_end = current_start + roped_query.shape[1]
# = 10920 + 4680 = 15600

sink_tokens = self.sink_size * frame_seqlen
# = 3 × 1560 = 4680（sink 区占用的 token 数）

kv_cache_size = kv_cache["k"].shape[1]
# = 18720（cache 数组容量）

num_new_tokens = roped_query.shape[1]
# = 4680
```

### 6.4 判断是否 recompute（228-230 行）

```python
cache_update_info = None
is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
```

| 触发条件 | 含义 |
|----------|------|
| `current_end <= global_end_index` | 这段位置已经被写过了 |
| `current_start > 0` | 不是视频第一块 |

`is_recompute=True` 表示这是 **同一段位置的重新前向**（4 步去噪中间步、context pass），需要"原位覆写"且"保护 sink 区"。

### 6.5 分支 A：cache 满了，需要 roll_and_insert（231-282 行）

进入条件三个 AND：
- `local_attn_size != -1`（启用了滑动窗口）
- `current_end > global_end_index`（不是 recompute，是新推进）
- `num_new_tokens + local_end_index > kv_cache_size`（再写下去就溢出）

#### 6.5.1 算驱逐数和滚动数（235-236 行）

```python
num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
# 多出来溢出的部分 = 必须驱逐的 token 数

num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
# 留下来的"非 sink"老 token 数（这部分要左移给新 token 让位）
```

具体例子：cache 已满 18720，要写入 4680 个新 token：

```
num_evicted_tokens = 4680 + 18720 - 18720 = 4680
num_rolled_tokens  = 18720 - 4680 - 4680 = 9360
                     总容量  驱逐    sink保留
```

意思是：驱逐最旧的 4680 token（非 sink 区开头的 4680 个），剩下 9360 个老 token 整体左移给新的 4680 让位。

#### 6.5.2 算新 token 的写入位置（244-246 行）

```python
local_end_index = kv_cache["local_end_index"].item() + current_end \
                  - kv_cache["global_end_index"].item() - num_evicted_tokens
local_start_index = local_end_index - num_new_tokens
```

公式直觉：global 增量 = `current_end - global_end_index`，扣除驱逐的部分，加到 local_end 上。

#### 6.5.3 准备临时 cache（250-257 行）

```python
temp_k = kv_cache["k"].clone()
temp_v = kv_cache["v"].clone()

# 把 sink 区之后、待保留的 num_rolled_tokens 段，从原位置左移到 sink 区紧邻处
temp_k[:, sink_tokens : sink_tokens + num_rolled_tokens] = \
    temp_k[:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
temp_v[:, ...] = ...  # v 同理
```

shape 流（cache 形状 `[1, 18720, 12, 128]`）：

```
源切片:    temp_k[:, 4680+4680 : 4680+4680+9360] = temp_k[:, 9360:18720]   ← 老 token 中要保留的
目标位置:  temp_k[:, 4680      : 4680+9360     ] = temp_k[:, 4680:14040]   ← sink 紧邻区

效果：sink 区不动；非 sink 区里最老的 4680 被丢弃，剩下的 9360 整体左移到位置 4680~14039
```

`.clone()` 防止"边读边写"的内存重叠错误。

#### 6.5.4 写入新 token（261-266 行）

```python
write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
roped_offset = max(0, write_start_index - local_start_index)
write_len = max(0, local_end_index - write_start_index)
if write_len > 0:
    temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset+write_len]
    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset+write_len]
```

| 变量 | 作用 |
|------|------|
| `write_start_index` | recompute 时 `max(local_start, sink_tokens)`，**强制不能写入 sink 区** |
| `roped_offset` | 如果写起点被前移到 sink_tokens，新 token 也要相应跳过前面这段 |
| `write_len` | 实际写多少个 token |

为什么 recompute 时要保护 sink？4 步去噪中，前面的步骤已经把"干净的 sink K/V"写好了，后面的步骤是用"含噪输入"再算一次，这次的 K/V 不应该污染 sink。
正常推进生成时（视频开头几帧）允许写入 sink 区，因为 sink 本身就是"前 3 帧"，不写谁来填？

#### 6.5.5 保存更新计划（269-282 行）

```python
cache_update_info = {
    "action": "roll_and_insert",
    "sink_tokens": sink_tokens,
    "num_rolled_tokens": num_rolled_tokens,
    "num_evicted_tokens": num_evicted_tokens,
    "local_start_index": local_start_index,
    "local_end_index": local_end_index,
    "write_start_index": write_start_index,
    "write_end_index": local_end_index,
    "new_k": roped_key[:, roped_offset:roped_offset+write_len],
    "new_v": v[:,         roped_offset:roped_offset+write_len],
    "current_end": current_end,
    "is_recompute": is_recompute,
}
```

只是**记账**，真实写入由 `_apply_cache_updates()` 在 attention 算完后执行。

### 6.6 分支 B：cache 还有余量，直接 append（286-315 行）

```python
local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
local_start_index = local_end_index - num_new_tokens

temp_k = kv_cache["k"].clone()
temp_v = kv_cache["v"].clone()

write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
if sink_recache_after_switch:
    write_start_index = local_start_index   # 强制覆盖 sink，prompt 切换后重建 sink
roped_offset = max(0, write_start_index - local_start_index)
write_len = max(0, local_end_index - write_start_index)
if write_len > 0:
    temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset+write_len]
    temp_v[:, write_start_index:local_end_index] = v[:,         roped_offset:roped_offset+write_len]

cache_update_info = {
    "action": "direct_insert",
    "local_start_index": local_start_index,
    "local_end_index": local_end_index,
    "write_start_index": write_start_index,
    "write_end_index": local_end_index,
    "new_k": roped_key[:, roped_offset:roped_offset+write_len],
    "new_v": v[:,         roped_offset:roped_offset+write_len],
    "current_end": current_end,
    "is_recompute": is_recompute,
}
```

逻辑和分支 A 几乎一样，但**没有 roll 操作**，只是把新 K/V 追加到 cache 末尾。

`sink_recache_after_switch=True` 是 **KV-Recache 机制** 的入口：用户切换 prompt 后，需要用新 prompt 重新生成前 3 帧的 K/V 并覆写 sink。这个 flag 强制 `write_start_index = local_start_index`，允许写入 sink 区。

---

## 七、Attention 计算（321-348 行）

### 7.1 有 sink 的情况（321-341 行）

```python
if sink_tokens > 0:
    local_budget = self.max_attention_size - sink_tokens
    # max_attention_size = local_attn_size × 1560 = 12 × 1560 = 18720
    # local_budget = 18720 - 4680 = 14040

    k_sink = temp_k[:, :sink_tokens]   # [1, 4680, 12, 128]，永久保留的前 3 帧
    v_sink = temp_v[:, :sink_tokens]   # [1, 4680, 12, 128]

    if local_budget > 0:
        local_start_for_window = max(sink_tokens, local_end_index - local_budget)
        # 滑动窗口左端：从 local_end_index 往前看 local_budget 个 token，但不越过 sink 区
        k_local = temp_k[:, local_start_for_window:local_end_index]
        v_local = temp_v[:, local_start_for_window:local_end_index]
        k_cat = torch.cat([k_sink, k_local], dim=1)
        v_cat = torch.cat([v_sink, v_local], dim=1)
    else:
        k_cat = k_sink
        v_cat = v_sink

    x = attention(roped_query, k_cat, v_cat)
```

shape 流（典型情况）：

```
k_sink:  [1, 4680, 12, 128]   sink 区，永久保留
k_local: [1, ≤14040, 12, 128] 滑动窗口里的近期 token
k_cat:   [1, ≤18720, 12, 128] 拼起来送进 attention
roped_query: [1, 4680, 12, 128]
attention 输出: [1, 4680, 12, 128]
```

注意：这里的 `attention` 函数（在 `wan/modules/attention.py`）调用 FlashAttention，**不需要传 causal=True**。因为 cache 本身的结构（只存历史 K/V）已经保证了因果性。

### 7.2 无 sink 的情况（342-348 行）

```python
else:
    window_start = max(0, local_end_index - self.max_attention_size)
    x = attention(
        roped_query,
        temp_k[:, window_start:local_end_index],
        temp_v[:, window_start:local_end_index]
    )
```

纯滑动窗口，没有 sink。窗口左端从 `local_end_index - max_attention_size` 开始（不能小于 0）。

---

## 八、输出投影 + 返回（350-358 行）

```python
x = x.flatten(2)         # [1, 4680, 12, 128] → [1, 4680, 1536]
x = self.o(x)            # Linear(1536→1536)
                         # x.shape = [1, 4680, 1536]

if kv_cache is not None:
    return x, (current_end, local_end_index, cache_update_info)
else:
    return x
```

| 返回项 | 含义 |
|--------|------|
| `x` | attention 后的特征，shape 和输入一致 |
| `current_end` | 用来更新 `kv_cache["global_end_index"]` |
| `local_end_index` | 用来更新 `kv_cache["local_end_index"]` |
| `cache_update_info` | 真实写入 K/V 的执行计划 |

外层 `_forward_inference()` 会把这三项收集起来，统一在所有 30 个 block 都算完后才调用 `_apply_cache_updates()` 真正写入 cache。

---

## 九、shape 总览表

| 阶段 | x | q/k/v | roped_q/k | k_cat / v_cat |
|------|---|-------|-----------|---------------|
| 入口 | `[1,4680,1536]` | — | — | — |
| qkv_fn 后 | — | `[1,4680,12,128]` | — | — |
| RoPE 后 | — | — | `[1,4680,12,128]` | — |
| 拼 sink+local | — | — | — | `[1,≤18720,12,128]` |
| attention 输出 | `[1,4680,12,128]` | — | — | — |
| flatten+Linear | `[1,4680,1536]` | — | — | — |

---

## 十、训练 vs 推理路径对比

```
                 训练分支（kv_cache=None）           推理分支（kv_cache=dict）
─────────────────────────────────────────────────────────────────────────────
位置编码          rope_apply（位置从 0）              causal_rope_apply（位置从 start_frame）
attention 实现    flex_attention + block_mask        attention（FlashAttention）
因果性保证        block_mask 显式定义                  KV cache 只存历史 → 隐式因果
长度处理          pad 到 128 倍数                      不需要 pad（FA 支持任意长度）
cache 操作        无                                   roll_and_insert / direct_insert
返回值            x                                   (x, (current_end, local_end_index, cache_update_info))
```

---

## 十一、关键设计要点

| 设计点 | 出处 | 作用 |
|--------|------|------|
| Q/K 用 RMSNorm，V 不归一化 | 94-95, 122-125 | Q/K 影响 attention 权重，归一化稳定数值；V 进入加权和，归一化会破坏信息 |
| sink_size 永久保留 | 214, 254-257, 324-325 | 让模型始终能看到视频开头，长视频不忘记起始内容 |
| 滑动窗口 + sink 拼接 | 332-333 | 限制 attention 长度为 O(local + sink)，O(L²) → O(L)，支持超长视频 |
| 延迟 cache 写入 | 229, 269-282, 305-315 | attention 始终基于干净 cache 计算，避免多卡 / 重入时不一致 |
| recompute 保护 sink | 261, 295 | 4 步去噪中前几步的 sink K/V 不被含噪输入污染 |
| sink_recache_after_switch | 296-297 | KV-Recache：prompt 切换后用新 prompt 重建 sink |
| RoPE 内部 float64 + type_as | 209, 211 | 复数旋转用高精度，结果转回 bf16 兼顾速度和精度 |

---

## 十二、与其他文档的关系

| 文档 | 内容 |
|------|------|
| [01_causal_model.md](01_causal_model.md) | 整个 CausalWanModel 的模块结构 |
| [02_kv_cache_attention.md](02_kv_cache_attention.md) | KV Cache 数据结构和 roll_and_insert 详细原理 |
| [13_rope_explained.md](13_rope_explained.md) | RoPE / 3D RoPE / causal_rope_apply 原理 |
| [14_inference_walkthrough.md](14_inference_walkthrough.md) | 整个推理流程的全程导读 |
| [15_self_attn_kv_cache_walkthrough.md](15_self_attn_kv_cache_walkthrough.md) | 仅聚焦 206-230 行的快速版 |
| **本文档** | **完整 97-358 行的逐行讲解** |

---
