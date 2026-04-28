# RoPE / 3D RoPE / causal_rope_apply 详解

**涉及文件**：
- [wan/modules/model.py](../wan/modules/model.py)（标准 RoPE 实现）
- [wan/modules/causal_model.py](../wan/modules/causal_model.py)（因果 RoPE 实现）

---

## 一、RoPE 是什么？原理从零讲起

### 1.1 问题背景：注意力机制不知道顺序

Transformer 的自注意力公式是：

```
score(Q_i, K_j) = Q_i · K_j / sqrt(d)
```

这个点积只衡量"语义相似度"，完全不含位置信息。  
词 A 在第 1 位还是第 100 位，算出来的分数一样。  
所以必须额外引入"位置编码"告诉模型每个 token 在哪里。

---

### 1.2 绝对位置编码的问题

最简单的方法是 Sinusoidal 绝对位置编码：把位置 p 转成一个向量 PE(p)，加到 embedding 上。

问题：模型只能看到位置的绝对值，感知不到两个 token **之间的距离**（相对位置）。  
比如 token 5 和 token 7 之间距离 2，token 100 和 token 102 之间也是距离 2，但绝对位置编码没法让模型把这两种情况等同看待。

---

### 1.3 RoPE 的核心思想：用旋转编码相对位置

RoPE（Rotary Position Embedding）的想法很优雅：

> 不把位置信息加到向量上，而是把向量**旋转一个和位置有关的角度**。  
> 旋转后，两个 token 的点积自然只依赖它们的**相对位置差**。

**数学直觉（二维版本）**：

把向量的每两个分量 (x₁, x₂) 看成复数 z = x₁ + i·x₂。  
位置 p 对应旋转角 θ·p（θ 是一个固定频率）。

```
旋转后的向量：z' = z · e^(i·θ·p)
```

当计算 Q_i 和 K_j 的点积时：

```
Q_i · K_j = Re(q_i · conj(k_j))
           = Re((q · e^(iθp_i)) · conj(k · e^(iθp_j)))
           = Re(q · conj(k) · e^(iθ(p_i - p_j)))
```

结果只依赖 `p_i - p_j`（位置差），不依赖 `p_i` 或 `p_j` 的绝对值。这就实现了相对位置感知。

---

### 1.4 实际实现：多频率旋转

实际 embedding 维度 d 很大（比如 128），把它拆成 d/2 对，每对用不同频率 θ_k 旋转：

```
θ_k = 1 / (10000^(2k/d)),  k = 0, 1, ..., d/2 - 1
```

低频（小 θ_k，k 大）：旋转慢，感知长距离位置差。  
高频（大 θ_k，k 小）：旋转快，感知短距离位置差。

对应代码（[model.py:29-36](../wan/modules/model.py#L29)）：

```python
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),           # 位置 0, 1, 2, ..., max_seq_len-1
        1.0 / torch.pow(theta,
            torch.arange(0, dim, 2).to(torch.float64).div(dim))
        # 频率 1/θ^(0/d), 1/θ^(2/d), 1/θ^(4/d), ...
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    # torch.polar(r, angle) = r * e^(i*angle)
    # 把角度转成复数单位圆上的点（模长为 1 的复数）
    return freqs  # shape: [max_seq_len, dim/2]，每个元素是复数
```

`torch.outer(a, b)` 是外积：结果 `[i, j] = a[i] * b[j]`。  
这里 a 是位置向量，b 是频率向量，外积得到的 `freqs[p, k]` = 位置 p 在频率 k 下的旋转角度。

`torch.polar(r, angle)` 把实数角度转成复数：`r * (cos(angle) + i*sin(angle))`，这里 r=1 所以就是 `e^(i*angle)`。

---

## 二、3D RoPE 在 WanModel 中的实现

### 2.1 为什么需要 3D？

普通文本是 1D 序列（时间轴）。视频是 3D 数据：时间 T、高度 H、宽度 W。

每个 latent token 在视频里有一个 (t, h, w) 坐标。  
3D RoPE 为三个维度分别计算旋转：  
- T 维度旋转 → 感知帧间时序位置  
- H 维度旋转 → 感知空间高度位置  
- W 维度旋转 → 感知空间宽度位置

三个维度的旋转角拼在一起，作用到 Q/K 向量上。

---

### 2.2 频率表的构造

每个注意力头的维度 d = dim / num_heads（1.3B 模型中 d=128）。  
这 128 个维度要分给三个轴：

```
T 轴分到：d - 4*(d//6) = 128 - 4*21 = 128-84 = 44 维（d//6=21，实际按整除结果）
H 轴分到：2*(d//6)     = 42 维
W 轴分到：2*(d//6)     = 42 维
共计：44 + 42 + 42 = 128 维 ✓
```

对应代码（[causal_model.py:612-617](../wan/modules/causal_model.py#L612)）：

```python
d = dim // num_heads   # = 128

self.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),   # T 轴频率表，shape [1024, 22]（复数维）
    rope_params(1024, 2 * (d // 6)),        # H 轴频率表，shape [1024, 21]
    rope_params(1024, 2 * (d // 6))         # W 轴频率表，shape [1024, 21]
], dim=1)
# self.freqs shape: [1024, 64]（复数，实际表示 128 维实数旋转）
```

`1024` 是预计算的最大序列长度（T/H/W 都不会超过 1024）。  
推理时从这张表里取前 f/h/w 行用于当前视频尺寸。

---

### 2.3 rope_apply：训练时如何使用

训练时，当前 block 处理整个序列（所有帧一次性进来），调用标准 `rope_apply`（[model.py:40-67](../wan/modules/model.py#L40)）：

```python
def rope_apply(x, grid_sizes, freqs):
    # x shape: [B, seq_len, num_heads, head_dim]
    # grid_sizes: 每个样本的 [f, h, w]，比如 [[21, 30, 52]]
    n, c = x.size(2), x.size(3) // 2
    # n = num_heads = 12, c = head_dim/2 = 64（复数维度数）

    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)
    # 按 T/H/W 拆分频率表

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w  # 比如 21*30*52 = 32760

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        # view_as_complex: 把最后一维的 [a, b] 解释为复数 a + ib
        # x_i shape: [seq_len, n, c]（复数张量）

        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # T 轴：取前 f 行
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # H 轴：取前 h 行
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),  # W 轴：取前 w 行
        ], dim=-1).reshape(seq_len, 1, -1)
        # freqs_i shape: [seq_len, 1, c]（1 是广播到所有 head）

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # 复数乘法 = 旋转，view_as_real 转回实数，flatten 恢复 [seq_len, n, head_dim]

        output.append(x_i)
    return torch.stack(output).type_as(x)
```

关键行 `freqs[0][:f]`：T 轴从第 0 帧开始取，取 f 行。  
整个视频的第 0 帧用 `freqs[0][0]` 旋转，第 1 帧用 `freqs[0][1]`，以此类推。

训练时在 [causal_model.py:176](../wan/modules/causal_model.py#L176) 调用：

```python
roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
roped_key   = rope_apply(k, grid_sizes, freqs).type_as(v)
```

---

## 三、causal_rope_apply：推理时的关键差异

### 3.1 问题所在

推理时是逐帧生成的：第 7 帧生成时，只把第 7 帧的 token 送入模型（而不是 0-7 帧全部）。

如果用 `rope_apply`，它会对当前帧的 f 个 token 做 `freqs[0][:f]`，也就是用第 0~f-1 帧的频率旋转。  
但这帧实际上是视频的第 7 帧！应该用 `freqs[0][7]` 旋转，不然模型以为这是开头的帧。

这就是 `causal_rope_apply` 存在的原因：**加入 `start_frame` 偏移量，告诉 RoPE 当前这批帧在完整视频中的绝对时间位置。**

---

### 3.2 causal_rope_apply 完整讲解

代码（[causal_model.py:32-60](../wan/modules/causal_model.py#L32)）：

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # 和 rope_apply 一样，拆分 T/H/W 频率表

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            #         ^^^^^^^^^^^^^^^^^^^^^^^^^^
            #         关键！不再是 [:f]（从 0 开始）
            #         而是 [start_frame : start_frame+f]（从当前帧的绝对位置开始）
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # H 轴不变
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),  # W 轴不变
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)
```

唯一的改动就是这一行，其余完全相同：

```
# rope_apply（训练）:
freqs[0][:f]                      # 始终从位置 0 开始，用于表示"这 f 帧是 0 到 f-1 帧"

# causal_rope_apply（推理）:
freqs[0][start_frame:start_frame + f]  # 从 start_frame 开始，表示"这 f 帧是 start_frame 到 start_frame+f-1 帧"
```

---

### 3.3 start_frame 是怎么计算的

调用处（[causal_model.py:206-211](../wan/modules/causal_model.py#L206)）：

```python
# kv_cache 存在说明是推理模式
frame_seqlen = math.prod(grid_sizes[0][1:]).item()
# frame_seqlen = h * w = 30 * 52 = 1560，每帧的 token 数

current_start_frame = current_start // frame_seqlen
# current_start：当前这批 token 在整个视频 token 序列中的起始下标
# 除以每帧 token 数 = 当前是视频的第几帧

roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
roped_key   = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)
```

举例：正在生成视频的第 7 帧（3 帧一组，正在生成第 7~9 帧）：

```
current_start = 7 * 1560 = 10920  （第 7 帧从 token 10920 开始）
frame_seqlen  = 1560
current_start_frame = 10920 // 1560 = 7

# causal_rope_apply 会用：
freqs[0][7:10]  # 第 7、8、9 帧对应的 T 轴旋转频率
```

这样模型的注意力矩阵就能正确感知到：当前 query 在时间轴上是第 7~9 帧，历史 key 在第 0~6 帧，位置差是正确的相对距离。

---

## 四、对比总结

```
                    rope_apply（训练）          causal_rope_apply（推理）
─────────────────────────────────────────────────────────────────────────
输入                全部帧 [0..F-1]             当前帧 [start_frame..+f]
T 轴频率取法         freqs[0][:f]               freqs[0][start_frame:start_frame+f]
H/W 轴频率取法       freqs[1][:h], freqs[2][:w]  相同
调用位置             causal_model.py:176         causal_model.py:208-211
触发条件             kv_cache is None（训练）    kv_cache is not None（推理）
目的                对训练时整序列编码绝对位置  对推理时片段编码绝对位置（保持一致性）
```

---

## 五、一张图总结 3D RoPE 的数据流

```
视频 latent: [B, F, 16, H, W]
      ↓ patchify
token 序列: [B, F*H/p*W/p, dim]，每个 token 有 (t, h, w) 坐标
      ↓ 进入每个 CausalWanAttentionBlock
      ↓ 线性变换
Q, K: [B, seq_len, num_heads, head_dim]
      ↓ RoPE 旋转
      ┌─────────────────────────────────────────┐
      │ 训练（kv_cache=None）：                  │
      │   rope_apply(Q, grid_sizes, self.freqs)  │
      │   T轴从 freqs[0][0] 开始                 │
      └─────────────────────────────────────────┘
      ┌─────────────────────────────────────────┐
      │ 推理（kv_cache 存在）：                  │
      │   causal_rope_apply(Q, ..., start_frame) │
      │   T轴从 freqs[0][start_frame] 开始       │
      └─────────────────────────────────────────┘
      ↓
旋转后的 Q', K'（点积只依赖相对位置差）
      ↓
注意力权重 = softmax(Q' · K'^T / sqrt(d))
```

H/W 轴（空间位置）始终从 0 开始，因为每帧内的空间布局不变。  
只有 T 轴（时间位置）需要偏移，因为推理时每次只处理一小段帧。
