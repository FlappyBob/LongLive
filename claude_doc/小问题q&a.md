# 小问题 Q&A

> 阅读源码过程中产生的零散问题，逐条记录。

---

## Q13：为什么 Phase-2 用 LoRA 而不是全参微调？

**代码位置**：[configs/longlive_train_long.yaml:96-104](../configs/longlive_train_long.yaml#L96-L104)、[trainer/distillation.py:208-212, 1310-1312](../trainer/distillation.py#L208-L212)

### 一句话

**Phase-1 已经把 1.3B 全参练好了，Phase-2 只是在它之上学"长视频一致性"和"prompt 切换"，用 LoRA 增量微调既省显存又能保住已学能力。**

### 五个具体原因

**1. 防止灾难性遗忘**
Phase-1 的 700 步 DMD 蒸馏已经把 Wan1.3B 改造成 4 步 causal 生成器，这是一份精细的能力。Phase-2 只在 21→240 帧的扩展上做增量调整，全参 + 5× 大学习率（`lr=1e-5` vs `2e-6`）很容易把 Phase-1 学到的短片质量打飞。LoRA 把改动限制在低秩子空间，原权重整体冻结，主体能力安全。

**2. 显存预算根本不够**
Phase-2 同时塞下了：240 帧序列 + 跨 chunk KV cache `(12+21)*1560` tokens × 30 层 + 14B real_score + 1.3B fake_score（critic 也要 LoRA: `apply_to_critic=true`）+ FSDP 通信 buffer。全参 AdamW 的 optimizer state（参数本身的 4×：fp32 master + m + v + grad）会再吃几个 GB。LoRA r=256 的可训参数只占总参数零点几个百分点，optimizer state 从 ~10 GB 量级降到 ~0.3 GB。

**3. 大学习率才能学动 streaming 行为**
Phase-2 要让模型学会"跨 chunk 复用 KV cache"、"在 switch 帧重置 cross-attn 还能续上"这些新行为，需要 `lr=1e-5`（5× 于 Phase-1）。这种 LR 直接打全参会破坏稳定。LoRA 在低秩子空间里放大学习率风险小很多，等价于"在小流形上大胆走"。

**4. ckpt 体积 / 分发友好**
全参 ckpt ~2.5 GB，LoRA adapter 只有几十 MB（[distillation.py:746-756](../trainer/distillation.py#L746-L756) 走 `_gather_lora_state_dict` 分支只 dump `generator_lora` + `critic_lora`）。便于发布、版本管理，也便于推理时按需切换不同 LoRA。

**5. rank=256, α=256 已经接近全参表达力**
[longlive_train_long.yaml:99-100](../configs/longlive_train_long.yaml#L99-L100) `rank=256, alpha=256` → scaling = α/r = 1.0。常见 LoRA 是 r=8~64，这里直接拉到 256，target 注意力 q/k/v/o，**实际表达力已经非常接近全参微调**，但仍享受上述四点好处。可以理解为"框定了一个足够大的子空间，但限定从 0 出发的增量"。

### 副作用：EMA 关掉了

LoRA 模式下 EMA 被显式禁用（[distillation.py:1310-1312](../trainer/distillation.py#L1310-L1312)）：

```python
if self.is_lora_enabled:
    print("EMA creation skipped at step {self.step} (disabled in LoRA mode)")
```

原因：EMA 维护参数的滑动平均，对 LoRA 这种"小增量"意义不大；且 LoRA 自身的低秩约束已经提供了足够的训练稳定性，不再需要 EMA 平滑。

### 类比

> Phase-1 像把一辆轿车改装成赛车（全参，慢慢调每个零件）。
> Phase-2 像在赛车上贴空气动力学套件——不动核心机械，只在外面加可拆卸组件，让它能跑长直道（240 帧）和换赛道（prompt switch）。LoRA 就是这个"可拆卸套件"。

---

## Q12：current_end > global_end_index 这个条件不是恒为真吗？global_end_index 在哪里更新？

**代码位置**：`causal_model.py:231, 888`，初始化 `pipeline/causal_inference.py:265`

### 这个条件为什么不恒为真

它在**正向推进生成新帧**时才为真，在 **recompute** 场景下为假。

例：第 7 块（current_start=10920，current_end=15600）做 4 步去噪 + context pass：

```
第 1 步去噪（t=1000）forward:
  调用前 global_end_index = 10920
  current_end > global_end_index → 15600 > 10920 → True ✓
  执行后 global_end_index 被更新为 15600

第 2 步（t=750）forward:
  调用前 global_end_index = 15600
  current_end > global_end_index → 15600 > 15600 → False ✗
  is_recompute = True，走原位覆写分支

第 3、4 步 + context pass 同理，全部 False
```

只有每个 block 的**第 1 步去噪**才把 cache 推进到新位置，后续对同一段反复前向都是覆写，不该再 roll。

### global_end_index 在哪里更新

```python
# causal_model.py:885-889 _apply_cache_updates()
is_recompute = False if update_info is None else update_info.get("is_recompute", False)
if not is_recompute:
    kv_cache[block_index]["global_end_index"].fill_(current_end)
    kv_cache[block_index]["local_end_index"].fill_(local_end_index)
```

### 调用时机

attention forward 不直接动 cache，只返回 `cache_update_info`。
等 30 个 block 都算完后，`_forward_inference()` 统一调 `_apply_cache_updates()`：
- 真实写入 K/V（roll_and_insert 或 direct_insert）
- 更新两个 index（仅当 `is_recompute=False`）

### 关键设计：recompute 时 index 不前进

`if not is_recompute` 这个守卫是核心：
- 推进生成 → 更新 index → 下次 forward 自动判定为 recompute=False
- recompute → index 不动 → 4 步去噪 + context pass 始终在同一段位置覆写

### 初始化

```python
# pipeline/causal_inference.py:265
"global_end_index": torch.tensor([0], dtype=torch.long, device=device)
```

视频生成开始时两个 index 都是 0。第一块（current_start=0）调用时：
- `current_end > global_end_index` → 4680 > 0 → True
- `current_start > 0` → False
- `is_recompute = False`（即使后两步也是，因为 current_start=0）

视频开头几帧会被多次写入，但 sink 区允许覆盖（`is_recompute=False` 时不保护 sink）。

---

## Q11：grid_sizes[0][1:] 是干什么的？

**代码位置**：`causal_model.py:206`

```python
frame_seqlen = math.prod(grid_sizes[0][1:]).item()
```

### grid_sizes 是什么

`grid_sizes` 是一个 tensor，shape `[B, 3]`，每行存放该样本视频的潜空间网格尺寸 `(F, H, W)`：

```
grid_sizes = [[3, 30, 52]]   # B=1，当前块 3 帧，每帧 30×52 个 patch
              ↑   ↑   ↑
              F   H   W
              ↑   └───┘
              帧数 空间网格
```

来源：`CausalWanModel._forward_inference()` 里从 patch_embedding 后的潜变量推出来的，传给每个 attention block 用。

### grid_sizes[0][1:] 取的是什么

```python
grid_sizes[0]      # 第一个样本的 (F, H, W) → tensor([3, 30, 52])
grid_sizes[0][1:]  # 切掉第 0 维（帧数），只保留空间维 → tensor([30, 52])
```

切掉 F 是因为我们要算"**每帧**有多少个 token"，这只取决于空间尺寸 H、W，和帧数无关。

### math.prod 把它们乘起来

```python
math.prod(tensor([30, 52])) = 30 × 52 = 1560
```

得到 `frame_seqlen = 1560`，即每帧的 token 数（每帧被 patch 化成 30×52=1560 个 patch token）。

### 为什么需要这个变量

后续多处都要用：

| 用途 | 代码位置 | 作用 |
|------|----------|------|
| 算当前块绝对帧号 | `current_start // frame_seqlen` | token 下标 → 帧号 |
| 算 sink 占多少 token | `sink_size × frame_seqlen` | 帧数 → token 数 |
| 滑动窗口位置计算 | `roll_and_insert` 中多处 | 帧粒度 ↔ token 粒度互转 |

简而言之，`grid_sizes[0][1:]` 是用来**把空间维度提取出来求积**，得到"每帧 token 数"这个换算系数，让代码能在"帧数"和"token 数"两套坐标之间自由切换。

---

## Q10：3D RoPE 原理是什么？causal_rope_apply 每个变量是什么意思？

**代码位置**：`causal_model.py:32-60`，频率表初始化 `causal_model.py:612-617`

### RoPE 的目的

Attention 的点积 `Q·K` 只看语义相似度，不知道两个 token 之间的距离。RoPE 解法：把 Q 和 K 各旋转一个和位置有关的角度，旋转后 Q·K 自动只依赖位置差（相对位置），不依赖绝对位置。

### 为什么视频需要 3D

视频的每个 token 有三个坐标 `(t, h, w)`。把 head_dim=128（实数）配对成 64 个复数，劈成三段分别旋转：

```
T轴（时间）: 22 个复数 → 感知帧间距离
H轴（高度）: 21 个复数 → 感知行间距离
W轴（宽度）: 21 个复数 → 感知列间距离
22 + 21 + 21 = 64 ✓
```

`self.freqs.shape = [1024, 64]`（复数），预计算好的频率表，1024 是支持的最大位置数。

### causal_rope_apply 逐行讲解（以第 7 块为例，start_frame=7）

```python
# x = Q 或 K，shape [1, 4680, 12, 128]
n, c = x.size(2), x.size(3) // 2
# n=12（头数），c=64（每头的复数维度数）

freqs = freqs.split([c-2*(c//3), c//3, c//3], dim=1)
# 拆成 [22, 21, 21] → freqs[0]=T轴[1024,22], freqs[1]=H轴[1024,21], freqs[2]=W轴[1024,21]

# f=3, h=30, w=52, seq_len=4680
x_i = torch.view_as_complex(x[i,:seq_len].reshape(seq_len, n, -1, 2))
# [4680,12,128] → [4680,12,64,2] → [4680,12,64] 复数
# 相邻两个实数 (a,b) → 复数 a+ib

freqs_i = torch.cat([
    freqs[0][start_frame:start_frame+f]  # [7:10] → [3,22]
    .view(f,1,1,-1).expand(f,h,w,-1),    # [3,30,52,22]：第7帧全用freqs[0][7]旋转
    freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),  # [3,30,52,21]：H轴从0开始
    freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1),  # [3,30,52,21]：W轴从0开始
], dim=-1).reshape(seq_len, 1, -1)
# → [4680, 1, 64]，1 广播到 12 个头

x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
# 复数乘法=旋转 → [4680,12,64]复数 → [4680,12,128]实数
```

### 与 rope_apply 的唯一区别

```
rope_apply（训练）:     freqs[0][:f]                   从第 0 帧开始
causal_rope_apply（推理）: freqs[0][start_frame:start_frame+f]  从当前块实际帧位置开始
```

必须用 start_frame：KV Cache 里第 0~6 帧的 K 是用 `freqs[0][0:7]` 旋转的，第 7~9 帧的 Q 必须用 `freqs[0][7:10]` 旋转，Q·K 的点积才能正确反映"距离 N 帧"的相对位置信息。H/W 轴不需要偏移，因为每帧的空间布局相同。

---

## Q9：norm1(x) 是把 3 帧一起归一化吗？chunk 内 3 帧是独立去噪的吗？

**代码位置**：`causal_model.py:433`，`model.py:89`

### LayerNorm 归一化的是哪个维度？

`WanLayerNorm` 是 `nn.LayerNorm(dim=1536)`，归一化的是**最后一个维度（通道 C=1536）**，不是序列维度 L。

```
x: [B, 4680, 1536]
         ↑       ↑
      序列L    通道C ← LayerNorm 只归一化这里

每个 token 独立：mean = mean(x[b, l, :])  over 1536
                 std  = std (x[b, l, :])  over 1536
                 out  = (x[b, l, :] - mean) / std
```

4680 个 token 互不影响，Frame 0 的归一化不受 Frame 1 的数值影响。"3 帧一起 norm"并不存在。

unflatten 是纯 reshape，不改变数值，只是给后续 per-frame modulation 提供维度索引：

```
norm1(x)                           # [B, 4680, 1536]，4680 token 独立归一化
  .unflatten(1, (3, 1560))         # reshape → [B, 3, 1560, 1536]，无计算
  * (1 + e[1])                     # e[1]: [B, 3, 1, 1536]，frame 0/1/2 各自 scale
  + e[0]                           # e[0]: [B, 3, 1, 1536]，frame 0/1/2 各自 shift
  .flatten(1, 2)                   # 回到 [B, 4680, 1536]，送入 self-attn
```

### chunk 内 3 帧是独立去噪还是共同去噪？

同一 chunk 的 3 帧共享同一个时间步 t，e[0]~e[5] 对 3 帧实际上值相同（来自同一个 t 的嵌入）。3 帧在同一次 forward 里一起过 self-attn，block_mask 控制因果性（frame 1 能看 frame 0，frame 2 能看 0 和 1）。这样做的好处是同一块内帧间可以互相参考来保持视觉一致性，同时计算效率更高。

---

## Q8：nn.Parameter 和普通 randn 有什么区别？为什么 modulation 是可训练参数？

**代码位置**：`causal_model.py:399`

```python
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```

### randn 只是初始值，不是最终值

`torch.randn(...)` 只是**起点**，告诉模型"从随机噪声开始"。

如果写成普通 tensor：

```python
self.modulation = torch.randn(1, 6, dim) / dim**0.5   # 普通 tensor
```

这个值训练过程中**永远不会变**，因为它没有梯度，优化器不知道它的存在。

如果包成 `nn.Parameter`：

```python
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```

PyTorch 自动给它设置 `requires_grad=True`，并把它注册进 `model.parameters()`。每次反向传播，loss 对它求梯度，优化器（AdamW）就会更新它。

### 类比

想象你给 30 个员工（Block）各发了一份随机性格问卷（randn 初始化）。
- 如果问卷是"锁死的"（普通 tensor）：员工永远保持初始性格，不会学习
- 如果问卷是"可修改的"（nn.Parameter）：员工在工作（训练）中不断调整自己的性格，最终学会最适合自己岗位的响应方式

### 训练后 modulation 会学到什么

每个 Block 的 `modulation[0, 2, :]`（gate for self-attn）会从接近 0 慢慢学大，代表"我这一层的 self-attn 应该贡献多少"。靠近输入的 Block 和靠近输出的 Block 学到的值会不同。

这就是为什么叫"可训练参数"——`randn` 是初始化方式，训练才是最终定型的过程。

---

## Q7：self.modulation 是干什么的？为什么要加到 e 上再 chunk？

**代码位置**：`causal_model.py:399, 428`

```python
# __init__
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

# forward
e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
```

### 问题拆解

`e` 是从时间步嵌入算出来的，shape `[B, F, 6, 1536]`，代表"现在是去噪第几步"的全局信号，30 个 Block 共享同一个 `e`。

但这里在加入之前，先加了 `self.modulation`，它的 shape 是 `[1, 6, 1536]`（`unsqueeze(1)` 后变 `[1, 1, 6, 1536]`，广播到 `[B, F, 6, 1536]`）。

### modulation 的作用：每个 Block 的"个性偏置"

可以这样理解：

```
最终调制信号 = 全局时间信号(e) + 本Block专属基准(modulation)
```

`e` 告诉所有 Block"现在是 t=250"，但第 5 层 Block 和第 25 层 Block 各自想用**不同的力度**去响应这个信号。`self.modulation` 就是每个 Block 学出来的"个性偏置"——它不随时间步变化，是一个固定的可训练参数。

类比：30 个人都收到同一条消息"今天很热"，但每人对"热"的反应不同（有人开空调，有人去游泳）。`e` 是那条消息，`modulation` 是每个人的个性。

### 初始化为什么除以 dim**0.5

```python
torch.randn(1, 6, dim) / dim**0.5
# dim=1536 → dim**0.5 ≈ 39.2
# 初始值约在 [-0.025, +0.025] 之间，非常小
```

这里有一个训练技巧：`e[2]` 和 `e[5]` 是残差 gate，初始值接近 0 时：

```python
x = x + y * e[2]   # e[2] ≈ 0 → self-attn 输出几乎不加入
x = x + y * e[5]   # e[5] ≈ 0 → FFN 输出几乎不加入
```

训练开始时，模型相当于"跳过"了 self-attn 和 FFN，只走 cross-attn 和残差直连，极其稳定。随着训练进行，gate 慢慢学大，self-attn 和 FFN 的影响才逐渐引入。这叫 **zero-init gate**，是 DiT 论文的核心训练稳定技巧。

### chunk(6, dim=2) 做了什么

```python
e.shape before chunk: [B, F, 6, 1536]
.chunk(6, dim=2) → 6 个 tensor，每个 [B, F, 1, 1536]
```

把 6 个调制向量拆开，分别用作 shift/scale/gate，方便后续按名字取用（`e[0]` 到 `e[5]`）。

---

## Q6：DiT Block 原理是什么？CausalWanAttentionBlock.forward() 在做什么？

**代码位置**：`causal_model.py:401-460`

---

### 一句话理解

DiT Block = "先让视频 token 相互交流 → 再向文字提问 → 最后自己思考"，而且这三件事的**力度**都由当前去噪时间步 t 来实时调控。

---

### 整体结构

```
输入 x: [B, L, 1536]   （视频 token 序列，L=帧数×1560）
输入 e: [B, F, 6, 1536]（来自时间步 t 的调制信号，6 个向量）

                ┌────────────────────────────────┐
   x ──┐        │  AdaLN  (e[0], e[1])           │
       ▼        │  norm1(x)·(1+e[1]) + e[0]      │
   norm1 + mod  │  → Self-Attention (视频自注意力) │  ← 视频 token 互相看
       ▼        │  × e[2]（gate）                 │
   x = x + y   └────────────────────────────────┘
       │
       ▼        ┌────────────────────────────────┐
   norm3(x)     │  Cross-Attention               │  ← 向文字提问
       ▼        │  x 问 context（文字 token）     │
   x = x + y   └────────────────────────────────┘
       │
       ▼        ┌────────────────────────────────┐
   norm2 + mod  │  AdaLN  (e[3], e[4])           │
       ▼        │  norm2(x)·(1+e[4]) + e[3]      │
   FFN          │  → Linear→GELU→Linear           │  ← 自己整合信息
       ▼        │  × e[5]（gate）                 │
   x = x + y   └────────────────────────────────┘

输出 x: [B, L, 1536]   （同 shape，内容更新）
```

---

### 三个子模块各自干什么

| 子模块 | 在做什么 | 类比 |
|--------|----------|------|
| **Self-Attention** | 视频 token 之间互相交流，"我这一帧的这个 patch 和其他帧的 patch 有什么关系？" | 同学之间讨论 |
| **Cross-Attention** | 视频 token 去读文字 token（context），"我应该长得像文字描述的样子吗？" | 看参考书 |
| **FFN** | 每个 token 独立做两层 MLP，整合上面收集到的信息 | 独立思考 |

三个操作都是**残差连接**（`x = x + y`），意思是"在现有基础上加一点点修正"，而不是推翻重来。

---

### 核心技巧：AdaLN —— 让时间步 t 控制行为

普通 LayerNorm：`γ × norm(x) + β`，γ 和 β 是**固定参数**，训练完就不变了。

DiT 的 **Adaptive LayerNorm**：γ 和 β 是**实时算出来的**，来自时间步嵌入 e：

```python
# causal_model.py:433
norm1(x).unflatten(...) * (1 + e[1]) + e[0]
#                           ^^^^scale    ^^^shift
#                           动态 γ        动态 β
```

效果：
- **t=1000**（全噪声）→ e 给出一套 γ,β → 模型行为偏"画结构"
- **t=250**（接近干净）→ e 给出另一套 γ,β → 模型行为偏"精修细节"

不同去噪步，同一个 Block 的参数不变，但因为调制信号 e 不同，行为就不同。

---

### 6 个调制向量 e[0]~e[5] 分别管什么

```python
# causal_model.py:428
e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
# 把 [B, F, 6, 1536] 拆成 6 个 [B, F, 1, 1536]
```

| 编号 | 作用 | 位置 |
|------|------|------|
| e[0] | Self-Attn 之前：shift（偏移） | `causal_model.py:433` |
| e[1] | Self-Attn 之前：scale（缩放） | `causal_model.py:433` |
| e[2] | Self-Attn 之后：gate（残差强度） | `causal_model.py:444` |
| e[3] | FFN 之前：shift | `causal_model.py:452` |
| e[4] | FFN 之前：scale | `causal_model.py:452` |
| e[5] | FFN 之后：gate | `causal_model.py:455` |

Cross-Attention 没有 AdaLN（只有 norm3，且可能是 Identity），因为文字本身不随去噪步变化，不需要调制。

---

### gate 是什么意思？

```python
x = x + (y * e[2])    # causal_model.py:444
```

`e[2]` 是一个 1536 维向量，初始值接近 0（因为 `self.modulation` 初始化很小）。

效果：训练早期，`e[2] ≈ 0` → Self-Attn 的输出几乎不加入残差 → 模型先学 FFN 和 cross-attn，再逐渐"开放"自注意力的影响。这是一个训练稳定性技巧。

---

### 完整数据流（具体 shape）

```
x: [1, 4680, 1536]   （context pass，3帧，每帧1560 token）
e: [1, 3, 6, 1536]   （3帧，每帧独立的调制信号）

norm1(x) → [1, 4680, 1536]
unflatten → [1, 3, 1560, 1536]
× (1+e[1]) + e[0] → [1, 3, 1560, 1536]  （每帧用自己的 γ,β 调制）
flatten → [1, 4680, 1536]
→ Self-Attn → y: [1, 4680, 1536]
unflatten y → [1, 3, 1560, 1536]
× e[2] → [1, 3, 1560, 1536]             （每帧独立 gate）
flatten → [1, 4680, 1536]
x = x + ... → [1, 4680, 1536]           （残差加回去）
```

---

## Q5：时间步嵌入具体是什么感觉？形象解释

**代码位置**：`wan/modules/model.py:15`，调用处 `causal_model.py:961`

### 问题：模型怎么"感知"现在噪声有多强？

去噪过程中，模型每次收到的输入都是含噪图像，但它需要知道**现在是第几步**——是 t=1000（刚开始，全是噪声）还是 t=250（快结束，接近干净）？

不同步骤模型的行为完全不同：
- t=1000 → "我看到的都是噪声，我要画出大致结构"
- t=250  → "基本形状已经有了，我要精修细节"

这个信息只有一个标量整数 t，怎么告诉模型？

### 类比：时钟的多根指针

想象一个时钟，只有一根指针（比如只有时针），你只能知道大概几点，精度很低。但如果同时有时针+分针+秒针，三根指针合在一起，你能精确到秒。

sinusoidal embedding 的思路一样：用 **128 个振荡器**（频率从高到低），同时"读"这个时间步 t，每个振荡器给出自己的 cos 和 sin 值，拼成 256 维向量。

```
振荡器编号 j     频率                 作用
───────────────────────────────────────────────────────
j=0         freq=1.0          快速振荡，区分相邻步（t=999 vs t=1000）
j=21        freq=0.316        中速
j=64        freq=0.031        慢速，区分阶段（早期 vs 中期 vs 晚期）
j=127       freq=0.0001       极慢，粗粒度感知（"噪声多" vs "噪声少"）
```

### 具体数值示例

设 t=1000（第一去噪步，最大噪声）：

```
j=0:   sinusoid = 1000 × 1.0     = 1000.0  → cos(1000) ≈  0.56,  sin(1000) ≈ -0.83
j=32:  sinusoid = 1000 × 0.178   = 178.0   → cos(178)  ≈ -0.29,  sin(178)  ≈  0.96
j=64:  sinusoid = 1000 × 0.0316  = 31.6    → cos(31.6) ≈ -0.29,  sin(31.6) ≈ -0.96
j=127: sinusoid = 1000 × 0.000102= 0.102   → cos(0.1)  ≈  0.995, sin(0.1)  ≈  0.10
```

再看 t=250（最后一步，接近干净）：

```
j=0:   sinusoid = 250 × 1.0     = 250.0   → cos(250)  ≈ -0.80,  sin(250)  ≈ -0.60
j=127: sinusoid = 250 × 0.000102= 0.0255  → cos(0.025)≈  0.9997, sin(0.025)≈  0.025
```

对比 t=1000 和 t=250 的差异：
- **高频维度（j=0）**：cos 从 0.56 变到 -0.80，差异很大 ✓ 能区分
- **低频维度（j=127）**：cos 从 0.995 变到 0.9997，差异极小——这两个"都属于早期去噪步"，所以低频不去区分它们，合理 ✓

再看 t=1000 和 t=999（相邻步）：
- **高频维度（j=0）**：sinusoid 差 1.0 → cos/sin 有可见差异，能区分 ✓
- **低频维度（j=127）**：sinusoid 差 0.0001 → cos/sin 几乎一样，不区分 ✓（它们确实"差不多"）

### 256 维向量长什么样（示意图）

```
维度索引:  0   1   2  ...  127  128  129  ...  255
         [cos₀ cos₁ cos₂ ... cos₁₂₇|sin₀ sin₁ sin₂ ... sin₁₂₇]
          ↑高频，区分细节          ↑低频，区分阶段

t=1000:  [ 0.56  ?   ?  ...  0.995 | -0.83  ?  ...  0.10 ]
t=500:   [-0.44  ?   ?  ...  0.998 |  0.90  ?  ...  0.051]
t=250:   [-0.80  ?   ?  ...  0.9997|  -0.60 ?  ...  0.025]
t=0:     [ 1.0   1.0 1.0...  1.0   |  0.0   0.0...  0.0  ]
         ↑ t=0 时 sinusoid 全为 0，所以 cos=1, sin=0
```

### 后续：Linear 把 256 维投影到 1536 维

```python
e = self.time_embedding(sinusoidal_embedding_1d(256, t.flatten()))
# [3, 256] → Linear(256→1536) → SiLU → Linear(1536→1536) → [3, 1536]
```

这两层 Linear 是**可学习的**，训练时模型自己学会从 256 维向量里提取"现在该怎么去噪"的信息，映射到工作维度 1536。

最后 `e0 = time_projection(e)` 把它扩展到 6×1536，作为每个 Transformer 块的 AdaLN 调制系数（控制每层的 scale 和 shift），让整个网络的行为随时间步连续变化。

### 一句话总结

sinusoidal embedding 就是用 128 个不同速度的"指针"同时指向 t，快指针区分相邻步，慢指针区分大阶段，合在一起给模型一个丰富连续的时间感知——和你看钟表时同时看时针+分针+秒针一个道理。

---

## Q4：`sinusoidal_embedding_1d(256, t)` 内部怎么变的，为什么要这么做？

**代码位置**：`wan/modules/model.py:15`，调用处 `causal_model.py:961`

### 先说"为什么要做这步"

模型需要知道**当前去噪到哪一步了**（t=1000？还是 t=250？），
但 t 只是一个标量整数，直接喂给 Linear 层效果很差——
就像让模型直接读一个数字而不是读一段描述，信息太稀疏。

解决方案：把标量 t **展开成一个 256 维的向量**，
让这个向量里每一维都从不同的"频率视角"去描述 t 是多大。
这就是 sinusoidal embedding，和 Transformer 里的位置编码是同一个思路。

### 代码逐行追踪

```python
# wan/modules/model.py:15-25
def sinusoidal_embedding_1d(dim, position):
    # dim=256, position=t.flatten()=[1000, 1000, 1000]（3帧同一时间步）

    assert dim % 2 == 0
    half = dim // 2   # = 128

    position = position.type(torch.float64)
    # position.shape = [3]，值都是 1000（第一去噪步）

    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
```

分解 `torch.outer` 的两个参数：

```
参数一: position = [1000, 1000, 1000]  shape=[3]

参数二: torch.pow(10000, -arange(128)/128)
      = [10000^(-0/128), 10000^(-1/128), ..., 10000^(-127/128)]
      = [1.0,  0.9329,  0.8706,  ...,  0.0001]    shape=[128]
      ↑ 第0维频率最高（乘以1.0 → 快速变化）
                                     ↑ 第127维频率最低（乘以0.0001 → 缓慢变化）
```

`torch.outer([3], [128])` 就是外积，结果是 [3, 128]：

```
sinusoid[i, j] = position[i] × 10000^(-j/128)

对 position=1000：
  sinusoid[0, 0]   = 1000 × 1.0     = 1000.0
  sinusoid[0, 1]   = 1000 × 0.9329  = 932.9
  sinusoid[0, 63]  = 1000 × 0.0316  = 31.6
  sinusoid[0, 127] = 1000 × 0.0001  = 0.1
```

然后分别取 cos 和 sin，拼起来：

```python
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    # torch.cos(sinusoid): [3, 128]   ← 低频到高频的余弦
    # torch.sin(sinusoid): [3, 128]   ← 低频到高频的正弦
    # cat(dim=1): [3, 256]
    return x
```

### 直觉：为什么 cos + sin，而不只用 cos？

| 只用 cos | cos + sin |
|----------|-----------|
| cos(θ) 在 θ≈0 和 θ≈2π 时相等，两个不同的 t 可能映射到同一个值 | cos+sin 共同确定"相位"，等价于给出了复平面上的点 (cos θ, sin θ)，每个 θ 唯一 |

### 不同时间步产生的向量长啥样？

```
t=1000（纯噪声）:   向量里高频分量变化剧烈（1000 × 大系数 → sin/cos 快速转）
t=250（接近干净）:  向量里高频分量较平缓
t=0（干净帧）:      向量全为 [cos(0)...cos(0), sin(0)...sin(0)] = [1,1,...,1, 0,0,...,0]
```

相邻时间步（如 750 和 749）产生的向量**非常相近**，
远离的时间步（如 1000 和 0）产生的向量**差异很大**。
这样 Linear 层就能通过向量的连续变化来感知"当前噪声有多强"。

### 后续变换

```python
# causal_model.py:960-963
e = self.time_embedding(
    sinusoidal_embedding_1d(256, t.flatten()).type_as(x))
# sinusoidal_embedding_1d 输出: [3, 256]（3帧，每帧256维描述时间步）
# time_embedding = Linear(256→1536) → SiLU → Linear(1536→1536)
# e.shape = [3, 1536]
# 这步把"时间步的数值描述"投影到模型的工作维度，
# 让后续 AdaLN 可以用它来调制每一层的 LayerNorm
```

**总结一句话**：sinusoidal_embedding_1d 是把标量时间步 t 展开成 256 维向量，
用不同频率的 sin/cos 让模型能区分精细的时间步差异，
就像时钟的时针/分针/秒针同时读数——每根针负责一种精度。

---

## Q3：`crossattn_cache` 是用来存什么 K/V 的？

**代码位置**：`wan/modules/model.py:161`，`pipeline/causal_inference.py:271`

### 存的是文本的 K/V，不是视频的

交叉注意力里：
- **Q** 来自视频 token（每帧都不同）
- **K/V** 来自文本 embedding（prompt 全程不变）

既然 prompt 全程不变，每个 Transformer 块对文本做一次 `self.k(context)` 和 `self.v(context)` 的线性投影就够了——后续所有帧直接复用结果，不需要重复计算。`crossattn_cache` 就是存这个。

### 代码逻辑（`model.py:174`）

```python
if crossattn_cache is not None:
    if not crossattn_cache["is_init"]:      # 第一次 forward
        crossattn_cache["is_init"] = True
        k = self.norm_k(self.k(context)).view(b, -1, n, d)  # 计算文本 K
        v = self.v(context).view(b, -1, n, d)               # 计算文本 V
        crossattn_cache["k"] = k            # 存入 cache
        crossattn_cache["v"] = v
    else:                                   # 第二次及以后
        k = crossattn_cache["k"]            # 直接读 cache，跳过线性投影
        v = crossattn_cache["v"]
```

### cache 的形状

```python
# causal_inference.py:278-280
"k": torch.zeros([1, 512, 12, 128])   # [B, 文本token数, heads, head_dim]
"v": torch.zeros([1, 512, 12, 128])
"is_init": False
```

512 是文本序列长度（UMT5 固定输出 512 个 token），12 heads × 128 head_dim = 1536。

### 与自注意力 KV Cache 的区别

| | `kv_cache`（自注意力）| `crossattn_cache`（交叉注意力）|
|--|--|--|
| 存的是什么 | 视频历史帧的 K/V | 文本 prompt 的 K/V |
| 何时更新 | 每生成一帧都更新 | 只在第一次 forward 时写入，之后永远不变 |
| 形状 | `[1, 18720, 12, 128]` | `[1, 512, 12, 128]` |
| 驱逐/滚动 | 有（Frame Sink + Sliding Window）| 无 |

---

## Q2：`frame_seq_length = 1560` 里的 30×52 是怎么来的？

**代码位置**：`pipeline/causal_inference.py:41`

### 从像素到 token 的两次下采样

```
原始分辨率 480 × 832（像素）
        │
        │ ① VAE 编码：空间 8× 下采样
        ▼
潜变量  60 × 104（latent，每帧 16 个通道）
        │
        │ ② Patch Embedding：Conv3d kernel=(1,2,2), stride=(1,2,2)，空间再 2× 下采样
        ▼
Patch   30 × 52（每帧的 patch 网格）
```

所以 **每帧的 token 数 = 30 × 52 = 1560**，这就是 `frame_seq_length`。

### 第①步：VAE 编码（8× 下采样）

`WanVAEWrapper` 的 `_video_vae` 模型把像素压缩到潜变量：

```
像素: [B, 3, F, 480, 832]
 ↓ VAE Encoder（时间不变，空间 /8）
潜变量: [B, 16, F, 60, 104]
```

- 480 / 8 = **60**
- 832 / 8 = **104**

### 第②步：Patch Embedding（再 2× 下采样）

`CausalWanModel` 里（`causal_model.py:587`）：

```python
self.patch_embedding = nn.Conv3d(
    in_dim=16, out_dim=1536,
    kernel_size=(1, 2, 2),   # 时间 kernel=1（不合并帧），空间 kernel=2×2
    stride=(1, 2, 2)         # 时间 stride=1，空间 stride=2（再 2× 下采样）
)
```

```
潜变量: [B, 16, F, 60, 104]
 ↓ Conv3d stride=(1,2,2)
Patch:  [B, 1536, F, 30, 52]
```

- 60 / 2 = **30**
- 104 / 2 = **52**

### 汇总

| 阶段 | 空间尺寸 | 下采样倍数 |
|------|---------|---------|
| 原始像素 | 480 × 832 | — |
| VAE 潜变量 | 60 × 104 | 8× |
| Patch token | 30 × 52 | 2×（再） |
| **每帧 token 数** | **30 × 52 = 1560** | **总计 16×** |

所以 `kv_cache_size = local_attn_size × 1560 = 12 × 1560 = 18720`，
含义是"KV Cache 能容纳 12 帧 × 每帧 1560 个 token"。

---

## Q1：`pipeline.vae.model.clear_cache()` 是干什么的？

**代码位置**：`inference.py:222`，`wan/modules/vae.py:602`

### 背景：VAE 用了因果 3D 卷积

VAE 的 Decoder 内部全是 `CausalConv3d`（因果三维卷积）。这种卷积在时间维度上是"只看过去不看未来"的，但为了正确处理时间边界，它需要保存上一帧的输出作为下一帧的"左 padding"。

`decode()` 的实现逻辑是**逐帧循环**（`vae.py:555`）：

```python
for i in range(iter_):   # iter_ = 帧数（如 120）
    self._conv_idx = [0]
    if i == 0:
        out = self.decoder(x[:, :, 0:1], feat_cache=self._feat_map, feat_idx=self._conv_idx)
    else:
        out_ = self.decoder(x[:, :, i:i+1], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        out = torch.cat([out, out_], dim=2)
```

每处理完一帧，`_feat_map` 里就会留下 decoder 中**每一个 CausalConv3d 层**的最后几帧输出，供下一帧卷积时拼接到左边作为时间 padding：

```python
# vae.py:CausalConv3d.forward():
if cache_x is not None:
    x = torch.cat([cache_x, x], dim=2)   # 把上一帧拼到左边
    ...
feat_cache[idx] = cache_x   # 把当前帧存起来给下一帧用
```

### `clear_cache()` 做了什么（`vae.py:602`）

```python
def clear_cache(self):
    self._conv_num = count_conv3d(self.decoder)   # decoder 中 Conv3d 层的数量
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num      # 全部清空为 None
    # encode 侧同样处理
    self._enc_conv_num = count_conv3d(self.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num
```

**就是把 `_feat_map` 列表里每个槽位都重置为 `None`**，释放掉 GPU 上存的每一层的中间特征缓存。

### 调用时机

| 位置 | 时机 | 目的 |
|------|------|------|
| `decode()` 开头 `vae.py:546` | 开始解码前 | 确保没有上一次解码的残留 |
| `decode()` 结尾 `vae.py:568` | 解码结束后 | 立即释放 GPU 显存 |
| `inference.py:222` | 整个视频生成完后 | 确保显存完全释放（双保险）|

### 为什么在 `inference.py` 还要再调一次？

`decode()` 结尾已经调过了，但 `decode_to_pixel_chunk()` 里有分块解码的逻辑，中间可能 `clear_cache()` 没被正确触发。`inference.py:222` 是一个**显式的兜底清理**，确保视频生成完后 GPU 不残留任何 VAE 特征缓存。
