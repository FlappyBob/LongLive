# LongLive 五大创新：结合代码的深度讲解

> 目标读者：对 SF training 不熟悉，从源码层面理解算法，只有 inference 卡。
> 每个创新点都给出最关键的代码行数区间，并解释 PyTorch 语法细节。

---

## 创新一：帧级自回归生成（Causal Attention + KV Cache）

### 算法本质

传统 Wan2.1 把整段视频一次性喂进 Transformer，帧和帧之间做**双向注意力**（每帧能看到未来帧）。这意味着生成时必须先知道全部帧数，不能"一帧一帧流式生成"。

LongLive 的改法：每次只生成 1 帧（或几帧），每帧的注意力只能看**过去帧**（因果注意力）。过去帧的 Key/Value 矩阵被存在 KV Cache 里，不用重复计算。

```
传统（双向，一次性）:
  帧1 ←→ 帧2 ←→ 帧3 ←→ ... ←→ 帧100   全部同时处理

LongLive（因果，流式）:
  帧1 →  帧2 →  帧3 →  ...  →  帧100
  ↑      ↑      ↑
  只看过去，过去帧的 K/V 存在 cache 里
```

### 关键代码：RoPE 位置编码的偏移

**文件**：[wan/modules/causal_model.py:32-60](../wan/modules/causal_model.py)（`causal_rope_apply` 函数）

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    ...
    freqs_i = torch.cat([
        freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),  # ← 时间维
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),                            # ← 高度维
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),                            # ← 宽度维
    ], dim=-1)
```

**PyTorch 语法解释**：
- `freqs[0][start_frame:start_frame+f]` — 从频率表里切出"从第 start_frame 帧开始的 f 帧"的时间频率。这里 `start_frame` 就是当前帧在整个视频里的绝对位置（比如第 20 帧）
- `.view(f, 1, 1, -1)` — 重塑形状，`-1` 让 PyTorch 自动算这一维的大小
- `.expand(f, h, w, -1)` — 广播复制到 `[f, h, w, dim]`，不占额外内存（共享内存的视图）
- `torch.cat([...], dim=-1)` — 在最后一维拼接 T/H/W 三个方向的频率

**为什么 `start_frame` 如此重要**：

```
普通 rope_apply（训练全序列用）:   freqs[0:F]           — 总从第 0 帧算
causal_rope_apply（推理逐帧用）:   freqs[start_frame:F] — 从当前帧位置算
```

如果生成第 20 帧时位置编码从 0 开始，它的 Q 向量跟 Cache 里第 0 帧存的 K 向量的位置编码完全对不上——注意力分数全乱，生成的内容就会崩掉。

**调用位置**：[wan/modules/causal_model.py:208-211](../wan/modules/causal_model.py)

```python
# current_start_frame = 当前帧在整个序列里的位置（从0开始）
current_start_frame = current_start // frame_seqlen
roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
roped_key   = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)
```

### KV Cache 数据结构

**文件**：[pipeline/causal_inference.py:261-267](../pipeline/causal_inference.py)

```python
for _ in range(self.num_transformer_blocks):   # 30 个 Block，各自独立一份 cache
    kv_cache1.append({
        "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index":  torch.tensor([0], dtype=torch.long, device=device)
    })
```

**PyTorch 语法解释**：
- `torch.zeros([B, cache_size, 12, 128])` — 四维张量，`B`=批次，`cache_size`=缓存的 token 数，`12`=注意力头数（Wan 1.3B 的配置），`128`=每个头的维度
- `dtype=dtype, device=device` — 和输入数据保持相同精度（通常 bfloat16）和设备（GPU）
- `global_end_index`：全局已生成了多少 token，只增不减（哪怕 cache 满了也不回退）
- `local_end_index`：cache 数组里当前有效数据的末尾位置（cache 满后 roll 操作会改变这个值）

**为什么需要两个 index？**

```
生成 20 帧后，cache_size = 12 帧（局部注意力窗口）:

global_end_index = 20 × 1560 = 31200   ← 全局视角：已生成了 31200 个 token
local_end_index  = 12 × 1560 = 18720   ← cache 实际只存了最近 12 帧（18720 token）

global 用于 RoPE 偏移计算（必须知道绝对位置）
local  用于 cache 读写的数组下标（相对于 cache 数组）
```

### 推理主循环

**文件**：[pipeline/causal_inference.py:144-209](../pipeline/causal_inference.py)

```python
for current_num_frames in all_num_frames:             # 外层：逐帧块循环
    noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

    for index, current_timestep in enumerate(self.denoising_step_list):  # 内层：多步去噪
        _, denoised_pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache1,
            current_start=current_start_frame * self.frame_seq_length    # ← 绝对位置
        )
        if index < len(self.denoising_step_list) - 1:
            next_timestep = self.denoising_step_list[index + 1]
            noisy_input = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones([batch_size * current_num_frames], ...)
            ).unflatten(0, denoised_pred.shape[:2])
```

**PyTorch 语法解释**：
- `.flatten(0, 1)` — 把前两维合并，`[B, F, C, H, W]` → `[B×F, C, H, W]`，方便批量处理
- `.unflatten(0, denoised_pred.shape[:2])` — 还原，`[B×F, C, H, W]` → `[B, F, C, H, W]`
- `torch.randn_like(x)` — 生成与 x 形状相同的标准正态噪声
- `denoising_step_list = [1000, 750, 500, 250]`：每帧被去噪 4 次，从高噪到干净；不是最后一步时，把当前预测出的干净帧重新加噪（加到下一步对应的噪声级别），再输入下一步

---

## 创新二：Frame Sink + 滑动窗口注意力

### 算法本质

生成 1050 帧时，如果存全部历史 KV，需要 `1050 × 1560 × 12 × 128 × 2 bytes ≈ 51 GB`，H100 也放不下。

解决方案两部分：
1. **滑动窗口**：KV Cache 只保留最近 12 帧的历史，旧帧被驱逐
2. **Frame Sink**：前 3 帧永远不被驱逐，永久占据 cache 开头

```
KV Cache 物理布局（sink=3帧, window=12帧, 共15帧位置）:

┌─────────────────────┬────────────────────────────────────┐
│  Sink (帧0, 1, 2)   │     Sliding Window (最近12帧)       │
│  4680 tokens        │  18720 tokens                       │
│  永不被驱逐          │  每次新帧进来，最旧的非sink帧被挤出  │
└─────────────────────┴────────────────────────────────────┘

Sink 帧 = 视频"片头"，保存场景、风格、角色等全局信息
Window  = 最近发生了什么，保持短程连贯
```

### 关键代码一：判断是否需要滚动（roll）

**文件**：[wan/modules/causal_model.py:231-233](../wan/modules/causal_model.py)

```python
if self.local_attn_size != -1 \
        and (current_end > kv_cache["global_end_index"].item()) \
        and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
```

**三个条件的含义**：
1. `local_attn_size != -1`：用的是局部注意力（不是全局模式）
2. `current_end > global_end_index`：这是真正在生成新帧（而不是 gradient checkpointing 的重算）
3. `num_new_tokens + local_end_index > kv_cache_size`：新帧放不下了，需要腾地方

**PyTorch 语法解释**：
- `.item()` — 把只含一个数的 tensor 转成 Python 的普通整数，才能做 `>` 比较

### 关键代码二：滚动操作（左移驱逐旧帧）

**文件**：[wan/modules/causal_model.py:253-258](../wan/modules/causal_model.py)

```python
temp_k = kv_cache["k"].clone()          # ← 先克隆，不直接改原 cache！
temp_v = kv_cache["v"].clone()

# 把 sink 之后的内容向左移动（驱逐最旧的 num_evicted_tokens 个 token）
temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
    temp_k[:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
    temp_v[:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
```

**PyTorch 语法解释**：
- `kv_cache["k"].clone()` — **深拷贝**，创建一份完全独立的内存副本。后面的操作在副本上做，不改原 cache（原因：gradient checkpointing 重算时，原 cache 必须保持不变）
- `temp_k[:, A:B] = temp_k[:, C:D].clone()` — 把 C~D 位置的数据**复制**到 A~B 位置。右边必须 `.clone()` 才能避免左右在同一块内存时的数据污染（先写左边会破坏右边的数据）
- `[:, sink_tokens:...]` — 第一个 `:` 表示"所有 batch"，`sink_tokens` 之前的 sink 区域完全不碰

**ASCII 示意**（sink=3帧, window=4帧, 新来2帧）：

```
before: [S1 S2 S3 | A  B  C  D ]     ← cache 满了，A 是最旧的非 sink 帧
         ← sink →   ← window →

num_evicted = 2（A、B 被驱逐）
num_rolled  = 2（C、D 向左移）

步骤1 左移: [S1 S2 S3 | C  D  _  _ ]    （A、B 位置被覆盖丢弃）
步骤2 插入: [S1 S2 S3 | C  D  E  F ]    ✓ 新帧 E、F 写入空位
```

### 关键代码三：注意力计算时拼接 sink + window

**文件**：[wan/modules/causal_model.py:321-341](../wan/modules/causal_model.py)

```python
if sink_tokens > 0:
    local_budget = self.max_attention_size - sink_tokens
    k_sink  = temp_k[:, :sink_tokens]                                      # 前 3 帧
    v_sink  = temp_v[:, :sink_tokens]
    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
    k_local = temp_k[:, local_start_for_window : local_end_index]          # 最近 N 帧
    v_local = temp_v[:, local_start_for_window : local_end_index]
    k_cat   = torch.cat([k_sink, k_local], dim=1)                          # 拼起来
    v_cat   = torch.cat([v_sink, v_local], dim=1)
    x = attention(roped_query, k_cat, v_cat)
```

**PyTorch 语法解释**：
- `temp_k[:, :sink_tokens]` — 切出前 `sink_tokens` 个 token（前 3 帧 × 1560 = 4680 个 token）
- `torch.cat([k_sink, k_local], dim=1)` — 在序列长度维（dim=1）拼接，结果形状 `[B, sink+window, 12, 128]`
- `roped_query` 只有当前新帧的 Q（shape `[B, 1560, 12, 128]`），而 K/V 是 sink+window 的历史，attention 计算时新帧 Q 向所有历史 K/V 提问

---

## 创新三：Context Pass（KV Cache 写入干净帧）

这是**只读代码才能发现**的细节，文档里几乎不提。

### 核心问题

去噪循环运行时，每次 `generator(noisy_input, kv_cache=...)` 都在用**带噪声的帧**更新 cache。循环结束后，cache 里存的是"最后一次去噪步骤输入的噪声帧"的 K/V，而不是干净帧的 K/V。

下一帧生成时，它的历史上下文是带噪声的，这会让视频质量随帧数增加而退化。

### 解决：每帧生成后额外再跑一次 forward，只为写干净 K/V

**文件**：[pipeline/causal_inference.py:192-200](../pipeline/causal_inference.py)

```python
# 此时 denoised_pred 是干净的 clean 帧（去噪循环结束后的输出）
context_timestep = torch.ones_like(timestep) * self.args.context_noise  # context_noise = 0

self.generator(
    noisy_image_or_video=denoised_pred,      # ← 输入干净帧，不是噪声帧！
    conditional_dict=conditional_dict,
    timestep=context_timestep,               # ← t=0，告诉模型"这是干净数据"
    kv_cache=self.kv_cache1,                 # ← 这次 forward 会更新 cache
    current_start=current_start_frame * self.frame_seq_length,
)
# 这次的输出被丢弃——没有 "_, pred = ..."，只要 cache 更新这个副作用
```

**PyTorch 语法解释**：
- `torch.ones_like(timestep)` — 创建与 `timestep` 形状完全相同的全 1 张量，`* self.args.context_noise`（=0）后变全零
- 这次 `self.generator(...)` 的返回值没有被任何变量接收（没有左边的 `=`），这在 Python 里完全合法——纯粹为了触发 forward 内部的 cache 写入副作用
- `t=0` 时 `σ_t ≈ 0`，flow matching 的公式 `xt = (1-σt)×x0 + σt×noise` 退化为 `xt ≈ x0`，模型计算的 K/V 就代表干净帧的特征

---

## 创新四：KV-Recache（交互式 Prompt 切换）

### 核心问题

切换 prompt 后，KV Cache 里的**自注意力 K/V**（self-attention）是用旧 prompt 语境算出来的——虽然 self-attn 不直接依赖文本，但模型是用旧文本 embedding 训练的，旧帧的特征表示带有旧 prompt 的"痕迹"。

更直接的问题是**交叉注意力 cache**（cross-attention cache，存储文本 K/V）：

```python
# 文件: pipeline/causal_inference.py:271-282
crossattn_cache.append({
    "k": torch.zeros([batch_size, 512, 12, 128], ...),   # 512 = 文本 token 数
    "v": torch.zeros([batch_size, 512, 12, 128], ...),
    "is_init": False    # ← 首次 forward 时计算并缓存，之后直接复用
})
```

这个 cross-attn cache 存的是旧 prompt 的文本 K/V。切换 prompt 后它必须清空并重算。

### 解决：_recache_after_switch

**文件**：[pipeline/interactive_causal_inference.py:34-96](../pipeline/interactive_causal_inference.py)

```python
def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):

    # Step 1: 清空 cross-attention cache（文本侧 K/V）
    for blk in self.crossattn_cache:
        blk["k"].zero_()        # ← in-place 清零，不重新分配内存
        blk["v"].zero_()
        blk["is_init"] = False  # ← 标记为未初始化，下次 forward 会重新计算

    # Step 2: 用新 prompt 重跑历史帧，让 self-attn KV Cache 也刷新到新 prompt 语境
    num_recache_frames = current_start_frame if self.local_attn_size == -1 \
                         else min(self.local_attn_size, current_start_frame)
    recache_start_frame = current_start_frame - num_recache_frames

    frames_to_recache = output[:, recache_start_frame:current_start_frame]

    self.generator(
        noisy_image_or_video=frames_to_recache,    # 已生成的历史干净帧
        conditional_dict=new_conditional_dict,      # ← 新 prompt 的文本嵌入！
        timestep=context_noise * ones(...),         # t=0，以干净帧身份写入
        kv_cache=self.kv_cache1,
        current_start=recache_start_frame * self.frame_seq_length,
        sink_recache_after_switch=not self.global_sink,
    )
```

**PyTorch 语法解释**：
- `.zero_()` — **in-place 清零**，结尾的下划线是 PyTorch in-place 操作的约定，直接修改原 tensor 的内存，不创建新 tensor（比 `= torch.zeros(...)` 省显存，因为不用重新分配）
- `output[:, A:B]` — **切片不拷贝**，是原 tensor 的视图，与原数据共享内存。如果 `output` 在 CPU（low_memory 模式），第 58-62 行会 `.to(target_device)` 移到 GPU

### sink_recache_after_switch 标志的作用

**文件**：[wan/modules/causal_model.py:296-298](../wan/modules/causal_model.py)

```python
write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
if sink_recache_after_switch:
    write_start_index = local_start_index  # ← 绕过 sink 保护，允许覆盖 sink 区域
```

正常重算时（`is_recompute=True`）sink 区域受保护不被覆盖，但 recache 时需要用**新 prompt** 的语境重写 sink 帧的 K/V，所以需要绕过这个保护。

---

## 创新五：Streaming Long Tuning（训练侧，理解用）

> 你现在没有训练卡，但理解这个能帮你明白为什么模型在推理时能保持长序列一致性。

### 核心思想

不能一次性训练 240 帧（OOM），但如果每次只训 21 帧，模型学不会"当前帧应该和 200 帧前的内容保持一致"。

解决：**把 240 帧切成 21 帧的 chunk，chunk 之间共享同一个 KV Cache**。

```
流式训练示意:

  t=0  生成 chunk0（帧0~20）  → 写入 KV Cache  → 计算 DMD loss
       ↓  KV Cache 保留
  t=1  生成 chunk1（帧21~41）→ 读 Cache 里的帧0~20 → 写入 → 计算 loss
       ↓  KV Cache 保留
  t=2  生成 chunk2（帧42~62）→ 读 Cache → 写入 → 计算 loss（这里开梯度）
  ...
```

### 关键代码：KV Cache 跨 chunk 持续保留

**文件**：[pipeline/streaming_training.py:73-233](../pipeline/streaming_training.py)（`generate_chunk_with_cache`）

```python
def generate_chunk_with_cache(self, noise, conditional_dict, current_start_frame, requires_grad):
    # noise 只是 21 帧的噪声，但 self.kv_cache1 里已有前面所有 chunk 的历史
    ...
    for block_index, current_num_frames in enumerate(all_num_frames):
        for step_idx, current_timestep in enumerate(self.denoising_step_list):
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                kv_cache=self.kv_cache1,   # ← 同一个 cache 对象，跨 chunk 持续传递
                current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
            )
```

**关键点**：`self.kv_cache1` 是同一个 Python 对象，`generate_chunk_with_cache` 被调用多次，每次都在同一个 cache 上读写，不重置。这模拟了推理时的行为（推理时也是同一个 cache 一直累积）。

### 梯度只在最后几个 chunk 计算

```python
if not requires_grad:
    start_gradient_frame_index = chunk_frames   # 全程 no_grad（第一个 chunk）
else:
    start_gradient_frame_index = 0             # 全程开梯度（后续 chunk）
```

训练时 Trainer（[trainer/distillation.py:1096-1098](../trainer/distillation.py)）会先生成一个"warmup chunk"（`requires_grad=False`），再生成"训练 chunk"（`requires_grad=True`）。

---

## 附：你能直接在 inference 侧调整的参数

不改代码，只改 YAML，即可改变以下行为：

```yaml
# configs/longlive_inference.yaml
model_kwargs:
  local_attn_size: 12   # ← 滑动窗口帧数。减小 → 更省显存，更快；增大 → 更连贯但慢
                        #   -1 = 全局注意力（最连贯，最耗显存）
  sink_size: 3          # ← Frame Sink 帧数。0 = 不用 sink；3 = 保留前3帧

num_output_frames: 120  # ← 生成帧数（1帧 ≈ 0.067秒@15fps）

context_noise: 0        # ← context pass 的噪声级别。0=干净帧写入cache（推荐）
```

**代码生效位置**：
- `local_attn_size` → [causal_model.py:87-88](../wan/modules/causal_model.py) 决定 `max_attention_size`
- `sink_size` → [causal_model.py:214](../wan/modules/causal_model.py) 计算 `sink_tokens = sink_size × 1560`
- 二者共同影响 [causal_model.py:231-348](../wan/modules/causal_model.py) 整个 cache 管理逻辑
