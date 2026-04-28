# InteractiveCausalInferencePipeline：多 prompt 交互推理

> 核心文件：[pipeline/interactive_causal_inference.py](../pipeline/interactive_causal_inference.py)

---

## 一、继承关系

```
CausalInferencePipeline (pipeline/causal_inference.py)
    └── InteractiveCausalInferencePipeline
            ├── 继承所有单 prompt 推理能力
            ├── 新增 self.global_sink 参数
            └── 覆盖 inference() 方法，支持多 prompt + 切换
```

---

## 二、多 prompt 推理接口

```python
def inference(
    self,
    noise: Tensor,                          # [B, T, C, H, W]
    text_prompts_list: List[List[str]],     # N_segs × B 条 prompt
    switch_frame_indices: List[int],        # 长度 = N_segs - 1
    ...
):
```

**例：3 段 prompt，在第 30、60 帧切换**

```
text_prompts_list = [
    ["A cat running in the park"],           # segment 0: 帧 0~29
    ["The cat jumps onto a bench"],          # segment 1: 帧 30~59
    ["The cat rests under the tree"],        # segment 2: 帧 60~119
]
switch_frame_indices = [30, 60]

生成流程：
  帧 0~29:   用 segment 0 的条件生成
  帧 30:     检测到 current_start_frame(30) >= switch_frame_indices[0](30)
             → _recache_after_switch(output, 30, cond_list[1])
  帧 30~59:  用 segment 1 的条件继续生成
  帧 60:     → _recache_after_switch(output, 60, cond_list[2])
  帧 60~119: 用 segment 2 的条件继续生成
```

---

## 三、_recache_after_switch：KV-Recache 机制

文件：[interactive_causal_inference.py:34-96](../pipeline/interactive_causal_inference.py)

### 3.1 完整流程

```python
def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
    
    # Step 1: [if not global_sink] 清空 Self-Attn KV Cache
    if not self.global_sink:
        for block_idx in range(30):
            cache = self.kv_cache1[block_idx]
            cache["k"].zero_()    # in-place 清零，不重新分配内存
            cache["v"].zero_()
            # 注意：global_end_index 和 local_end_index 不清零！
            # 它们保留位置信息，确保后续写入位置正确
    
    # Step 2: 清空 Cross-Attn Cache（文本 KV，切换 prompt 必须重算）
    for blk in self.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False   # 标记为未初始化，下次 forward 重新计算
    
    # Step 3: 确定要 recache 的帧范围
    if current_start_frame == 0:
        return  # 第一帧不需要 recache
    
    num_recache_frames = (current_start_frame if local_attn_size == -1
                          else min(local_attn_size, current_start_frame))
    recache_start_frame = current_start_frame - num_recache_frames
    
    # Step 4: 取历史干净帧（从 output buffer）
    frames_to_recache = output[:, recache_start_frame:current_start_frame]
    if frames_to_recache.device.type == 'cpu':
        frames_to_recache = frames_to_recache.to(target_device)  # CPU→GPU
    
    # Step 5: 准备 BlockMask（重新生成因果掩码）
    block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
        device, num_frames=num_recache_frames, frame_seqlen=1560,
        num_frame_per_block=self.num_frame_per_block,
        local_attn_size=self.local_attn_size
    )
    self.generator.model.block_mask = block_mask
    
    # Step 6: 用新 prompt 重跑历史帧的 forward（写入新 K/V）
    context_timestep = ones([B, num_recache_frames]) * context_noise  # t=0
    with torch.no_grad():
        self.generator(
            noisy_image_or_video=frames_to_recache,    # 历史干净帧
            conditional_dict=new_conditional_dict,      # ← 新 prompt！
            timestep=context_timestep,                  # t=0
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=recache_start_frame * frame_seq_length,
            sink_recache_after_switch=(not self.global_sink),  # 是否覆盖 sink
        )
    
    # Step 7: 再次清空 Cross-Attn Cache（recache 结束后 is_init 变 True）
    for blk in self.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False
```

---

## 四、global_sink 参数的影响

| `global_sink` | KV-Recache 行为 | Sink 帧行为 |
|---------------|----------------|-------------|
| `True`（推理默认） | 不清空 KV Cache，只 recache window 帧 | Sink 帧 K/V 保持不变（全局锚点） |
| `False`（训练默认） | 清空整个 KV Cache，重新写入 | Sink 帧被新 prompt 的 K/V 覆盖 |

**为什么推理时 global_sink=True？**

Sink 帧（视频开头）是整个视频的视觉"锚"——主角长相、场景背景、画风。即使换了 prompt，这些视觉元素应该保持一致，否则会出现突兀的外观变化。`global_sink=True` 保护 sink 帧不被 recache 覆盖。

**为什么训练时 global_sink=False？**

训练时每个 batch 是独立的，不需要跨 batch 的视觉一致性。清空整个 cache 确保每次训练从干净状态开始。

---

## 五、为什么还要清空 Cross-Attn Cache？

Cross-Attn Cache 存储的是**文本 prompt 的 Key/Value**：

```python
# wan/modules/model.py（交叉注意力层）
if crossattn_cache is not None and crossattn_cache["is_init"]:
    k_text = crossattn_cache["k"]    # 直接复用缓存的文本 K/V
    v_text = crossattn_cache["v"]
else:
    k_text = self.k(context)         # 重新计算
    v_text = self.v(context)
    if crossattn_cache is not None:
        crossattn_cache["k"] = k_text
        crossattn_cache["v"] = v_text
        crossattn_cache["is_init"] = True
```

当 prompt 切换后，旧 prompt 的文本 K/V 不再正确，必须清零并重新计算。

`is_init=False` 会强制在下次 forward 时重新调用 `self.k(context)` 和 `self.v(context)`，用新 prompt 的嵌入计算。

---

## 六、recache 帧数的选择

```python
num_recache_frames = (current_start_frame if local_attn_size == -1
                      else min(local_attn_size, current_start_frame))
```

| 情况 | num_recache_frames | 含义 |
|------|-------------------|------|
| 全局注意力（-1） | = current_start_frame | 重建整个历史 |
| 局部注意力（12帧） | = min(12, current_start_frame) | 只重建最近 12 帧（滑动窗口范围） |

**为什么只 recache window 范围内的帧？**

模型在局部注意力模式下，每帧只能看到最近 N 帧。超出 window 的帧（如果 sink 没覆盖到的话）无论如何都不会被当前帧"看见"，所以 recache 它们没有意义。

---

## 七、prompt 切换的视觉连贯性

切换 prompt 后的视频应该：
1. **新 prompt 的语义**：下一帧开始按新 prompt 生成
2. **旧画面的外观延续**：主角还是同一个，不突然"传送"到新场景

LongLive 的做法：
- Self-Attn KV Cache 里存的是**视觉特征**（pixel 级别的样子），recache 时重跑历史帧可以保留视觉特征，但加入新 prompt 的语义引导
- Cross-Attn 切换到新 prompt，引导**后续**帧朝新方向走
- Sink 帧（global_sink=True）保持固定，提供全局视觉锚点

结果：下一帧的生成既有新 prompt 的指导，又有历史帧的视觉连续性。

---

## 八、interactive_inference.py 的配置差异

```yaml
# configs/longlive_interactive_inference.yaml
global_sink: true       # 推理时保护 sink
context_noise: 0        # t=0 写干净帧

# 例示配置（example/interactive_example.jsonl）
# 每行是一个 JSON：{ "prompts": [...], "switch_frames": [...] }
```

```python
# interactive_inference.py（入口文件）
# 读取 JSONL 格式的多 prompt 输入：
# {"prompts": ["prompt1", "prompt2"], "switch_frames": [30]}
```
