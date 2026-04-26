# InteractiveCausalInferencePipeline — 多提示词交互推理

**文件**：[pipeline/interactive_causal_inference.py](../pipeline/interactive_causal_inference.py)

---

## 一、定位

`InteractiveCausalInferencePipeline` 继承 `CausalInferencePipeline`，在其基础上增加了**提示词中途切换**能力。

应用场景：在视频生成过程中，让视频内容从"猫咪跳跃"无缝过渡到"猫咪打盹"。

```
生成时序:
  帧 0~20  : prompt A "a cat jumping"
  帧 21~59 : prompt B "a cat sleeping"  ← 切换点 switch_frame=21
  帧 60~119: prompt C "a cat dreaming"  ← 再次切换 switch_frame=60
```

---

## 二、核心挑战：KV Cache 的 prompt 不一致性

切换 prompt 之后，KV Cache 中存储的是**旧 prompt** 下计算的 K/V（cross-attention 的文本嵌入是旧 prompt 的），新 prompt 的文本嵌入与旧 K/V 不兼容，会导致生成的内容无法平滑过渡。

**解决方案：KV-Recache**

```
切换前: cache 中是 prompt A 语境下的 K/V

切换后（_recache_after_switch）:
  1. 清空 cache（或保留 global_sink 的 frames）
  2. 用 prompt B 重新跑已生成的历史帧
  3. cache 现在存储的是 prompt B 语境下的历史 K/V
  4. 后续帧在 prompt B 语境下继续生成
```

---

## 三、_recache_after_switch() 详解

```python
def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
```

**两种模式**（由 `global_sink` 控制）：

#### 模式 1：`global_sink=False`（局部 recache，训练默认）

```
1. 清空 KV Cache 的 k/v（但不重置 global/local_end_index）
2. 清空 crossattn_cache（is_init=False）
3. 重新运行历史帧（最多 local_attn_size 帧）用新 prompt 填充 cache
   num_recache_frames = min(local_attn_size, current_start_frame)
   recache_start_frame = current_start_frame - num_recache_frames
```

#### 模式 2：`global_sink=True`（全局 recache，推理默认）

```
1. 不清空 KV Cache 的 k/v（保留 sink 帧的 K/V 不变）
2. 清空 crossattn_cache
3. 重新运行历史帧（同上）
```

**recache 的 forward 参数**：
```python
self.generator(
    noisy_image_or_video=frames_to_recache,  # 已生成的 clean 帧
    conditional_dict=new_conditional_dict,    # 新 prompt 的嵌入
    timestep = context_noise * ones(...),     # t ≈ 0（clean context）
    kv_cache=self.kv_cache1,
    current_start=recache_start_frame × 1560,
    sink_recache_after_switch=not self.global_sink   # 保护 sink 区域
)
```

---

## 四、inference() — 多段生成循环

```python
inference(
    noise: [B, T, 16, 60, 104],
    text_prompts_list: List[List[str]],   # N 段提示词
    switch_frame_indices: List[int],       # N-1 个切换帧索引
    return_latents=False,
    low_memory=False
)
```

示例调用：
```python
# 生成 60 帧视频，在第 20 帧切换 prompt
pipeline.inference(
    noise=noise,
    text_prompts_list=[["a cat jumping"], ["a cat sleeping"]],
    switch_frame_indices=[20]
)
```

**生成循环逻辑**：

```
初始化: segment_idx=0, next_switch_pos=switch_frame_indices[0]

for current_start_frame in [0, 1, ..., num_blocks-1]:

    # 检查是否到达切换点
    if current_start_frame >= next_switch_pos:
        segment_idx += 1
        _recache_after_switch(output, current_start_frame, cond_list[segment_idx])
        next_switch_pos = switch_frame_indices[segment_idx] 或 None

    cond_in_use = cond_list[segment_idx]

    [空间去噪循环（同父类）]

    [context pass（同父类）]
```

---

## 五、与父类的差异对比

| 特性 | CausalInferencePipeline | InteractiveCausalInferencePipeline |
|------|------------------------|-------------------------------------|
| 提示词数量 | 1 个 | N 个（multi-segment） |
| 切换点 | 无 | switch_frame_indices |
| KV-Recache | 无 | ✓ _recache_after_switch |
| global_sink | true（父类默认） | 可配置 |
| VAE 解码 | decode_to_pixel 或 chunk | 始终 decode_to_pixel |

---

## 六、SwitchCausalInferencePipeline

另一个子类 `SwitchCausalInferencePipeline`（[pipeline/switch_causal_inference.py](../pipeline/switch_causal_inference.py)）是简化版，只支持**单次切换**（2 段提示词），接口为：

```python
inference(noise, text_prompts_first, text_prompts_second, switch_frame_index)
```

训练时可视化（`_setup_visualizer`）使用这个类来监控模型是否学会了 prompt 切换能力。
