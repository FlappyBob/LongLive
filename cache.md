我的背景：

1. 我不懂SF training，对training和inference都没有工程细节上面的理解，内心的mindmap很模糊。所以我只能从源代码层面来尝试理解。
2. 我不懂pytorch很多语法逻辑，你要细致的讲，不要跳过繁琐的语法，对于其中的优化你要细讲。
3. 跳过繁琐的点火，代码架构，根据核心创新点直接讲算法的部分。
4. 目前我只有能支持inference部分的卡，现在只能从inference有关的架构做出改变（我猜是casual attention sink的部分）

你要：

1. 详细讲一下longlive的创新之处。
2. 讲创新之处的时候要结合代码，点睛一样的指出最关键部分的代码，给出代码引用行数区间。

请你结合代码，详细讲一下longlive的创新之处，我列举在下面了，写在claude_doc里面
来源：https://deepwiki.com/NVlabs/LongLive/1-longlive-overview

### 小问题
1. 3D ROPE是怎么加入到causalwan里面的，原理是什么？你先给我讲清楚rope原理是什么？；是干什么的，在哪里加；之后再讲casual_rope是干什么的，在哪里加。答案写入claude_doc里面


## LongLive: Key Innovations

### 1. Frame-level Autoregressive Video Generation

LongLive reformulates video generation as a frame-level autoregressive process.  
Instead of generating the whole video with bidirectional attention, each frame only attends to previously generated frames.

**Why it matters:**

- Enables causal attention for streaming video generation.
- Supports KV caching to avoid recomputing past frames.
- Makes real-time generation possible, reaching around **20.7 FPS on a single H100**.
- Allows long videos to be generated incrementally rather than all at once.

---

### 2. KV-Recache for Interactive Prompt Switching

LongLive introduces a KV-recache mechanism for interactive generation.  
When the user changes the prompt during generation, the system refreshes the KV cache using the new text condition.

**Why it matters:**

- Allows the video to follow new prompts mid-generation.
- Maintains visual continuity across prompt changes.
- Avoids abrupt scene breaks when switching narrative segments.
- Enables interactive multi-prompt long video generation.

---

### 3. Streaming Long Tuning

LongLive does not train on full long videos in one pass.  
Instead, it trains chunk by chunk while reusing historical KV cache from previous chunks.

**Why it matters:**

- Aligns training with inference: **train-long-test-long**.
- Reduces memory cost by processing long videos in short chunks.
- Uses teacher distillation to preserve quality.
- Enables minute-long generation with relatively efficient training.

---

### 4. Frame Sink + Short Window Attention

LongLive combines local sliding-window attention with a frame-sink mechanism.  
The first few frames are permanently kept in the KV cache, while recent frames are handled with a short attention window.

**Why it matters:**

- Reduces attention memory from quadratic growth to bounded local attention.
- Preserves long-range consistency through the initial sink frames.
- Prevents the model from forgetting the beginning of the video.
- Makes generation of very long videos, such as **1050 frames / 240 seconds**, feasible.

---

### 5. Efficient Long-Video System Design

LongLive integrates causal attention, KV caching, frame sink, LoRA tuning, and quantized inference into one practical system.

**Why it matters:**

- Supports real-time long video generation.
- Supports interactive multi-segment generation.
- Can fine-tune from short-clip models to long-video generation.
- Supports INT8 / FP8-style efficient inference with small quality loss.

---

## claude_doc 文档导航

以下文档由 Claude 通过完整阅读源码生成，适合重写前的系统理解。

| 文档 | 内容 |
|------|------|
| [00_overview.md](claude_doc/00_overview.md) | 名词/动词/引擎/点火钥匙、模块结构、完整 Call Graph |
| [01_causal_model.md](claude_doc/01_causal_model.md) | CausalWanModel、CausalWanSelfAttention、BlockMask、RoPE |
| [02_kv_cache_attention.md](claude_doc/02_kv_cache_attention.md) | KV Cache 数据结构、roll_and_insert、sink、recompute 保护 |
| [03_wan_wrapper.md](claude_doc/03_wan_wrapper.md) | WanDiffusionWrapper、flow_pred↔x0 转换、TextEncoder、VAE |
| [04_causal_inference_pipeline.md](claude_doc/04_causal_inference_pipeline.md) | 推理主循环、空间/时序去噪、context pass |
| [05_interactive_inference.md](claude_doc/05_interactive_inference.md) | 多 prompt 切换推理、KV-Recache 机制 |
| [06_training_pipelines.md](claude_doc/06_training_pipelines.md) | SelfForcing 训练、Streaming 训练、exit_flags 梯度控制 |
| [07_dmd_model.md](claude_doc/07_dmd_model.md) | DMD 损失原理、generator_loss、critic_loss、DMDSwitch |
| [08_trainer.md](claude_doc/08_trainer.md) | Trainer 初始化、训练主循环、LoRA 配置、checkpoint 管理 |
| [09_configs_and_entry.md](claude_doc/09_configs_and_entry.md) | 两阶段配置对比、入口文件、数据格式 |
| [10_memory_and_distributed.md](claude_doc/10_memory_and_distributed.md) | FSDP、KV Cache 显存估算、梯度检查点兼容性、EMA |
| [11_innovations_deep_dive.md](claude_doc/11_innovations_deep_dive.md) | **五大创新逐行代码讲解**：因果 RoPE、KV Cache、Frame Sink roll 操作、Context Pass、KV-Recache |
| [13_rope_explained.md](claude_doc/13_rope_explained.md) | RoPE 原理从零讲起、3D RoPE 频率表构造、rope_apply vs causal_rope_apply 对比（start_frame 偏移机制）|
| [14_inference_walkthrough.md](claude_doc/14_inference_walkthrough.md) | **推理流程全程导读（重写版）**：每行代码带 Shape 注释，以 longlive_inference.yaml 具体配置为例，覆盖完整调用链 inference.py → WanDiffusionWrapper → CausalWanModel → KV Cache → VAE |
| [15_self_attn_kv_cache_walkthrough.md](claude_doc/15_self_attn_kv_cache_walkthrough.md) | **CausalWanSelfAttention 推理路径详解**：3D RoPE 应用 + cache 坐标计算 + is_recompute 判断（recompute vs 推进生成）+ 延迟更新 cache_update_info 设计 |
| [16_self_attn_full_walkthrough.md](claude_doc/16_self_attn_full_walkthrough.md) | **CausalWanSelfAttention.forward() 全程逐行导读**：97-358 行完整讲解，覆盖训练分支（flex_attention + block_mask）和推理分支（causal_rope + roll_and_insert + sink + 滑动窗口 attention），含训练 vs 推理对照表 |
