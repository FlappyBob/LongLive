# longlive_inference.yaml 参数说明

---

## 去噪调度

| 参数 | 值 | 作用 |
|------|-----|------|
| `denoising_step_list` | [1000, 750, 500, 250] | 每帧的多步去噪序列。每个数字是一个 timestep，帧从高噪（1000）逐步去噪到干净（250）。步数越多质量越好，越慢。 |
| `warp_denoising_step` | true | 把上面列表里的整数 index（0~999）映射成 scheduler 的真实 sigma 值。开启时列表里**不能**包含 0。 |

---

## 模型结构

| 参数 | 值 | 作用 |
|------|-----|------|
| `num_frame_per_block` | 3 | 每次生成几帧再更新 KV Cache。3 = 每 3 帧为一组依次生成。改小（如 1）更慢但理论上更准；改大更快但每帧获得的去噪步骤减少。 |
| `model_name` | Wan2.1-T2V-1.3B | 基础模型名，对应 `wan_models/` 目录下的权重文件夹。 |
| `model_kwargs.local_attn_size` | 12 | 滑动窗口大小（帧数）。每帧最多看过去 12 帧的历史（加 sink）。`-1` = 全局注意力（最连贯，最耗显存）。 |
| `model_kwargs.timestep_shift` | 5.0 | Flow Matching 的 sigma 曲线偏移参数，Wan 官方值，不要动。 |
| `model_kwargs.sink_size` | 3 | Frame Sink 帧数。前 3 帧永远留在 KV Cache 里不被驱逐，保持全局一致性。0 = 不用 sink。 |

---

## 推理控制

| 参数 | 值 | 作用 |
|------|-----|------|
| `data_path` | `.../vidprom_filtered_extended.txt` | 提示词文件路径，每行一个文本 prompt。 |
| `output_folder` | videos/long | 生成视频的保存目录。 |
| `inference_iter` | -1 | 跑几条 prompt。`-1` = 跑完整个文件。 |
| `num_output_frames` | 120 | 生成视频的总帧数。120 帧 ≈ 8 秒（@15fps）。增大直接变慢，同时 KV Cache 占用更多显存。 |
| `use_ema` | false | 是否用 EMA 权重推理（一般质量略好但需要保存了 EMA checkpoint）。 |
| `seed` | 0 | 随机种子，固定后同 prompt 每次生成结果相同。 |
| `num_samples` | 1 | 每个 prompt 生成几个视频。 |
| `save_with_index` | true | 文件名带编号后缀（`_0.mp4`, `_1.mp4`），避免覆盖。 |
| `global_sink` | true | 推理时 sink 帧是否作为全局上下文。`true` = sink 帧在 KV-Recache 时不会被清空（推理默认开，训练默认关）。 |
| `context_noise` | 0 | Context Pass 时加在干净帧上的噪声级别（timestep 值）。0 = 完全干净帧写入 cache。调大会让历史上下文略"模糊"，有时能改善风格一致性，但一般不动。 |

---

## 权重路径

| 参数 | 值 | 作用 |
|------|-----|------|
| `generator_ckpt` | `.../longlive_base.pt` | 阶段一训练产出的基础模型权重（完整参数，~2.5GB）。 |
| `lora_ckpt` | `.../lora.pt` | 阶段二训练产出的 LoRA 增量权重（仅 LoRA 矩阵，~几十 MB）。推理时叠加在 `generator_ckpt` 上。 |

---

## LoRA 配置

| 参数 | 值 | 作用 |
|------|-----|------|
| `adapter.type` | "lora" | 适配器类型，目前只支持 lora。 |
| `adapter.rank` | 256 | LoRA 秩。推理时必须与训练时一致，否则权重无法加载。 |
| `adapter.alpha` | 256 | LoRA 缩放系数。实际缩放比例 = `alpha/rank`，两个相等时缩放比为 1。 |
| `adapter.dropout` | 0.0 | 推理时 dropout 无效，固定为 0。 |
| `adapter.dtype` | "bfloat16" | LoRA 参数的数据类型，与主模型保持一致即可。 |
| `adapter.verbose` | false | 是否打印所有被 LoRA 替换的层名。调试时改 true。 |
