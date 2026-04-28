# Sink-Size 对照实验脚本使用说明

> 脚本位置：[experiments/sink_size_sweep.py](../experiments/sink_size_sweep.py)
> 示例 prompts：[experiments/prompts_sample.txt](../experiments/prompts_sample.txt)

---

## 一、脚本做什么

在 `local_attn_size = 12`（即 KV 滑窗为 12 个 latent 帧）的前提下，对照不同 `sink_size` 在长视频生成中的稳定性。

```
固定：local_attn_size = 12, num_frame_per_block = 3, fps = 16
扫描：sink_size ∈ {0, 3, 6, 9}    (= 0% / 25% / 50% / 75% × 12)
视频：每条 prompt × 每个 sink_size 各生成一段 120s 视频
抽帧：在 30s / 60s / 90s / 120s 各抽一帧
拼图：每条 prompt 一张 4×4 grid PNG
       行 = sink_size，列 = 秒数
```

---

## 二、产物目录

```
experiments/out_sink_sweep/
├── prompts.txt              # 实际使用的 prompts（脚本从输入复制过来）
├── _config_sink0.yaml       # 4 份临时 config（基于 *_infinity.yaml 改写）
├── _config_sink3.yaml
├── _config_sink6.yaml
├── _config_sink9.yaml
├── sink0/                   # 各 sink_size 独立输出文件夹（避免覆盖）
│   └── rank0-{prompt_idx}-0_lora.mp4
├── sink3/
├── sink6/
├── sink9/
├── frames/                  # ffmpeg 抽出的 PNG（按 prompt × sink × 秒命名）
├── grid_prompt0.png         # 4×4 拼图
├── grid_prompt0.txt         # 对应 prompt 文本（便于回溯）
├── grid_prompt1.png
└── ...
```

---

## 三、参数对账表

| 配置项 | 取值 | 来源 |
|---|---|---|
| `base_config` | `configs/longlive_inference_infinity.yaml` | 长视频必须 `use_infinite_attention=true` |
| `num_output_frames` | `480` (latent) | `120s × 16fps / 4 (VAE stride)` 且 `% num_frame_per_block == 0` |
| `local_attn_size` | `12` | 来自 base config |
| `num_frame_per_block` | `3` | 来自 base config |
| `sink_size` | `0 / 3 / 6 / 9` | 由脚本逐次覆写 |
| `master_port` | `29500 + sink_size` | 避免端口冲突 |
| `output_folder` | `out_sink_sweep/sink{N}/` | 避免不同 sink 视频互相覆盖 |

帧数换算：

```
pixel_frames = seconds × fps
latent_frames = round(pixel_frames / temporal_stride)   # Wan2.1 stride = 4
然后向上取整到 num_frame_per_block 的倍数
```

| seconds | pixel | latent | 实际秒 |
|---|---|---|---|
| 30 | 480 | 120 | 30.00 |
| 60 | 960 | 240 | 60.00 |
| 90 | 1440 | 360 | 90.00 |
| 120 | 1920 | 480 | 120.00 |

---

## 四、用法

### 4.1 干跑（不依赖 torch / 推理权重，只验证配置生成）

```bash
python3 experiments/sink_size_sweep.py --dry_run
```

会生成 4 份 yaml 并打印将执行的 `torchrun` 命令，不会真正跑推理。适合在没装依赖的机器上检查参数。

### 4.2 完整跑

```bash
# 用默认 prompts (experiments/prompts_sample.txt 共 3 条)
python3 experiments/sink_size_sweep.py

# 用自定义 prompts 文件（每行一条）
python3 experiments/sink_size_sweep.py --prompts_file my_prompts.txt

# 单条 prompt 快速试跑
python3 experiments/sink_size_sweep.py --prompt "a calm zen garden with falling cherry blossoms"
```

### 4.3 调整扫描维度

```bash
# 自定义 sink_size 列表
python3 experiments/sink_size_sweep.py --sink_sizes 0 6 12

# 自定义抽帧时间点
python3 experiments/sink_size_sweep.py --seconds 15 45 75 105

# 改视频长度（自动重算 num_output_frames）
python3 experiments/sink_size_sweep.py --video_seconds 60
```

### 4.4 推理已经跑过，只重建 grid

```bash
python3 experiments/sink_size_sweep.py --skip_inference
```

会跳过 `torchrun`，直接从 `out_sink_sweep/sink{N}/` 找 mp4，重抽帧、重拼图。改 cell 尺寸或抽帧时间点后非常方便。

---

## 五、依赖

| 依赖 | 用途 | 检查 |
|---|---|---|
| `torch` / `torchrun` / 项目本身 | 跑 `inference.py` | `torchrun --help` |
| `omegaconf`（推荐）或 `PyYAML`（fallback） | 写临时 config | `python3 -c "import omegaconf"` |
| `ffmpeg` / `ffprobe` | 抽帧 | `ffmpeg -version` |
| `Pillow` | 拼 grid | `python3 -c "import PIL"` |
| 模型权重 | 推理 | `longlive_models/models/{longlive_base,lora}.pt` |

---

## 六、常见问题

**Q: 视频文件名匹配不到？**
A: `inference.py` 的命名规则为 `rank{rank}-{idx}-{seed}_{type}.mp4`，其中 `type` ∈ `{regular, ema, lora}`。脚本用 glob `rank*-{prompt_idx}-0_*.mp4`，所以三种类型都能匹配；如果改了 `num_samples` 或多卡运行，需要按需扩展。

**Q: 想换长度但保留同样的对照？**
A: 直接改 `--video_seconds`；脚本会重算 `num_output_frames` 并向上取整到 `num_frame_per_block` 的倍数。注意抽帧时间点 `--seconds` 要相应调整，否则会被自动 clip 到视频末尾。

**Q: 单卡 120s × 4 sink × N prompts 太慢？**
A: 先用 `--prompt "..."` 跑单条试通；之后可以分多次 `--sink_sizes 0` / `--sink_sizes 3` 单跑，结果落到同一个 `out_sink_sweep/sink{N}/`，最后用 `--skip_inference` 一次性拼图。
