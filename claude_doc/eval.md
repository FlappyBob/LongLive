# LongLive 复现实验与局限

目标：用本地 3090 Ti 24GB 跑 LongLive，理解长视频生成限制。论文主结果基于 H100：短视频 VBench 中 LongLive 20.7 FPS；30s VBench-Long 总分 83.52；交互 60s 用自建 160 条验证集；单 H100 展示可到 240s。

## 本地结果

| 实验 | 配置 | 结果 | 显存峰值 | 产物 |
|---|---|---:|---:|---|
| 12 latent frame | `claude_doc/eval_short_12.yaml` | 成功，136s | 24038MB | `videos/eval_short_12/rank0-0-0_lora.mp4`，45 帧，2.812s |
| 24 latent frame | `claude_doc/eval_short_24.yaml` | 成功，149s | 24200MB | `videos/eval_short_24/rank0-0-0_lora.mp4`，93 帧，5.812s |
| 60 latent frame | `claude_doc/eval_long_60.yaml` | 失败，VAE decode OOM | 24192MB | 无视频；剩 86MB 时还需 586MB |

完整表格记录在 `claude_doc/longlive_eval_records.csv`。Google Sheet 当前 403：`The caller does not have permission`，权限开后可直接同步 CSV。

## 自我修正

1. benchmark wrapper 里 `/usr/bin/time` 不存在，改用 bash 时间戳。
2. `peft 0.19.1` 与 `torchao 0.7.0` 冲突；升级到 `torchao 0.17.0` 又不兼容 `torch 2.5.1`。最终卸载 `torchao`，让 PEFT 跳过 torchao dispatcher，LoRA 推理跑通。
3. 60 latent frame 不是扩散阶段先崩，而是 VAE decode OOM。代码默认整段 decode：`utils/wan_wrapper.py:96-117`；已有 chunk decode 函数：`utils/wan_wrapper.py:119-150`，但主推理只在 `use_infinite_attention` 时走 chunk：`pipeline/causal_inference.py:219-223`。

## 方法局限

1. **硬件依赖强。** 论文的实时 20.7 FPS 和 240s 是 H100 条件；本地 24GB 卡 24 latent frame 已到 24200MB，60 latent frame 在 VAE decode OOM。`inference.py:61-63` 还强制 `low_memory=True`，说明普通显存环境本来就吃紧。

2. **端到端速度离实时远。** 本地 12/24 latent frame 的完整 wall FPS 约 0.33/0.62；去掉加载后约 2.65/2.91 FPS，仍远低于论文 H100 的 20.7 FPS。大权重加载也重：Wan 约 17GB，LongLive 约 8.2GB。

3. **长上下文靠窗口和 sink 折中。** 配置固定 `local_attn_size: 12`、`sink_size: 3`：`configs/longlive_inference.yaml:9-12`。代码里 KV cache 大小只按窗口算：`pipeline/causal_inference.py:109-127`。这带来速度/显存优势，但远距离细节仍可能丢失；论文也承认窗口越小一致性越弱，frame sink 是补救，不是完整记忆。

4. **交互是离散 prompt switch，不是真正任意编辑。** `interactive_inference.py:144-159` 要预先给 `switch_frame_indices`，配置示例也是固定切点：`configs/longlive_interactive_inference.yaml:20-27`。实时交互更像按帧段换 prompt，不是可随时局部修改物体、轨迹或物理状态。

5. **KV recache 改善切换，但不是免费。** 切换时会用新 prompt 重建近期 cache：`pipeline/interactive_causal_inference.py:34-90`。论文说 10s 单切换约多 6% 时间；多次切换、低端 GPU、长上下文下成本会累积。

6. **训练复现成本高。** 训练配置使用 60s/240 latent 长 rollout 设定：`configs/longlive_train_long.yaml:87-89`。论文称完整长视频 tuning 约 32 GPU-days，另有 64 H100 约 12 小时实现细节；本地单卡只能复现推理和小样本，不能等价复现训练。

7. **评测不能只看能跑。** 论文用 VBench、VBench-Long、CLIP 分段语义和用户研究；本地只跑了资源 benchmark，没有跑完整官方 prompt suite。结论应限定为“本地可运行性/显存边界”，不是质量排名复现。

## 复现命令

```bash
.venv/bin/python inference.py --config_path claude_doc/eval_short_12.yaml
.venv/bin/python inference.py --config_path claude_doc/eval_short_24.yaml
.venv/bin/python inference.py --config_path claude_doc/eval_long_60.yaml
```
