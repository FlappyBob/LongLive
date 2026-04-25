# LongLive 复现速查

论文：`arXiv:2509.22622`。LongLive 是 frame-level AR 长视频生成：KV-recache 处理交互换 prompt，streaming long tuning 对齐长视频训练/推理，short window attention + frame sink 保持长程一致性。官方测试：A100/H100、40GB+ VRAM、64GB RAM；本机 RTX 3090 Ti 24GB 可验环境，完整长视频可能 OOM。

## 环境

本机已装：Python 3.10.20，torch 2.5.1+cu121，torchvision 0.20.1+cu121，flash-attn 2.8.3；未安装 torchao。驱动 535.288.01 只适合 cu121；官方 `torch 2.8.0+cu128` 需要更新驱动。

```bash
cd /workspace/LongLive
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install pip
uv pip install --torch-backend cu121 "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
uv pip install -r <(grep -vE '^(nvidia-pyindex|nvidia-tensorrt|pycuda|torchao)($|[[:space:]])' requirements.txt) tensorboard packaging
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=2 uv pip install flash-attn --no-build-isolation
```

修正记录：

- `nvidia-pyindex` 构建失败：它服务 TensorRT/PyCUDA，源码入口不直接依赖，基础复现先跳过。
- `peft 0.19.1` 会拒绝 `torchao 0.7.0`；`torchao 0.17.0` 又不兼容 torch 2.5，会报 `torch.int1`。普通 LoRA 推理直接卸载 torchao；INT8 量化需另配新版 torch 栈。
- `huggingface-cli` 已废弃；用 `hf download`。
- `one_logger_utils` 不是公开 PyPI 包；已补本地 no-op `one_logger_utils.py`，训练时建议加 `--disable-wandb --no-one-logger`。

检查：

```bash
python - <<'PY'
import torch
from torchvision.io import write_video
from wan.modules import attention
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("write_video", callable(write_video), "fa2", attention.FLASH_ATTN_2_AVAILABLE, "fa3", attention.FLASH_ATTN_3_AVAILABLE)
PY
python inference.py --help
python interactive_inference.py --help
python train.py --help
```

## 下载

会用 Hugging Face；如限速或 gated，先 `hf auth login`。

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Efficient-Large-Model/LongLive --local-dir longlive_models
```

训练再下：

```bash
hf download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
```

默认配置需要：

```bash
test -f wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
test -f wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
test -f longlive_models/models/longlive_base.pt
test -f longlive_models/models/lora.pt
```

## 使用

单 prompt 长视频：

```bash
source .venv/bin/activate
bash inference.sh
```

交互多 prompt：

```bash
bash interactive_inference.sh
```

无限长配置：

```bash
torchrun --nproc_per_node=1 --master_port=29500 inference.py --config_path configs/longlive_inference_infinity.yaml
```

训练顺序：

```bash
torchrun --nproc_per_node=8 train.py --config_path configs/longlive_train_init.yaml --logdir logs --disable-wandb --no-one-logger
torchrun --nproc_per_node=8 train.py --config_path configs/longlive_train_long.yaml --logdir logs --disable-wandb --no-one-logger
```

提示词：每段都重复主体和场景锚点；适合长镜头里的动作/物体/风格变化，不适合快速分镜硬切。

来源：`https://arxiv.org/abs/2509.22622`，`https://nvlabs.github.io/LongLive/docs/`，`https://github.com/NVlabs/LongLive`。
