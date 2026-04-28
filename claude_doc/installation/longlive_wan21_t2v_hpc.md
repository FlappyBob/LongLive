# LongLive + Wan2.1-T2V-1.3B on BigPurple

## 目标

只围绕 `/gpfs/scratch/sy3535/code/LongLive` 这个 repo，按 LongLive 当前 README 的要求，在 BigPurple A100 机器上完成 `Wan2.1-T2V-1.3B` 推理，并跑出几条预览视频。

## 机器说明

- 登录节点: `bigpurple-ln2`
- GPU 节点: `a100_dev` 分区的 A100 80GB
- 本次实际推理节点: `a100-4002`
- 建议先在登录节点准备环境，再用 `srun` 申请 GPU 跑推理

## 目录约定

- repo: `/gpfs/scratch/sy3535/code/LongLive`
- conda env: `/gpfs/scratch/sy3535/code/LongLive/.condaenv`
- Wan 权重: `/gpfs/scratch/sy3535/code/LongLive/wan_models/Wan2.1-T2V-1.3B`
- LongLive 权重: `/gpfs/scratch/sy3535/code/LongLive/longlive_models`
- 预览 prompts: `/gpfs/scratch/sy3535/code/LongLive/claude_doc/installation/preview_prompts.txt`
- 预览 config: `/gpfs/scratch/sy3535/code/LongLive/claude_doc/installation/longlive_preview_inference.yaml`
- 输出视频: `/gpfs/scratch/sy3535/code/LongLive/videos/preview`

## 安装流程

### 1. 创建 repo 内环境

```bash
cd /gpfs/scratch/sy3535/code/LongLive
source /gpfs/scratch/sy3535/miniconda3/etc/profile.d/conda.sh
conda create -p /gpfs/scratch/sy3535/code/LongLive/.condaenv -y python=3.10
```

### 2. 安装 PyTorch

```bash
/gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/python -m pip install \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装 LongLive 依赖

README 里给的是:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

HPC 上直接跑 `pip install -r requirements.txt` 时，和推理无关的两个包先出问题:

- `nvidia-pyindex` 构建失败，报 `ModuleNotFoundError: No module named 'pip'`
- `pycuda` 构建失败，报 `cuda.h: No such file or directory`

因此这里采用了更稳的做法: 先装推理必需依赖，再单独装 `flash-attn`。

```bash
/gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/python -m pip install \
  diffusers==0.31.0 transformers==5.6.2 tokenizers==0.22.2 \
  accelerate>=1.1.1 tqdm datasets imageio easydict ftfy dashscope \
  imageio-ffmpeg wandb omegaconf einops av==13.1.0 \
  git+https://github.com/openai/CLIP.git open_clip_torch starlette \
  pycocotools lmdb matplotlib sentencepiece pydantic==2.10.6 scikit-image \
  "huggingface_hub[cli]==1.12.0" dominate peft
```

### 4. 安装 flash-attn

集群上需要显式给出 CUDA 路径和临时目录:

```bash
mkdir -p /gpfs/scratch/sy3535/code/LongLive/.tmp
mkdir -p /gpfs/scratch/sy3535/code/LongLive/.pip-cache

TMPDIR=/gpfs/scratch/sy3535/code/LongLive/.tmp \
PIP_CACHE_DIR=/gpfs/scratch/sy3535/code/LongLive/.pip-cache \
CUDA_HOME=/gpfs/share/apps/cuda/12.6 \
PATH=/gpfs/share/apps/cuda/12.6/bin:$PATH \
MAX_JOBS=4 \
TORCH_CUDA_ARCH_LIST=8.0 \
/gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/python -m pip install \
  flash-attn --no-build-isolation
```

### 5. 下载权重

```bash
HF_HUB_DISABLE_XET=1 \
/gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/huggingface-cli download \
  Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir /gpfs/scratch/sy3535/code/LongLive/wan_models/Wan2.1-T2V-1.3B

HF_HUB_DISABLE_XET=1 \
/gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/huggingface-cli download \
  Efficient-Large-Model/LongLive \
  --local-dir /gpfs/scratch/sy3535/code/LongLive/longlive_models
```

## 遇到的问题与解决

### 问题 1: flash-attn 已安装，但导入时 GLIBC 不兼容

现象:

```text
ImportError: /lib64/libc.so.6: version `GLIBC_2.32` not found
```

原因:

- wheel 能装上，但 A100 节点实际加载 `flash_attn_2_cuda` 时，系统 `glibc` 版本不够新

解决:

- 修改 `wan/modules/attention.py`
- 修改 `wan/utils/prompt_extend.py`
- 把原来只捕获 `ModuleNotFoundError` 的逻辑改成同时捕获 `ImportError`
- 当 flash-attn 不可用时，自动回退到 PyTorch `scaled_dot_product_attention`

这样做以后，即使 flash-attn 在当前节点不可用，LongLive 也还能正常推理，只是性能会比最佳状态保守一些。

### 问题 2: LoRA 权重加载时报 transformers / peft 兼容问题

现象:

```text
ImportError: cannot import name 'EmbeddingParallel' from transformers.integrations.tensor_parallel
```

原因:

- 初始安装的 `transformers==4.55.4` 和 `peft==0.19.1` 不匹配

解决:

- 升级到:
  - `transformers==5.6.2`
  - `tokenizers==0.22.2`
  - `huggingface_hub==1.12.0`

升级后，`longlive_models/models/lora.pt` 可以正常加载。

## 预览推理

预览配置:

- 分辨率: `832x480`
- 输出帧数: `24`
- 输出视频 fps: `16`
- LoRA: 开启
- prompt 文件: `claude_doc/installation/preview_prompts.txt`

推理命令:

```bash
cd /gpfs/scratch/sy3535/code/LongLive

srun -p a100_dev --gres=gpu:a100:1 -c 8 --mem=96G --time=01:00:00 \
  /gpfs/scratch/sy3535/code/LongLive/.condaenv/bin/torchrun \
  --nproc_per_node=1 --master_port=29500 \
  inference.py --config_path claude_doc/installation/longlive_preview_inference.yaml
```

## 输出结果

生成完成的文件在:

- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/rank0-0-0_lora.mp4`
- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/rank0-1-0_lora.mp4`
- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/rank0-2-0_lora.mp4`

为了便于查看，也复制了可读文件名版本:

- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/teacup_pour_lora.mp4`
- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/flying_tabby_lora.mp4`
- `/gpfs/scratch/sy3535/code/LongLive/videos/preview/neon_city_walk_lora.mp4`

这三条视频的实际信息:

- `832x480`
- `93` 帧
- `16 fps`
- 约 `5.81` 秒

## 本次保留的 repo 内文件

- `claude_doc/installation/preview_prompts.txt`
- `claude_doc/installation/longlive_preview_inference.yaml`
- `claude_doc/installation/longlive_wan21_t2v_hpc.md`

## 最短复现结论

在 BigPurple 上，LongLive + `Wan2.1-T2V-1.3B` 是可以跑通的。最关键的两个点不是显存，而是:

1. `flash-attn` 在节点上的 `GLIBC` 导入兼容
2. `peft` 和 `transformers` 的版本匹配

把这两个点处理好后，A100 80GB 上可以稳定跑出预览视频。
