"""
Sink-size sweep: 同一 prompt 下，对照 local_attn_size=12 时不同 sink_size 的长视频效果。

实验设计：
  - 行：sink_size ∈ {0, 3, 6, 9}  (= 0/25/50/75% × 12-frame KV window)
  - 列：抽帧时间 ∈ {30s, 60s, 90s, 120s}
  - 每行一个独立推理：同 prompt × 不同 sink_size，输出 120s 视频
  - 每个 prompt 生成一张 4×4 grid PNG

工作流：
  1) 读 prompts 文件 (默认 ./prompts_sample.txt，每行一个 prompt)
  2) 对每个 sink_size 写一份临时 yaml（基于 longlive_inference_infinity.yaml），
     覆盖 sink_size / num_output_frames / output_folder / data_path。
  3) 用 torchrun 跑 inference.py，复用 repo 现有入口。
  4) 用 ffmpeg 在 30/60/90/120s 抽帧。
  5) 用 PIL 拼成 4×4 grid，每个 prompt 一张 PNG。

注意：
  - 默认 base config 用 *_infinity.yaml，因为 120s 长视频需要 use_infinite_attention=true。
  - num_output_frames 是 *latent* 帧；Wan VAE 的 temporal stride=4，
    所以 120s @ fps=16 → 1920 pixel → 480 latent；并向上取整到 num_frame_per_block 的倍数。
  - 视频文件名形如 rank0-{prompt_idx}-0_lora.mp4 (LoRA 启用时)；脚本会用 glob 匹配。
  - 不同 sink_size 用不同 master_port 避免端口冲突；分别落到 out_dir/sink{N}/ 子目录。

用法示例：
  # 干跑 (写 config / 列出会执行的命令，不真跑推理):
  python experiments/sink_size_sweep.py --dry_run

  # 跑示例 prompts:
  python experiments/sink_size_sweep.py --prompts_file experiments/prompts_sample.txt

  # 单条 prompt：
  python experiments/sink_size_sweep.py --prompt "a cinematic shot of ..."

  # 跳过推理，只用已有 mp4 重建 grid：
  python experiments/sink_size_sweep.py --prompts_file experiments/prompts_sample.txt --skip_inference
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_file", type=str, default=None,
                   help="文本文件，每行一个 prompt。和 --prompt 二选一。")
    p.add_argument("--prompt", type=str, default=None, help="单条 prompt（快速试跑用）。")
    p.add_argument("--sink_sizes", type=int, nargs="+", default=[0, 3, 6, 9],
                   help="要扫描的 sink_size 列表（默认 0 3 6 9）。")
    p.add_argument("--seconds", type=float, nargs="+", default=[30, 60, 90, 120],
                   help="抽帧时间点（秒）。")
    p.add_argument("--video_seconds", type=float, default=120.0,
                   help="每段视频长度（秒）。默认 120s。")
    p.add_argument("--fps", type=int, default=16, help="保存视频帧率。inference.py 写死 16，一般不要改。")
    p.add_argument("--num_frame_per_block", type=int, default=3,
                   help="num_output_frames 必须是它的倍数。和 base_config 一致。")
    p.add_argument("--temporal_stride", type=int, default=4,
                   help="VAE 时间下采样倍率。Wan2.1 = 4。")
    p.add_argument("--base_config", type=str,
                   default=str(REPO_ROOT / "configs" / "longlive_inference_infinity.yaml"),
                   help="基础 yaml；脚本会复制并覆盖 sink_size / num_output_frames / paths。")
    p.add_argument("--output_dir", type=str,
                   default=str(REPO_ROOT / "experiments" / "out_sink_sweep"),
                   help="所有产物（config / 视频 / grid）都落在这里。")
    p.add_argument("--master_port", type=int, default=29500,
                   help="torchrun master_port 起始值；每个 sink 用 base+sink 避免冲突。")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_inference", action="store_true",
                   help="跳过 torchrun 推理，只用已有 mp4 重建 grid。")
    p.add_argument("--dry_run", action="store_true",
                   help="只写 config 与打印命令，不真正跑推理或抽帧。")
    p.add_argument("--cell_w", type=int, default=480, help="grid 单元格宽度（像素）。")
    p.add_argument("--cell_h", type=int, default=270, help="grid 单元格高度（像素）。")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def latent_frames_for(seconds: float, fps: int, num_frame_per_block: int, temporal_stride: int) -> int:
    """Pixel frames = seconds*fps；除以 VAE temporal_stride 得 latent 帧数；向上取整到块倍数。"""
    pixel_frames = round(seconds * fps)
    latent = max(num_frame_per_block, round(pixel_frames / temporal_stride))
    if latent % num_frame_per_block != 0:
        latent = ((latent // num_frame_per_block) + 1) * num_frame_per_block
    return latent


def write_temp_config(base_config_path: Path, *, sink_size: int, num_output_frames: int,
                      data_path: Path, output_folder: Path, seed: int, save_to: Path) -> None:
    """优先用 omegaconf（保 yaml 风格一致）；缺时回退到 PyYAML。"""
    overrides = {
        "num_output_frames": int(num_output_frames),
        "data_path": str(data_path),
        "output_folder": str(output_folder),
        "save_with_index": True,
        "inference_iter": -1,
        "seed": int(seed),
    }
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(base_config_path))
        if "model_kwargs" not in cfg or cfg.model_kwargs is None:
            cfg.model_kwargs = {}
        cfg.model_kwargs.sink_size = int(sink_size)
        for k, v in overrides.items():
            cfg[k] = v
        OmegaConf.save(cfg, str(save_to))
        return
    except ImportError:
        pass
    import yaml
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model_kwargs", {})
    cfg["model_kwargs"]["sink_size"] = int(sink_size)
    cfg.update(overrides)
    with open(save_to, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_inference(config_path: Path, master_port: int, dry_run: bool) -> None:
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        f"--master_port={master_port}",
        str(REPO_ROOT / "inference.py"),
        "--config_path", str(config_path),
    ]
    print("[run]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def find_video_for(output_folder: Path, prompt_idx: int, seed_idx: int = 0) -> Path:
    # filename: rank{rank}-{idx}-{seed}_{type}.mp4，其中 type ∈ {regular, ema, lora}
    pattern = f"rank*-{prompt_idx}-{seed_idx}_*.mp4"
    candidates = sorted(output_folder.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No video matching {pattern!r} in {output_folder}. "
            f"是不是 inference 没跑成功，或者 prompt_idx 越界了？"
        )
    return candidates[0]


def extract_frame_with_ffmpeg(video_path: Path, second: float, out_png: Path) -> None:
    """用 ffmpeg 在指定时间点抽一帧。如果 second 超过视频长度，抽最后一帧。"""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # 先获取视频时长，clip 到 [0, duration - 1/fps]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = second  # fallback
    ts = max(0.0, min(second, max(0.0, duration - 0.05)))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{ts:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(out_png),
    ]
    subprocess.run(cmd, check=True)


def build_grid(images_2d: List[List["PIL.Image.Image"]],  # noqa: F821
               row_labels: List[str], col_labels: List[str],
               cell_w: int, cell_h: int) -> "PIL.Image.Image":  # noqa: F821
    from PIL import Image, ImageDraw, ImageFont
    rows = len(images_2d)
    cols = len(images_2d[0])
    margin_top, margin_left = 60, 160
    W = margin_left + cols * cell_w
    H = margin_top + rows * cell_h
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 22)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    for j, c in enumerate(col_labels):
        x = margin_left + j * cell_w + cell_w // 2
        draw.text((x, margin_top // 2), c, fill="white", anchor="mm", font=font)
    for i, row in enumerate(images_2d):
        y = margin_top + i * cell_h + cell_h // 2
        draw.text((margin_left // 2, y), row_labels[i], fill="white", anchor="mm", font=font)
        for j, im in enumerate(row):
            canvas.paste(im.resize((cell_w, cell_h)), (margin_left + j * cell_w, margin_top + i * cell_h))
    return canvas


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 准备 prompts ----
    if args.prompts_file:
        prompts = [l.strip() for l in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        sample = REPO_ROOT / "experiments" / "prompts_sample.txt"
        if not sample.exists():
            print(f"[error] 没指定 --prompts_file/--prompt，且 {sample} 不存在", file=sys.stderr)
            return 2
        prompts = [l.strip() for l in sample.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[info] 用默认 prompts 文件 {sample} ({len(prompts)} 条)")

    prompts_file = out_dir / "prompts.txt"
    prompts_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    num_output_frames = latent_frames_for(
        args.video_seconds, args.fps, args.num_frame_per_block, args.temporal_stride
    )
    actual_seconds = num_output_frames * args.temporal_stride / args.fps
    print(f"[info] video_seconds={args.video_seconds:g} → "
          f"num_output_frames(latent)={num_output_frames} (≈{actual_seconds:.2f}s @ fps={args.fps})")
    print(f"[info] sink_sizes={args.sink_sizes}, sample_seconds={args.seconds}")
    print(f"[info] base_config={args.base_config}")

    # 抽帧时间点不能超过实际视频长度，自动 clip
    sample_seconds = [min(s, actual_seconds - 1.0 / args.fps) for s in args.seconds]
    if sample_seconds != list(args.seconds):
        print(f"[warn] 抽帧时间点超过视频长度，已 clip 为 {sample_seconds}")

    # ---- 2. 推理：每个 sink_size 一次 ----
    sink_dirs = {}
    for sink in args.sink_sizes:
        sub = out_dir / f"sink{sink}"
        sub.mkdir(parents=True, exist_ok=True)
        sink_dirs[sink] = sub
        cfg_path = out_dir / f"_config_sink{sink}.yaml"
        if not args.skip_inference:
            write_temp_config(
                base_config_path=Path(args.base_config),
                sink_size=sink,
                num_output_frames=num_output_frames,
                data_path=prompts_file,
                output_folder=sub,
                seed=args.seed,
                save_to=cfg_path,
            )
            print(f"[info] wrote {cfg_path} (sink_size={sink}, output→{sub})")
            run_inference(cfg_path, master_port=args.master_port + sink, dry_run=args.dry_run)

    if args.dry_run:
        print("[dry_run] 推理已跳过；不抽帧、不拼图。配置文件已生成可检查。")
        return 0

    # ---- 3. 抽帧 + 4. 拼 grid ----
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[error] 需要 PIL/Pillow 来拼 grid: pip install pillow", file=sys.stderr)
        return 3
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[error] 需要系统里有 ffmpeg / ffprobe", file=sys.stderr)
        return 4

    from PIL import Image
    for p_idx, prompt in enumerate(prompts):
        rows: List[List[Image.Image]] = []
        for sink in args.sink_sizes:
            try:
                video = find_video_for(sink_dirs[sink], p_idx)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                rows = []
                break
            row_imgs = []
            for s in sample_seconds:
                png = out_dir / "frames" / f"prompt{p_idx}_sink{sink}_t{int(round(s)):03d}s.png"
                extract_frame_with_ffmpeg(video, s, png)
                row_imgs.append(Image.open(png).convert("RGB"))
            rows.append(row_imgs)
        if not rows:
            print(f"[skip] prompt {p_idx} 缺视频，跳过 grid")
            continue
        col_labels = [f"{int(round(s))}s" for s in sample_seconds]
        row_labels = [f"sink={s}" for s in args.sink_sizes]
        grid = build_grid(rows, row_labels, col_labels, args.cell_w, args.cell_h)
        grid_path = out_dir / f"grid_prompt{p_idx}.png"
        grid.save(grid_path)
        # 同时写一份带 prompt 的说明
        (out_dir / f"grid_prompt{p_idx}.txt").write_text(prompt + "\n", encoding="utf-8")
        print(f"[saved] {grid_path}  ({prompt[:60]}...)")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
