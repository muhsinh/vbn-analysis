#!/usr/bin/env python3
"""
Run SuperAnimal-quadruped on a single video file.

Usage:
    python run_superanimal.py <input_video.mp4> <output.h5>

Requirements:
    pip install "deeplabcut>=3.0.0rc1" --no-deps --prefer-binary
    pip install filterpy ruamel.yaml munkres dlclibrary
"""
import sys, os, subprocess
from pathlib import Path

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

VIDEO  = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
DEST   = OUTPUT.parent
DEST.mkdir(parents=True, exist_ok=True)

if not VIDEO.exists():
    sys.exit(f"ERROR: video not found: {VIDEO}")

# ── Patch DLC to use our detector batch size (default is 1, very slow) ───────
import deeplabcut as dlc
import torch

_dlc_base = dlc.__path__[0]

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    det_b   = max(4, min(48, int((free_gb - 2.0) / 0.4)))
    print(f"CUDA free VRAM: {free_gb:.1f} GB → detector_batch_size={det_b}")
else:
    det_b = 4
    print("No CUDA GPU — using CPU (will be slow)")

for _f in [
    f"{_dlc_base}/pose_estimation_pytorch/apis/videos.py",
    f"{_dlc_base}/pose_estimation_pytorch/apis/tracking_dataset.py",
]:
    if Path(_f).exists():
        subprocess.call([
            "sed", "-i",
            f's/get("detector_batch_size", [0-9]\\+)/get("detector_batch_size", {det_b})/g',
            _f,
        ])

# ── Run inference ─────────────────────────────────────────────────────────────
device   = "cuda" if torch.cuda.is_available() else "cpu"
pose_b   = 16
vtype    = "avi" if VIDEO.suffix.lower() == ".avi" else "mp4"

print(f"Input:  {VIDEO} ({VIDEO.stat().st_size/1024**3:.2f} GB)")
print(f"Output: {OUTPUT}")
print(f"Device: {device}, det_b={det_b}, pose_b={pose_b}")

dlc.video_inference_superanimal(
    videos=[str(VIDEO)],
    superanimal_name="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    scale_list=[200],
    videotype=vtype,
    dest_folder=str(DEST),
    create_labeled_video=False,
    plot_trajectories=False,
    batch_size=pose_b,
    detector_batch_size=det_b,
    device=device,
)

# ── Find and rename output ────────────────────────────────────────────────────
import shutil
h5s = sorted(DEST.glob(f"{VIDEO.stem}*superanimal*.h5"))
if not h5s:
    h5s = sorted(VIDEO.parent.glob(f"{VIDEO.stem}*superanimal*.h5"))
if not h5s:
    sys.exit(f"ERROR: no .h5 output found in {DEST}")

if str(h5s[0]) != str(OUTPUT):
    shutil.move(str(h5s[0]), str(OUTPUT))

print(f"\nDone → {OUTPUT} ({OUTPUT.stat().st_size/1024**2:.0f} MB)")
