#!/usr/bin/env python3
"""
Download side.mp4 for VBN session 1055240613 and run SuperAnimal-quadruped inference.

Usage:
    python run_superanimal.py [output.h5]

    output.h5 defaults to ./session_1055240613_superanimal.h5

Install deps first:
    pip install "deeplabcut>=3.0.0rc1" --no-deps --prefer-binary boto3
    pip install filterpy ruamel.yaml munkres dlclibrary
"""

SESSION_ID  = 1055240613
S3_BUCKET   = "allen-brain-observatory"
S3_KEY      = f"visual-behavior-neuropixels/raw-data/{SESSION_ID}/behavior_videos/side.mp4"
MIN_SIZE_GB = 2.0  # anything smaller than this = truncated/failed download

import sys
import os
import re
import json
import shutil
import tempfile
import subprocess
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


# ── Python version ────────────────────────────────────────────────────────────

if sys.version_info < (3, 9):
    die(f"Python 3.9+ required, got {sys.version.split()[0]}")


# ── Dependency checks (clear messages, not tracebacks) ───────────────────────

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError:
    die("boto3 not installed.\n  Fix: pip install boto3")

try:
    import torch
except ImportError:
    die("PyTorch not installed. Install from https://pytorch.org then re-run.")

try:
    import deeplabcut as dlc
    print(f"DLC {dlc.__version__} OK")
except ImportError:
    die(
        "deeplabcut not installed.\n"
        '  Fix: pip install "deeplabcut>=3.0.0rc1" --no-deps --prefer-binary\n'
        "       pip install filterpy ruamel.yaml munkres dlclibrary"
    )
except Exception as e:
    die(f"deeplabcut imported but failed to initialise: {type(e).__name__}: {e}")


# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
         else Path(f"session_{SESSION_ID}_superanimal.h5").resolve()
VIDEO  = OUTPUT.parent / "side.mp4"

try:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
except PermissionError:
    die(f"Cannot create output directory: {OUTPUT.parent}")


# ── Download side.mp4 ─────────────────────────────────────────────────────────

def download_video(dest: Path) -> None:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Get expected size from S3 metadata
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
        expected_bytes: int | None = meta["ContentLength"]
        print(f"S3 object size: {expected_bytes / 1024**3:.2f} GB")
    except Exception as e:
        warn(f"Could not read S3 object size: {e}")
        expected_bytes = None

    # Skip if already fully downloaded
    if dest.exists():
        actual = dest.stat().st_size
        if expected_bytes and actual == expected_bytes:
            print(f"side.mp4 already complete ({actual / 1024**3:.2f} GB) — skipping download")
            return
        reason = (
            f"only {actual / 1024**2:.0f} MB (too small)"
            if actual < MIN_SIZE_GB * 1024**3
            else f"{actual / 1024**3:.2f} GB (expected {expected_bytes / 1024**3:.2f} GB)"
            if expected_bytes
            else f"{actual / 1024**3:.2f} GB present but expected size unknown"
        )
        warn(f"Existing side.mp4 is {reason} — re-downloading")
        dest.unlink()

    size_str = f"~{expected_bytes / 1024**3:.1f} GB" if expected_bytes else "unknown size"
    print(f"Downloading side.mp4 from S3 ({size_str}) ...")

    tmp = dest.with_suffix(".download.tmp")
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        body = obj["Body"]
        total = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = body.read(8 * 1024 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if expected_bytes:
                    pct = 100 * total / expected_bytes
                    print(f"\r  {total / 1024**3:.2f} / {expected_bytes / 1024**3:.2f} GB  ({pct:.0f}%)",
                          end="", flush=True)
                else:
                    print(f"\r  {total / 1024**3:.2f} GB downloaded", end="", flush=True)
        print()  # newline after progress bar

    except KeyboardInterrupt:
        tmp.unlink(missing_ok=True)
        die("Download interrupted — partial file removed. Re-run to resume.")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        die(f"Download failed: {type(e).__name__}: {e}")

    # Verify completed download
    actual = tmp.stat().st_size
    if actual < MIN_SIZE_GB * 1024**3:
        tmp.unlink(missing_ok=True)
        die(f"Download produced only {actual / 1024**2:.0f} MB — likely a network error. Re-run.")
    if expected_bytes and actual != expected_bytes:
        tmp.unlink(missing_ok=True)
        die(f"Download size mismatch: got {actual / 1024**3:.2f} GB, expected {expected_bytes / 1024**3:.2f} GB. Re-run.")

    tmp.rename(dest)
    print(f"Download complete: {dest} ({actual / 1024**3:.2f} GB)")


download_video(VIDEO)


# ── GPU inventory ─────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}"
              f"  free={free_b / 1024**3:.1f} / {total_b / 1024**3:.1f} GB")
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    det_b   = max(4, min(64, int((free_gb - 2.0) / 0.4)))
    device  = "cuda"
    print(f"Detector batch size: {det_b}  (from {free_gb:.1f} GB free VRAM on cuda:0)")
else:
    warn("No CUDA GPU detected — running on CPU. Expect many hours of runtime.")
    det_b  = 4
    device = "cpu"

pose_b = 16


# ── Patch DLC detector batch size ─────────────────────────────────────────────
# DLC 3.x does config.get("detector_batch_size", 1) internally, ignoring the
# kwarg passed to video_inference_superanimal. Patch the source with Python
# regex so it works cross-platform (no sed dependency).
# Inference runs in a subprocess so the patch is picked up on fresh import.

def patch_dlc(dlc_base: str, batch_size: int) -> int:
    pattern     = re.compile(r'get\("detector_batch_size",\s*\d+\)')
    replacement = f'get("detector_batch_size", {batch_size})'
    targets = [
        os.path.join(dlc_base, "pose_estimation_pytorch", "apis", "videos.py"),
        os.path.join(dlc_base, "pose_estimation_pytorch", "apis", "tracking_dataset.py"),
    ]
    n_patched = 0
    for fpath in targets:
        if not os.path.exists(fpath):
            warn(f"DLC patch target not found (skipping): {os.path.basename(fpath)}")
            continue
        try:
            content = Path(fpath).read_text(encoding="utf-8")
            if not pattern.search(content):
                warn(f"Patch pattern not found in {os.path.basename(fpath)} — "
                     "batch size may default to 1 (inference will be slow)")
                continue
            Path(fpath).write_text(pattern.sub(replacement, content), encoding="utf-8")
            # Remove stale .pyc so subprocess re-compiles from patched .py
            pyc_dir = os.path.join(os.path.dirname(fpath), "__pycache__")
            stem    = os.path.splitext(os.path.basename(fpath))[0]
            for pyc in Path(pyc_dir).glob(f"{stem}.*.pyc") if os.path.isdir(pyc_dir) else []:
                pyc.unlink(missing_ok=True)
            n_patched += 1
            print(f"Patched {os.path.basename(fpath)} → detector_batch_size={batch_size}")
        except PermissionError:
            warn(f"No write permission to patch {fpath}. "
                 "Try running with sudo, or inference will be slow.")
        except Exception as e:
            warn(f"Could not patch {os.path.basename(fpath)}: {e}")
    return n_patched


n_patched = patch_dlc(dlc.__path__[0], det_b)
if n_patched == 0:
    warn("DLC batch size patch did not apply — detector will run at batch=1 (slow but correct)")


# ── Run inference in subprocess ───────────────────────────────────────────────
# Subprocess starts a fresh Python process so the patched DLC files are
# imported clean (not the already-loaded version in this process).

args = {
    "videos":            [str(VIDEO)],
    "dest_folder":       str(OUTPUT.parent),
    "batch_size":        pose_b,
    "detector_batch_size": det_b,
    "device":            device,
}

runner_code = f"""\
import os, sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import deeplabcut as dlc
import json

args = {json.dumps(args)}

print(f"Starting inference: det_b={{args['detector_batch_size']}} "
      f"pose_b={{args['batch_size']}} device={{args['device']}}", flush=True)

try:
    dlc.video_inference_superanimal(
        videos=args["videos"],
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        scale_list=[200],
        videotype="mp4",
        dest_folder=args["dest_folder"],
        create_labeled_video=False,
        plot_trajectories=False,
        batch_size=args["batch_size"],
        detector_batch_size=args["detector_batch_size"],
        device=args["device"],
    )
except MemoryError as e:
    print(f"OUT OF MEMORY: {{e}}", file=sys.stderr)
    sys.exit(2)
except Exception as e:
    print(f"INFERENCE ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

runner_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False, encoding="utf-8")
runner_file.write(runner_code)
runner_file.close()

print(f"\nRunning inference ...")
print(f"  Video:  {VIDEO} ({VIDEO.stat().st_size / 1024**3:.2f} GB)")
print(f"  Output: {OUTPUT}\n")

try:
    ret = subprocess.call([sys.executable, runner_file.name])
finally:
    os.unlink(runner_file.name)

if ret == 2:
    die(
        "GPU ran out of memory. Options:\n"
        "  1. Close other GPU applications and re-run\n"
        "  2. Re-run — det_b will be recalculated from available VRAM"
    )
if ret != 0:
    die("Inference subprocess failed — see error above")


# ── Find and move output H5 ───────────────────────────────────────────────────

h5s: list[Path] = []
for search_dir in [OUTPUT.parent, VIDEO.parent]:
    for pattern in [f"{VIDEO.stem}*superanimal*.h5", f"{VIDEO.stem}*.h5"]:
        h5s = sorted(search_dir.glob(pattern))
        if h5s:
            break
    if h5s:
        break

if not h5s:
    die(
        f"No .h5 output found in {OUTPUT.parent} or {VIDEO.parent}\n"
        "DLC may have saved it next to the video. Check both directories manually."
    )

src = h5s[0]
if src.resolve() != OUTPUT.resolve():
    shutil.move(str(src), str(OUTPUT))

size_mb = OUTPUT.stat().st_size / 1024**2
if size_mb < 1.0:
    die(f"Output .h5 is only {size_mb:.2f} MB — inference likely failed silently. "
        "Check that 'Running detector with batch size X' appeared above.")

print(f"\nDone.")
print(f"  Output: {OUTPUT} ({size_mb:.0f} MB)")
print(f"\nSend this file back and place it at:")
print(f"  outputs/pose/session_{SESSION_ID}_superanimal.h5")
