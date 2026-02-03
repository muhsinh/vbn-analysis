#!/usr/bin/env python
"""Prepare (download + stage) the next N session videos.

This is a helper to automate the "find the next videos" loop:
1) read a list of session IDs
2) for each session, ensure a raw MP4 exists locally (download from S3 if needed)
3) stage the chosen video into outputs/<session_id>/videos via symlink (default)

It intentionally avoids the AllenSDK cache for MP4s: the raw videos are hosted
separately on S3.

Usage:
  python scripts/prepare_next_videos.py --sessions-file sessions.txt --n 5

sessions.txt can be:
  - one session id per line
  - a CSV that contains a 'session_id' column
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Iterable

from vbn.config import get_cache_dir, get_outputs_dir, get_video_dirs, get_video_preferred_camera, get_video_stage
from vbn.utils import setup_logging, print_diagnostic, ensure_dir
from vbn.video import resolve_video_path, stage_video, validate_video_open


DEFAULT_S3_PREFIX = "s3://allen-brain-observatory/visual-behavior-neuropixels/raw-data/"


def _read_session_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Sessions file not found: {path}")

    if path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "session_id" not in reader.fieldnames:
                raise ValueError(f"CSV must contain a 'session_id' column: {path}")
            ids: list[int] = []
            for row in reader:
                raw = (row.get("session_id") or "").strip()
                if not raw:
                    continue
                ids.append(int(raw))
            return ids

    # Plain text: pull all integers
    text = path.read_text()
    ids = [int(m.group(0)) for m in re.finditer(r"\b\d+\b", text)]
    return ids


def _aws_available() -> bool:
    try:
        subprocess.run(["aws", "--version"], check=False, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def _list_s3_candidates(session_id: int, s3_prefix: str) -> list[tuple[int, str]]:
    """Return [(size_bytes, s3_key_or_url), ...] for entries that match the session id and look like videos."""
    # Note: this scans the prefix listing output. It's simple and robust, but may be slow.
    cmd = ["aws", "s3", "ls", s3_prefix, "--no-sign-request", "--recursive"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "aws s3 ls failed")

    results: list[tuple[int, str]] = []
    needle = str(session_id)
    for line in proc.stdout.splitlines():
        # Typical: "2020-01-01 00:00:00    12345 path/to/file.mp4"
        if needle not in line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        _, _, size_str, key = parts
        if not key.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        try:
            size = int(size_str)
        except ValueError:
            size = 0

        if s3_prefix.startswith("s3://"):
            url = s3_prefix.rstrip("/") + "/" + key
        else:
            url = key
        results.append((size, url))

    return results


def _choose_best_s3_key(candidates: Iterable[tuple[int, str]], camera: str) -> str | None:
    camera = camera.lower()
    scored: list[tuple[int, int, int, str]] = []
    for size, url in candidates:
        s = url.lower()
        ext_rank = 0
        if s.endswith(".mp4"):
            ext_rank = 0
        elif s.endswith(".mov"):
            ext_rank = 1
        elif s.endswith(".avi"):
            ext_rank = 2
        else:
            ext_rank = 3

        cam_bonus = 0
        if camera != "any":
            if camera == "body" and any(k in s for k in ("body", "side", "behavior")):
                cam_bonus = 1
            elif camera == "eye" and any(k in s for k in ("eye", "pupil")):
                cam_bonus = 1
            elif camera == "face" and any(k in s for k in ("face", "front")):
                cam_bonus = 1

        # Higher cam_bonus and size are better, ext_rank lower is better.
        scored.append((-cam_bonus, ext_rank, -size, url))

    if not scored:
        return None
    return min(scored)[-1]


def _download_s3(url: str, dest_dir: Path) -> Path:
    dest_dir = ensure_dir(dest_dir)
    filename = Path(url).name
    dest = dest_dir / filename
    if dest.exists():
        return dest

    cmd = ["aws", "s3", "cp", url, str(dest), "--no-sign-request"]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"aws s3 cp failed for: {url}")
    return dest


def main() -> int:
    default_camera = get_video_preferred_camera()
    default_stage = get_video_stage()

    parser = argparse.ArgumentParser(description="Prepare (download + stage) the next N session videos")
    parser.add_argument("--sessions-file", type=Path, required=True, help="Text or CSV file containing session ids")
    parser.add_argument("--n", type=int, default=5, help="How many sessions to prepare (default: 5)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into the sessions list (default: 0)")
    parser.add_argument(
        "--camera",
        choices=["body", "eye", "face", "any"],
        default=default_camera,
        help=f"Which camera to prefer (default: {default_camera})",
    )
    parser.add_argument(
        "--stage",
        choices=["symlink", "copy", "none"],
        default=default_stage,
        help=f"Stage method into outputs/<session_id>/videos (default: {default_stage})",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Where to store downloaded raw videos (defaults to first VBN_VIDEO_DIRS entry, else ~/data/vbn_videos)",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=DEFAULT_S3_PREFIX,
        help=f"S3 prefix to search (default: {DEFAULT_S3_PREFIX})",
    )
    parser.add_argument(
        "--skip-if-staged",
        action="store_true",
        help="Skip sessions that already have outputs/<id>/videos/videos.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without downloading/staging")
    args = parser.parse_args()

    logger = setup_logging()

    session_ids = _read_session_ids(args.sessions_file)
    if args.start_index < 0 or args.start_index >= len(session_ids):
        raise ValueError(f"--start-index out of range: {args.start_index} (list size: {len(session_ids)})")

    outputs_dir = get_outputs_dir()
    cache_dir = get_cache_dir()

    configured_video_dirs = get_video_dirs()
    if args.video_dir is not None:
        raw_video_dir = args.video_dir.expanduser().resolve()
    elif configured_video_dirs:
        raw_video_dir = configured_video_dirs[0]
    else:
        raw_video_dir = (Path.home() / "data" / "vbn_videos").resolve()

    # Always include the raw video dir for discovery, plus any configured dirs.
    video_dirs = [raw_video_dir, *configured_video_dirs]

    if not _aws_available() and not args.dry_run:
        logger.error("AWS CLI not found. Install it first (brew install awscli) and try again.")
        return 1

    prepared: list[dict[str, str]] = []
    attempted = 0

    for session_id in session_ids[args.start_index :]:
        if attempted >= args.n:
            break

        staged_index = outputs_dir / str(session_id) / "videos" / "videos.json"
        if args.skip_if_staged and staged_index.exists():
            continue

        attempted += 1

        # Try local first (fast).
        local = resolve_video_path(
            session_id=session_id,
            camera=args.camera,
            video_dirs=video_dirs,
            cache_dir=cache_dir,
            outputs_dir=outputs_dir,
        )

        downloaded = None
        if local is None:
            if args.dry_run:
                logger.info(f"[dry-run] Would search S3 for session {session_id}")
            else:
                logger.info(f"No local video found for session {session_id}; searching S3...")
                s3_candidates = _list_s3_candidates(session_id, args.s3_prefix)
                best_url = _choose_best_s3_key(s3_candidates, camera=args.camera)
                if best_url is None:
                    logger.warning(f"No matching video keys found on S3 for session {session_id}")
                    continue
                logger.info(f"Downloading: {best_url}")
                downloaded = _download_s3(best_url, raw_video_dir)
                local = downloaded

        if local is None:
            continue

        if args.dry_run:
            logger.info(f"[dry-run] Would validate+stage session {session_id}: {local}")
            prepared.append(
                {"session_id": str(session_id), "video": str(local), "staged": "(dry-run)"}
            )
            continue

        validation = validate_video_open(local)
        if not validation.get("ok", False):
            logger.warning(f"Video exists but could not be opened: {validation.get('error')}")
            logger.warning(f"Path: {local}")
            continue

        staged = stage_video(
            local,
            session_id=session_id,
            method=args.stage,
            outputs_dir=outputs_dir,
            selected_camera=args.camera,
            validation=validation,
        )

        prepared.append(
            {"session_id": str(session_id), "video": str(local), "staged": str(staged)}
        )

    print_diagnostic("Prepared Videos", {
        "Sessions file": args.sessions_file,
        "Start index": args.start_index,
        "Requested N": args.n,
        "Attempted": attempted,
        "Prepared": len(prepared),
        "Raw video dir": raw_video_dir,
        "Video dirs": [str(p) for p in video_dirs],
        "Outputs dir": outputs_dir,
        "S3 prefix": args.s3_prefix,
        "Camera": args.camera,
        "Stage": args.stage,
    })

    if prepared:
        logger.info("Prepared sessions:")
        for item in prepared:
            logger.info(f"  - {item['session_id']}: {item['staged']}")

    if attempted == 0:
        logger.warning("No sessions attempted. Check your --start-index and sessions file contents.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

