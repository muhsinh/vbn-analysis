"""Unsigned S3 helpers for VBN video assets (download-only)."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from config import get_config


def _client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 downloads") from exc
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _filename(camera: str, kind: str) -> str:
    if kind == "video":
        return f"{camera}.mp4"
    if kind == "timestamps":
        return f"{camera}_timestamps.npy"
    if kind == "metadata":
        return f"{camera}_metadata.json"
    raise ValueError(f"Unknown asset kind: {kind}")


def s3_key(session_id: int, camera: str, kind: str, base_path: str | None = None) -> str:
    cfg = get_config()
    base = base_path or cfg.video_base_path
    return f"{base}/{session_id}/behavior_videos/{_filename(camera, kind)}"


def s3_uri(session_id: int, camera: str, kind: str, bucket: str | None = None, base_path: str | None = None) -> str:
    cfg = get_config()
    bucket_name = bucket or cfg.video_bucket
    return f"s3://{bucket_name}/{s3_key(session_id, camera, kind, base_path)}"


def http_url(session_id: int, camera: str, kind: str, bucket: str | None = None, base_path: str | None = None) -> str:
    cfg = get_config()
    bucket_name = bucket or cfg.video_bucket
    key = s3_key(session_id, camera, kind, base_path)
    return f"https://{bucket_name}.s3.amazonaws.com/{key}"


def list_video_assets(session_id: int, cameras: list[str] | None = None) -> dict[str, dict[str, str]]:
    cfg = get_config()
    cams = cameras or cfg.video_cameras
    assets: dict[str, dict[str, str]] = {}
    for camera in cams:
        assets[camera] = {
            "s3_uri_video": s3_uri(session_id, camera, "video", cfg.video_bucket, cfg.video_base_path),
            "s3_uri_timestamps": s3_uri(session_id, camera, "timestamps", cfg.video_bucket, cfg.video_base_path),
            "s3_uri_metadata": s3_uri(session_id, camera, "metadata", cfg.video_bucket, cfg.video_base_path),
            "http_url_video": http_url(session_id, camera, "video", cfg.video_bucket, cfg.video_base_path),
        }
    return assets


def download_asset(s3_uri_path: str, local_path: Path) -> Path:
    parsed = urlparse(s3_uri_path)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got {s3_uri_path}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = _client()
    client.download_file(bucket, key, str(local_path))
    return local_path

