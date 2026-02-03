import os
import sys
from pathlib import Path

import pytest


def _add_src_to_path():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from vbn.video import resolve_video_path, stage_video  # noqa: E402


SESSION_ID = 1055240613


def _write_dummy(path: Path, size_bytes: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)


def test_resolve_matches_session_id_in_parent_dir(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    video = root / str(SESSION_ID) / "body.mp4"
    _write_dummy(video, size_bytes=10)

    resolved = resolve_video_path(
        session_id=SESSION_ID,
        camera="body",
        video_dirs=[root],
        cache_dir=None,
        outputs_dir=None,
    )
    assert resolved is not None
    assert resolved.resolve() == video.resolve()


def test_resolve_case_insensitive_extension(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    video = root / f"eye_{SESSION_ID}.MP4"
    _write_dummy(video, size_bytes=10)

    resolved = resolve_video_path(
        session_id=SESSION_ID,
        camera="eye",
        video_dirs=[root],
        cache_dir=None,
        outputs_dir=None,
    )
    assert resolved is not None
    assert resolved.resolve() == video.resolve()


def test_resolve_camera_keyword_from_path_component(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    video = root / str(SESSION_ID) / "front" / "video.mp4"
    _write_dummy(video, size_bytes=10)

    resolved = resolve_video_path(
        session_id=SESSION_ID,
        camera="face",
        video_dirs=[root],
        cache_dir=None,
        outputs_dir=None,
    )
    assert resolved is not None
    assert resolved.resolve() == video.resolve()


def test_resolve_prefers_mp4_over_other_extensions(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    mp4 = root / str(SESSION_ID) / "body_small.mp4"
    avi = root / str(SESSION_ID) / "body_big.avi"
    _write_dummy(mp4, size_bytes=10)
    _write_dummy(avi, size_bytes=10_000)

    resolved = resolve_video_path(
        session_id=SESSION_ID,
        camera="body",
        video_dirs=[root],
        cache_dir=None,
        outputs_dir=None,
    )
    assert resolved is not None
    assert resolved.resolve() == mp4.resolve()


def test_stage_video_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks not supported on this platform")

    src_root = tmp_path / "videos"
    src = src_root / str(SESSION_ID) / "body.mp4"
    _write_dummy(src, size_bytes=10)

    out_root = tmp_path / "outputs"
    try:
        staged = stage_video(src, session_id=SESSION_ID, method="symlink", outputs_dir=out_root)
    except RuntimeError as e:
        pytest.skip(f"symlink creation failed in this environment: {e}")

    assert staged.exists()
    assert staged.is_symlink()
    assert staged.resolve() == src.resolve()
    assert (out_root / str(SESSION_ID) / "videos" / "videos.json").exists()

