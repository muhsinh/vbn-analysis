"""Re-download truncated NWB files for sessions 1055240613 and 1043752325.

Run from repo root with the vbn-analysis conda env active:
    python scripts/redownload_nwb.py

The script deletes any truncated local copy so AllenSDK is forced to
pull the full file from S3.  Expected sizes:
  - 1055240613 : ~2.99 GB
  - 1043752325 : several GB
"""
from __future__ import annotations

import sys
from pathlib import Path

SESSIONS = [1055240613, 1043752325]

NWB_ROOT = Path(
    "data/allensdk_cache/visual-behavior-neuropixels-0.5.0/"
    "behavior_ecephys_sessions"
)

# Minimum acceptable size (1 GB).  Anything smaller is assumed truncated.
MIN_SIZE_BYTES = 1 * 1024**3


def _delete_truncated(session_id: int) -> None:
    nwb_path = NWB_ROOT / str(session_id) / f"ecephys_session_{session_id}.nwb"
    if nwb_path.exists():
        size_mb = nwb_path.stat().st_size / 1024**2
        if nwb_path.stat().st_size < MIN_SIZE_BYTES:
            print(f"[{session_id}] Deleting truncated file ({size_mb:.0f} MB): {nwb_path}")
            nwb_path.unlink()
        else:
            print(f"[{session_id}] File looks complete ({size_mb:.0f} MB) — skipping delete.")
    else:
        print(f"[{session_id}] NWB file not present — will download fresh.")


def main() -> None:
    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError:
        sys.exit(
            "ERROR: allensdk not found.  "
            "Activate the vbn-analysis conda environment first."
        )

    cache_dir = Path("data/allensdk_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for sid in SESSIONS:
        _delete_truncated(sid)

    # Step 2 — instantiate cache (downloads manifest if needed)
    print("\nConnecting to AllenSDK S3 cache …")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=str(cache_dir)
    )

    for sid in SESSIONS:
        print(f"\n[{sid}] Requesting session data (this may take several minutes) …")
        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            print(f"[{sid}] Download complete.")

            # Verify the file is now present and large enough
            nwb_path = NWB_ROOT / str(sid) / f"ecephys_session_{sid}.nwb"
            if nwb_path.exists():
                size_gb = nwb_path.stat().st_size / 1024**3
                print(f"[{sid}] NWB on disk: {size_gb:.2f} GB at {nwb_path}")
            else:
                # AllenSDK may place the file elsewhere — ask the session object
                nwb_attr = getattr(session, "nwb_path", None)
                print(f"[{sid}] NWB attribute on session object: {nwb_attr}")

        except Exception as exc:
            print(f"[{sid}] ERROR: {exc}")
            print(f"[{sid}] You can also download directly via:")
            print(f"        aws s3 cp s3://allen-brain-observatory/"
                  f"visual-behavior-neuropixels/raw-data/"
                  f"behavior_ecephys_sessions/{sid}/ecephys_session_{sid}.nwb "
                  f"{NWB_ROOT}/{sid}/ecephys_session_{sid}.nwb")

    print("\nDone.  Run the pipeline notebook to verify.")


if __name__ == "__main__":
    main()
