"""Download additional VBN NWBs for cross-session replication.

Uses AllenSDK's VisualBehaviorNeuropixelsProjectCache to fetch full session
NWB files. Skips if already cached. Designed to run unattended overnight.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Sessions selected by coverage + image set matching 1055240613
SESSION_IDS = [1067588044, 1115086689]

CACHE_DIR = Path("/Users/muh/projects/vbn-analysis/data/allensdk_cache")


def main():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )

    print(f"Cache dir: {CACHE_DIR}")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

    for sid in SESSION_IDS:
        print(f"\n{'='*70}")
        print(f"Session {sid}")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            # Force NWB download via get_ecephys_session
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            n_units = len(session.get_units())
            duration = time.time() - t0
            print(f"  ✓ Downloaded in {duration/60:.1f} min — {n_units} units")
            # Free the session
            del session
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            continue

    print("\nDONE.")


if __name__ == "__main__":
    main()
