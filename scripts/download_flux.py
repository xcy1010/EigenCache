#!/usr/bin/env python3
"""Utility script to download FLUX checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FLUX weights into the local models directory")
    parser.add_argument(
        "--repo-id",
        default="black-forest-labs/FLUX.1-dev",
        help="Hugging Face repository id (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        help="Optional git revision (branch/tag/commit) to download.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_MODELS_DIR / "FLUX.1-dev"),
        help="Destination directory for the snapshot (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        help="Restrict download to files matching the given glob patterns (can be repeated).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional Hugging Face cache directory; defaults to the library setting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear the destination directory before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    if args.force and local_dir.exists():
        for item in local_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                import shutil

                shutil.rmtree(item)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_kwargs: dict[str, object] = {
        "repo_id": args.repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if args.revision:
        snapshot_kwargs["revision"] = args.revision
    if args.allow_pattern:
        snapshot_kwargs["allow_patterns"] = args.allow_pattern
    if args.token:
        snapshot_kwargs["token"] = args.token
    if args.cache_dir:
        snapshot_kwargs["cache_dir"] = args.cache_dir

    destination = snapshot_download(**snapshot_kwargs)
    print(f"[INFO] Snapshot downloaded to: {destination}")


if __name__ == "__main__":
    main()
