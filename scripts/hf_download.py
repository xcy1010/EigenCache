#!/usr/bin/env python3
"""Generic Hugging Face snapshot downloader with sensible defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model or dataset snapshot")
    parser.add_argument("repo_id", help="Repository id, e.g. black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default="model",
        help="Repository type (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        help="Optional git revision to download (branch, tag, or commit).",
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Destination directory for the snapshot (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        help="Restrict download to files matching the given glob pattern (can be repeated).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        help="Skip files that match the given glob pattern (can be repeated).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN env var if set).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional cache directory to reuse downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing directory before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.force:
        # Remove previous contents so the snapshot matches the requested revision.
        for child in destination.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)

    snapshot_kwargs: dict[str, object] = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "local_dir": str(destination),
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if args.revision:
        snapshot_kwargs["revision"] = args.revision
    if args.allow_pattern:
        snapshot_kwargs["allow_patterns"] = args.allow_pattern
    if args.ignore_pattern:
        snapshot_kwargs["ignore_patterns"] = args.ignore_pattern
    if args.token:
        snapshot_kwargs["token"] = args.token
    if args.cache_dir:
        snapshot_kwargs["cache_dir"] = args.cache_dir

    snapshot_path = snapshot_download(**snapshot_kwargs)
    print(f"[INFO] Snapshot downloaded to: {snapshot_path}")


if __name__ == "__main__":
    main()
