#!/usr/bin/env python3
"""Download the exact public PROCEDURE checkpoints into the Docker build tree."""

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Div97/orena-focus-procedure-w64-nw2"
REVISION = "4b49db331bf8e49007d19135aee48dfcf27c1bfa"
TARGET = Path(__file__).resolve().parents[1] / "resources"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=TARGET,
        allow_patterns=["w64/**", "nw2/**"],
    )
    print(f"Downloaded {REPO_ID}@{REVISION} to {TARGET}")


if __name__ == "__main__":
    main()
