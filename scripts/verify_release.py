#!/usr/bin/env python3
"""Verify source files and, when downloaded, both merged checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "inference.py": "3d12f0bd959408a6ae1f88f325e56e0e20d6097af7ec38e0e9cba7c7e84dbd7b",
    "requirements.txt": "307f1dd2031f13b0e696bb63fcce743a19da144d5417119fa4c24c1825906a1b",
    "resources/candidate_provenance.json": "9bc9d5e778976442c4e340348a21d02765327d3e06cd18f1c96f1d96a8a1306d",
    "resources/engines.py": "ff2c8d3fe8feac43f4eee3c0c849783036d7da28a3ab88f0620249b604cdd8d2",
    "resources/infer_format.py": "256165783ae8a6905639b8cb7c6015a8f791a9926d0e103eb93c54bd391f5509",
    "resources/reappearance_presence.py": "18f391b134989ed0a8876b4b123ba75f743073e33dbd31a029f2d92a677bb7b7",
    "resources/robust_io.py": "65a564595a83b11ef7f79b822f77d4834869894b60a876ac24e71cc97780de71",
    "resources/video_sampler.py": "4e63fe61b1a1b34e655d2c090a966fb910bb91d565d511fe3747a87e3a0fbf87",
    "resources/w64/model.safetensors": "0647d7204fccbf708e5b15d8312d03062e86bea61310580b5eca97999c33fe4d",
    "resources/nw2/model.safetensors": "3c032078c4e98a33bd7deb6b0f285ed45f0414d118fa1a086bd35d64546ba3e9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failed = False
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.exists():
            if relative.startswith(("resources/w64/", "resources/nw2/")):
                print(f"SKIP {relative}: run scripts/download_weights.py")
                continue
            print(f"MISS {relative}")
            failed = True
            continue
        actual = sha256(path)
        status = "OK" if actual == expected else "FAIL"
        print(f"{status} {relative} {actual}")
        failed |= actual != expected
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
