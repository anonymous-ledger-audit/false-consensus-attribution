#!/usr/bin/env python3
"""Create the release-level SHA-256 manifest for the artifacts directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "artifact_manifest_sha256.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(item for item in ARTIFACTS.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "algorithm": "sha256",
        "scope": "distributed non-checkpoint experiment artifacts",
        "files": files,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(files)} files.")


if __name__ == "__main__":
    main()
