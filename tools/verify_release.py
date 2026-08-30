#!/usr/bin/env python3
"""Verify the anonymous experiment-code release without importing it."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "ICLR"
MANIFEST_PATH = ROOT / "code_manifest_sha256.json"

EXPECTED_SCRIPTS = {
    "e1_structural_stress_test.py",
    "e2_smooth_realizability.py",
    "e3_tabular_controls.py",
    "e3_tabular_cross_architecture.py",
    "e3_tabular_crossnet.py",
    "e3_tabular_mlp.py",
    "e3_tabular_mlp_endpoint_stability.py",
    "e3_tabular_mlp_threshold_surface.py",
    "e3_tabular_paper_figures.py",
    "e3_tabular_protocol.py",
    "e3_vision_cnn.py",
    "e3_vision_controls.py",
    "e3_vision_paper_artifacts.py",
    "e3_vision_protocol.py",
    "fig_forest_vs_cycle.py",
    "fig_ledger_aggregation.py",
}

FORBIDDEN_PATTERNS = {
    "email address": re.compile(
        r"(?<![\w.])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.])",
        re.IGNORECASE,
    ),
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    "POSIX home path": re.compile(r"/(?:Users|home)/[^/\s]+/", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    actual = {path.name for path in SOURCE_DIR.glob("*.py")}

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        manifest = {}
        failures.append(f"invalid source manifest: {exc}")

    missing = sorted(EXPECTED_SCRIPTS - actual)
    unexpected = sorted(actual - EXPECTED_SCRIPTS)
    if missing:
        failures.append(f"missing scripts: {missing}")
    if unexpected:
        failures.append(f"unexpected scripts: {unexpected}")

    hashes: dict[str, str] = {}
    for name in sorted(actual):
        path = SOURCE_DIR / name
        text = path.read_text(encoding="utf-8")
        try:
            compile(text, str(path.relative_to(ROOT)), "exec")
        except SyntaxError as exc:
            failures.append(f"syntax error in {name}: {exc}")

        for label, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{name}:{line}: {label}")

        hashes[name] = sha256(path)
        manifest_key = f"ICLR/{name}"
        if manifest.get(manifest_key) != hashes[name]:
            failures.append(f"manifest mismatch: {manifest_key}")

    unexpected_manifest_entries = sorted(set(manifest) - {
        f"ICLR/{name}" for name in EXPECTED_SCRIPTS
    })
    if unexpected_manifest_entries:
        failures.append(
            f"unexpected manifest entries: {unexpected_manifest_entries}"
        )

    print(f"Expected scripts : {len(EXPECTED_SCRIPTS)}")
    print(f"Present scripts  : {len(actual)}")
    print(f"Syntax checks    : {'PASS' if not any('syntax error' in x for x in failures) else 'FAIL'}")
    print(f"Anonymity scan   : {'PASS' if not any('marker' in x or 'path' in x or 'email' in x for x in failures) else 'FAIL'}")
    print(f"Manifest check   : {'PASS' if not any('manifest' in x for x in failures) else 'FAIL'}")

    if failures:
        print("\nRelease verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Inventory check  : PASS")
    print("\nSHA-256 source inventory:")
    for name, digest in hashes.items():
        print(f"  {digest}  ICLR/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
