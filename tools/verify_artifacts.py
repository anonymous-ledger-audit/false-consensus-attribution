#!/usr/bin/env python3
"""Verify the compact frozen artifact release without training a model."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RELEASE_MANIFEST = ROOT / "artifact_manifest_sha256.json"

EXPECTED_DIRECTORIES = {
    "e1_structural_outputs",
    "e2_outputs",
    "e3_tabular_protocol",
    "e3_tabular_controls",
    "e3_tabular_mlp",
    "e3_tabular_crossnet",
    "e3_tabular_cross_architecture",
    "e3_vision_protocol",
    "e3_vision_controls",
    "e3_vision_cnn",
    "paper_figures",
}

LEGACY_NAMES = {
    "cell_summary.csv",
    "FIGURE_MANIFEST.txt",
    "parsed_audit_rows.csv",
    "replicate_summary.csv",
}

TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
FORBIDDEN_TEXT = {
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1"}


def verify_csv(path: Path, expected_rows: int) -> tuple[int, bool]:
    rows = 0
    certified = True
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if "resolved" in row:
                certified &= truthy(row["resolved"])
            if "certification_pass" in row:
                certified &= truthy(row["certification_pass"])
    return rows, rows == expected_rows and certified


def verify_file_map(
    directory: Path,
    files: dict[str, str],
) -> tuple[int, int, list[str]]:
    checked = 0
    skipped_checkpoints = 0
    failures: list[str] = []
    for name, expected in files.items():
        path = directory / name
        if path.suffix.lower() == ".pt":
            skipped_checkpoints += 1
            continue
        if not path.exists():
            failures.append(f"missing manifest file: {path.relative_to(ROOT)}")
            continue
        checked += 1
        if sha256(path) != expected:
            failures.append(f"hash mismatch: {path.relative_to(ROOT)}")
    return checked, skipped_checkpoints, failures


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        value = value[key]
    return value


def main() -> int:
    failures: list[str] = []

    release_manifest = read_json(RELEASE_MANIFEST)["files"]
    actual_artifact_files = {
        path.relative_to(ROOT).as_posix(): path
        for path in ARTIFACTS.rglob("*")
        if path.is_file()
    }
    if set(release_manifest) != set(actual_artifact_files):
        failures.append("release artifact-manifest inventory mismatch")
    for relative, path in actual_artifact_files.items():
        item = release_manifest.get(relative)
        if item is None:
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"release artifact-manifest mismatch: {relative}")

    actual_directories = {path.name for path in ARTIFACTS.iterdir() if path.is_dir()}
    if actual_directories != EXPECTED_DIRECTORIES:
        failures.append(
            "artifact-directory mismatch: "
            f"expected={sorted(EXPECTED_DIRECTORIES)} actual={sorted(actual_directories)}"
        )

    if list(ARTIFACTS.rglob("*.pt")):
        failures.append("trained checkpoints are present in the compact release")

    for path in ARTIFACTS.rglob("*"):
        if not path.is_file():
            continue
        if path.name in LEGACY_NAMES or re.fullmatch(r"fig[0-9]_.*", path.stem):
            failures.append(f"legacy artifact present: {path.relative_to(ROOT)}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for label, pattern in FORBIDDEN_TEXT.items():
                if pattern.search(text):
                    failures.append(f"{label}: {path.relative_to(ROOT)}")

    # Stage manifests.
    tab_protocol_dir = ARTIFACTS / "e3_tabular_protocol"
    tab_protocol_manifest = read_json(tab_protocol_dir / "manifest_sha256.json")
    _, _, manifest_failures = verify_file_map(
        tab_protocol_dir, tab_protocol_manifest
    )
    failures.extend(manifest_failures)

    stage_manifests = [
        ("e3_vision_protocol", "manifest_sha256.json"),
        ("e3_vision_controls", "control_manifest_sha256.json"),
        ("e3_vision_cnn", "cnn_manifest_sha256.json"),
    ]
    skipped_checkpoints = 0
    for directory_name, manifest_name in stage_manifests:
        directory = ARTIFACTS / directory_name
        manifest = read_json(directory / manifest_name)
        _, skipped, stage_failures = verify_file_map(directory, manifest["files"])
        skipped_checkpoints += skipped
        failures.extend(stage_failures)

    # Portable tabular paper-figure manifest.
    paper_dir = ARTIFACTS / "paper_figures"
    tab_figure_manifest = read_json(paper_dir / "e3_figure_manifest_sha256.json")
    for section in ("inputs", "outputs"):
        for label, item in tab_figure_manifest[section].items():
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                failures.append(f"non-portable figure path: {label}")
                continue
            path = ROOT / relative
            if not path.exists() or sha256(path) != item["sha256"]:
                failures.append(f"tabular figure manifest mismatch: {label}")

    # Vision paper-artifact manifest and its three input manifests.
    vision_paper_manifest = read_json(paper_dir / "e3b_artifact_manifest_sha256.json")
    vision_manifest_paths = {
        "protocol": ARTIFACTS / "e3_vision_protocol/manifest_sha256.json",
        "controls": ARTIFACTS / "e3_vision_controls/control_manifest_sha256.json",
        "cnn": ARTIFACTS / "e3_vision_cnn/cnn_manifest_sha256.json",
    }
    for label, path in vision_manifest_paths.items():
        if sha256(path) != vision_paper_manifest["input_manifests"][label]:
            failures.append(f"vision input-manifest mismatch: {label}")
    for name, item in vision_paper_manifest["outputs"].items():
        path = paper_dir / name
        if (
            not path.exists()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            failures.append(f"vision paper-artifact mismatch: {name}")

    # Frozen pass flags.
    pass_checks = {
        "E1": nested(
            read_json(ARTIFACTS / "e1_structural_outputs/e1_summary.json"),
            "checks",
            "all_pass",
        ),
        "E2": nested(
            read_json(ARTIFACTS / "e2_outputs/e2_summary.json"), "all_pass"
        ),
        "E3a protocol": nested(
            read_json(ARTIFACTS / "e3_tabular_protocol/protocol.json"), "all_pass"
        ),
        "E3a controls": nested(
            read_json(ARTIFACTS / "e3_tabular_controls/control_summary.json"),
            "all_pass",
        ),
        "E3a MLP": nested(
            read_json(ARTIFACTS / "e3_tabular_mlp/mlp_summary.json"),
            "all_hard_checks_pass",
        ),
        "E3a CrossNet": nested(
            read_json(ARTIFACTS / "e3_tabular_crossnet/crossnet_summary.json"),
            "all_hard_checks_pass",
        ),
        "E3b protocol": nested(
            read_json(ARTIFACTS / "e3_vision_protocol/protocol.json"),
            "protocol_checks",
            "all_pass",
        ),
        "E3b controls": nested(
            read_json(ARTIFACTS / "e3_vision_controls/control_summary.json"),
            "all_pass",
        ),
        "E3b CNN": nested(
            read_json(ARTIFACTS / "e3_vision_cnn/cnn_summary.json"),
            "all_hard_checks_pass",
        ),
    }
    for label, passed in pass_checks.items():
        if passed is not True:
            failures.append(f"hard-pass flag is false: {label}")

    cross_architecture = read_json(
        ARTIFACTS / "e3_tabular_cross_architecture/cross_architecture_summary.json"
    )
    if (
        cross_architecture.get("n_endpoints") != 100
        or cross_architecture.get("n_seeds_per_architecture") != 5
    ):
        failures.append("cross-architecture paired design mismatch")

    control_summary = read_json(
        ARTIFACTS / "e3_tabular_controls/control_summary.json"
    )
    if control_summary.get("protocol_dir") != "artifacts/e3_tabular_protocol":
        failures.append("tabular protocol path is not release-relative")

    audit_tables = {
        "E3a controls": ("e3_tabular_controls/control_audits.csv", 1000),
        "E3a MLP": ("e3_tabular_mlp/mlp_audits.csv", 500),
        "E3a CrossNet": ("e3_tabular_crossnet/crossnet_audits.csv", 500),
        "E3b controls": ("e3_vision_controls/control_audits.csv", 1000),
        "E3b CNN": ("e3_vision_cnn/cnn_audits.csv", 500),
    }
    audit_report: dict[str, int] = {}
    for label, (relative, expected_rows) in audit_tables.items():
        rows, passed = verify_csv(ARTIFACTS / relative, expected_rows)
        audit_report[label] = rows
        if not passed:
            failures.append(f"audit table failed: {label} ({rows} rows)")

    print(f"Artifact directories : {len(actual_directories)}")
    print(f"Paper artifacts      : {len(list(paper_dir.iterdir()))}")
    print(f"Skipped checkpoints  : {skipped_checkpoints} manifest entries")
    for label, rows in audit_report.items():
        print(f"{label:<20}: {rows} certified rows")
    print(f"Hard-pass flags      : {'PASS' if all(pass_checks.values()) else 'FAIL'}")
    print(f"Release manifest     : {'PASS' if not any('release artifact-manifest' in x for x in failures) else 'FAIL'}")
    print(f"Internal manifests   : {'PASS' if not any('manifest' in x for x in failures) else 'FAIL'}")
    print(f"Anonymity scan       : {'PASS' if not any('path' in x or 'email' in x for x in failures) else 'FAIL'}")

    if failures:
        print("\nArtifact verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Artifact release     : PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
