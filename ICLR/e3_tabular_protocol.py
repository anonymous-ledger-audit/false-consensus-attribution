#!/usr/bin/env python3
"""
E3a — California Housing protocol freeze
========================================

This script freezes the tabular experimental contract BEFORE any model fitting
or attribution audit is run.

Primary contract
----------------
Dataset:
    California Housing (scikit-learn)

Split:
    70% train / 15% validation / 15% test
    split seed = 20260830

Preprocessing:
    StandardScaler fit on TRAIN ONLY

Primary baseline:
    The ACTUAL training observation closest to the standardized training center
    (Euclidean distance to 0 in train-standardized feature space).

Audit set:
    100 held-out TEST observations
    10 from each test-target decile
    audit seed = 20260831

Important:
    - The same frozen split, baseline, and 100 audit endpoints must be reused
      for every model family and every model seed.
    - This script does NOT fit any predictive model.
    - This script does NOT compute any attribution.
    - After the protocol is frozen, do not change these choices in response to
      attribution results.

Outputs
-------
e3_tabular_protocol/
    split_indices.npz
    audit_indices.csv
    baseline.csv
    preprocessing.json
    protocol.json
    manifest_sha256.json

Run
---
python e3_tabular_protocol.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    payload = (
        str(arr.dtype).encode("utf-8")
        + str(arr.shape).encode("utf-8")
        + arr.tobytes()
    )
    return sha256_bytes(payload)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(
        as_frame=True,
        download_if_missing=True,
    )

    X_df = data.data.copy()
    y_ser = data.target.copy()

    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy(dtype=np.float64)

    n, d = X.shape
    all_idx = np.arange(n, dtype=np.int64)

    if d != 8:
        raise RuntimeError(f"Expected 8 California Housing features, found {d}.")

    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=args.split_seed,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=args.split_seed,
        shuffle=True,
    )

    train_idx = np.sort(train_idx.astype(np.int64))
    val_idx = np.sort(val_idx.astype(np.int64))
    test_idx = np.sort(test_idx.astype(np.int64))

    scaler = StandardScaler()
    scaler.fit(X[train_idx])

    X_train_std = scaler.transform(X[train_idx])
    center = np.zeros(d, dtype=np.float64)

    dist2 = np.sum((X_train_std - center) ** 2, axis=1)
    baseline_train_position = int(np.argmin(dist2))
    baseline_dataset_index = int(train_idx[baseline_train_position])

    baseline_raw = X[baseline_dataset_index].copy()
    baseline_std = scaler.transform(baseline_raw.reshape(1, -1))[0]
    baseline_target = float(y[baseline_dataset_index])

    test_targets = y[test_idx]

    deciles = pd.qcut(
        pd.Series(test_targets),
        q=10,
        labels=False,
        duplicates="raise",
    ).to_numpy(dtype=int)

    rng = np.random.default_rng(args.audit_seed)

    chosen_test_positions = []
    for decile in range(10):
        candidates = np.flatnonzero(deciles == decile)

        if len(candidates) < args.per_decile:
            raise RuntimeError(
                f"Target decile {decile} contains only {len(candidates)} "
                f"test points; need {args.per_decile}."
            )

        picked = rng.choice(
            candidates,
            size=args.per_decile,
            replace=False,
        )
        chosen_test_positions.extend(int(x) for x in picked)

    chosen_test_positions = np.asarray(chosen_test_positions, dtype=np.int64)

    audit_dataset_idx = test_idx[chosen_test_positions]
    audit_deciles = deciles[chosen_test_positions]
    audit_targets = y[audit_dataset_idx]

    order = np.lexsort((audit_dataset_idx, audit_targets, audit_deciles))

    chosen_test_positions = chosen_test_positions[order]
    audit_dataset_idx = audit_dataset_idx[order]
    audit_deciles = audit_deciles[order]
    audit_targets = audit_targets[order]

    audit_rows = []
    for audit_id, (test_pos, dataset_idx, decile) in enumerate(
        zip(chosen_test_positions, audit_dataset_idx, audit_deciles)
    ):
        raw = X[int(dataset_idx)]
        std = scaler.transform(raw.reshape(1, -1))[0]

        row = {
            "audit_id": int(audit_id),
            "dataset_index": int(dataset_idx),
            "test_position": int(test_pos),
            "target_decile": int(decile) + 1,
            "target": float(y[int(dataset_idx)]),
        }

        for j, name in enumerate(X_df.columns):
            row[f"raw__{name}"] = float(raw[j])
            row[f"std__{name}"] = float(std[j])

        audit_rows.append(row)

    audit_df = pd.DataFrame(audit_rows)

    np.savez_compressed(
        out_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        baseline_dataset_index=np.asarray([baseline_dataset_index], dtype=np.int64),
        audit_dataset_idx=audit_dataset_idx.astype(np.int64),
        audit_test_positions=chosen_test_positions.astype(np.int64),
    )

    audit_df.to_csv(out_dir / "audit_indices.csv", index=False)

    baseline_row = {
        "dataset_index": baseline_dataset_index,
        "train_position": baseline_train_position,
        "target": baseline_target,
        "standardized_distance_to_train_center": float(
            np.sqrt(dist2[baseline_train_position])
        ),
    }

    for j, name in enumerate(X_df.columns):
        baseline_row[f"raw__{name}"] = float(baseline_raw[j])
        baseline_row[f"std__{name}"] = float(baseline_std[j])

    pd.DataFrame([baseline_row]).to_csv(
        out_dir / "baseline.csv",
        index=False,
    )

    preprocessing = {
        "feature_names": list(X_df.columns),
        "n_features": int(d),
        "feature_preprocessing": "StandardScaler fit on training split only",
        "scaler_mean": {
            name: float(v)
            for name, v in zip(X_df.columns, scaler.mean_)
        },
        "scaler_scale": {
            name: float(v)
            for name, v in zip(X_df.columns, scaler.scale_)
        },
        "train_raw_feature_min": {
            name: float(v)
            for name, v in zip(X_df.columns, X[train_idx].min(axis=0))
        },
        "train_raw_feature_max": {
            name: float(v)
            for name, v in zip(X_df.columns, X[train_idx].max(axis=0))
        },
        "train_target_mean": float(np.mean(y[train_idx])),
        "train_target_std": float(np.std(y[train_idx], ddof=0)),
        "train_target_q05": float(np.quantile(y[train_idx], 0.05)),
        "train_target_q50": float(np.quantile(y[train_idx], 0.50)),
        "train_target_q95": float(np.quantile(y[train_idx], 0.95)),
        "train_target_q95_minus_q05": float(
            np.quantile(y[train_idx], 0.95)
            - np.quantile(y[train_idx], 0.05)
        ),
        "note_on_future_model_scale": (
            "The primary attribution normalization s is model-specific. "
            "For each fitted model, compute "
            "s = Q0.95(f(X_train)) - Q0.05(f(X_train)) using training outputs only."
        ),
    }

    write_json(
        preprocessing,
        out_dir / "preprocessing.json",
    )

    split_union = np.concatenate([train_idx, val_idx, test_idx])

    split_disjoint = (
        len(np.intersect1d(train_idx, val_idx)) == 0
        and len(np.intersect1d(train_idx, test_idx)) == 0
        and len(np.intersect1d(val_idx, test_idx)) == 0
    )

    split_complete = (
        len(split_union) == n
        and len(np.unique(split_union)) == n
        and np.array_equal(np.sort(split_union), all_idx)
    )

    audit_unique = len(np.unique(audit_dataset_idx)) == len(audit_dataset_idx)
    audit_is_test_only = np.all(np.isin(audit_dataset_idx, test_idx))
    audit_not_train = not np.any(np.isin(audit_dataset_idx, train_idx))
    baseline_in_train = np.any(train_idx == baseline_dataset_index)
    baseline_not_test = not np.any(test_idx == baseline_dataset_index)

    decile_counts = {
        str(int(k)): int(v)
        for k, v in audit_df["target_decile"].value_counts().sort_index().items()
    }
    ten_per_decile = all(
        decile_counts.get(str(k), 0) == args.per_decile
        for k in range(1, 11)
    )

    standardized_train_mean_error = float(
        np.max(np.abs(np.mean(X_train_std, axis=0)))
    )
    standardized_train_std_error = float(
        np.max(np.abs(np.std(X_train_std, axis=0, ddof=0) - 1.0))
    )

    checks = {
        "split_disjoint": bool(split_disjoint),
        "split_complete": bool(split_complete),
        "baseline_in_train": bool(baseline_in_train),
        "baseline_not_in_test": bool(baseline_not_test),
        "audit_n_exact": bool(len(audit_dataset_idx) == args.per_decile * 10),
        "audit_unique": bool(audit_unique),
        "audit_test_only": bool(audit_is_test_only),
        "audit_not_train": bool(audit_not_train),
        "audit_exactly_10_per_target_decile": bool(ten_per_decile),
        "train_standardized_mean_max_abs": standardized_train_mean_error,
        "train_standardized_std_max_abs_error": standardized_train_std_error,
        "train_standardization_check": bool(
            standardized_train_mean_error <= 1e-12
            and standardized_train_std_error <= 1e-12
        ),
    }

    boolean_checks = [
        v for v in checks.values()
        if isinstance(v, bool)
    ]
    all_pass = bool(all(boolean_checks))

    protocol = {
        "experiment": "E3a tabular protocol freeze",
        "dataset": {
            "name": "California Housing",
            "source": "sklearn.datasets.fetch_california_housing",
            "n_rows": int(n),
            "n_features": int(d),
            "feature_names": list(X_df.columns),
            "X_sha256": sha256_array(X),
            "y_sha256": sha256_array(y),
        },
        "split": {
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "split_seed": int(args.split_seed),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "baseline": {
            "rule": (
                "actual training observation closest in Euclidean distance "
                "to the train-standardized feature center"
            ),
            "dataset_index": baseline_dataset_index,
            "train_position": baseline_train_position,
            "target": baseline_target,
            "standardized_distance_to_train_center": float(
                np.sqrt(dist2[baseline_train_position])
            ),
        },
        "audit": {
            "audit_seed": int(args.audit_seed),
            "selection_rule": (
                "from the fixed test split, sample exactly 10 endpoints "
                "uniformly without replacement from each empirical target decile"
            ),
            "per_target_decile": int(args.per_decile),
            "n_audit_endpoints": int(len(audit_dataset_idx)),
            "target_decile_counts": decile_counts,
            "same_endpoints_for_all_models_and_model_seeds": True,
        },
        "future_explanation_contract": {
            "input_space": (
                "train-standardized feature space; baseline and endpoint correspond "
                "to actual/raw observations under the invertible train-fitted scaler"
            ),
            "explained_quantity": "scalar predicted house value",
            "path": "straight line from frozen baseline to held-out endpoint",
            "endpoint_game": "all 2^8 = 256 Boolean corners evaluated exhaustively",
            "primary_baseline": "frozen observed training row defined above",
            "primary_model_output_scale": (
                "s = Q0.95(f(X_train)) - Q0.05(f(X_train)), computed per fitted "
                "model from training outputs only"
            ),
        },
        "future_model_seeds": [20260840, 20260841, 20260842, 20260843, 20260844],
        "checks": checks,
        "all_pass": all_pass,
    }

    write_json(
        protocol,
        out_dir / "protocol.json",
    )

    files_to_hash = [
        "split_indices.npz",
        "audit_indices.csv",
        "baseline.csv",
        "preprocessing.json",
        "protocol.json",
    ]

    manifest = {
        name: sha256_file(out_dir / name)
        for name in files_to_hash
    }

    write_json(
        manifest,
        out_dir / "manifest_sha256.json",
    )

    print()
    print("=" * 72)
    print("E3a — CALIFORNIA HOUSING PROTOCOL FREEZE")
    print("=" * 72)
    print(f"Dataset rows                     : {n}")
    print(f"Features                         : {d}")
    print()
    print(f"Train / validation / test        : "
          f"{len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"Split seed                       : {args.split_seed}")
    print()
    print(f"Primary baseline dataset index   : {baseline_dataset_index}")
    print(f"Baseline train position          : {baseline_train_position}")
    print(f"Baseline target                  : {baseline_target:.6f}")
    print(f"Baseline std distance to center  : "
          f"{np.sqrt(dist2[baseline_train_position]):.6f}")
    print()
    print(f"Audit seed                       : {args.audit_seed}")
    print(f"Audit endpoints                  : {len(audit_dataset_idx)}")
    print("Audit count by target decile     :")
    for k in range(1, 11):
        print(f"  decile {k:2d}: {decile_counts.get(str(k), 0)}")
    print()
    print(f"Train standardized mean max |.| : "
          f"{standardized_train_mean_error:.3e}")
    print(f"Train standardized std max error: "
          f"{standardized_train_std_error:.3e}")
    print()
    print("Protocol checks:")
    for name, value in checks.items():
        if isinstance(value, bool):
            print(f"  {name}: {value}")
    print(f"  all_pass: {all_pass}")
    print()
    print("Frozen future model seeds:")
    print(" ", protocol["future_model_seeds"])
    print()
    print(f"Outputs                          : {out_dir.resolve()}")
    print("=" * 72)

    if not all_pass:
        raise RuntimeError(
            "Protocol freeze FAILED one or more checks. "
            "Do not proceed to model fitting."
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="Freeze E3a California Housing protocol before model fitting."
    )
    p.add_argument(
        "--out-dir",
        default="./e3_tabular_protocol",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=20260830,
    )
    p.add_argument(
        "--audit-seed",
        type=int,
        default=20260831,
    )
    p.add_argument(
        "--per-decile",
        type=int,
        default=10,
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
