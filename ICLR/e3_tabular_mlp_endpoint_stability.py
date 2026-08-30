#!/usr/bin/env python3
"""
E3a — MLP endpoint stability analysis
=====================================

Collapses the 500 MLP audits to the correct paired unit: the 100 frozen
endpoints, each evaluated across 5 independently trained model seeds.

Run from the repository root:
    python ICLR/e3_tabular_mlp_endpoint_stability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SEEDS = [20260840, 20260841, 20260842, 20260843, 20260844]
EXPECTED_AUDITS = 100
FC_KAPPA = 0.02
FC_TAU = 0.05


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run(args):
    input_path = Path(args.input_csv)
    out_dir = input_path.parent

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_csv(input_path)

    required = {
        "seed", "audit_id", "dataset_index", "target",
        "D_over_s", "H_over_s", "chi",
        "false_consensus_primary", "resolved", "certification_pass",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for col in ["false_consensus_primary", "resolved", "certification_pass"]:
        if df[col].dtype != bool:
            df[col] = (
                df[col].astype(str).str.lower()
                .map({"true": True, "false": False})
            )

    seeds = sorted(df["seed"].astype(int).unique().tolist())
    audit_ids = sorted(df["audit_id"].astype(int).unique().tolist())

    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"Unexpected seeds: {seeds}")
    if len(audit_ids) != EXPECTED_AUDITS:
        raise RuntimeError(
            f"Expected {EXPECTED_AUDITS} audit IDs, found {len(audit_ids)}"
        )
    if len(df) != EXPECTED_AUDITS * len(EXPECTED_SEEDS):
        raise RuntimeError(f"Expected 500 rows, found {len(df)}")

    per_endpoint_n = df.groupby("audit_id").size()
    if not (per_endpoint_n == len(EXPECTED_SEEDS)).all():
        raise RuntimeError("Some endpoints do not have exactly 5 seeds.")

    if not (df.groupby("audit_id")["dataset_index"].nunique() == 1).all():
        raise RuntimeError("dataset_index varies within an audit_id.")
    if not (df.groupby("audit_id")["target"].nunique() == 1).all():
        raise RuntimeError("target varies within an audit_id.")
    if not df["resolved"].all():
        raise RuntimeError("At least one MLP audit is unresolved.")
    if not df["certification_pass"].all():
        raise RuntimeError("At least one MLP audit failed certification.")

    df["visible_agreement"] = df["D_over_s"] <= FC_KAPPA
    df["material_hidden"] = df["H_over_s"] >= FC_TAU
    df["fc_recomputed"] = df["visible_agreement"] & df["material_hidden"]

    mismatch = int(
        (df["fc_recomputed"] != df["false_consensus_primary"]).sum()
    )
    if mismatch:
        raise RuntimeError(
            f"Stored FC flag disagrees with recomputation in {mismatch} rows."
        )

    rows = []
    for audit_id, g in df.groupby("audit_id", sort=True):
        g = g.sort_values("seed")
        rows.append({
            "audit_id": int(audit_id),
            "dataset_index": int(g["dataset_index"].iloc[0]),
            "target": float(g["target"].iloc[0]),
            "fc_seed_count": int(g["fc_recomputed"].sum()),
            "visible_agreement_seed_count": int(g["visible_agreement"].sum()),
            "material_hidden_seed_count": int(g["material_hidden"].sum()),
            "median_D_over_s": float(g["D_over_s"].median()),
            "min_D_over_s": float(g["D_over_s"].min()),
            "max_D_over_s": float(g["D_over_s"].max()),
            "std_D_over_s": float(g["D_over_s"].std(ddof=0)),
            "median_H_over_s": float(g["H_over_s"].median()),
            "min_H_over_s": float(g["H_over_s"].min()),
            "max_H_over_s": float(g["H_over_s"].max()),
            "std_H_over_s": float(g["H_over_s"].std(ddof=0)),
            "median_chi": (
                float(g["chi"].dropna().median())
                if g["chi"].notna().any() else np.nan
            ),
            "fc_seed_fraction": float(g["fc_recomputed"].mean()),
        })

    endpoint_df = pd.DataFrame(rows).sort_values(
        ["fc_seed_count", "median_H_over_s", "audit_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    fc_only = endpoint_df[endpoint_df["fc_seed_count"] > 0].copy()

    all_path = out_dir / "endpoint_stability_all.csv"
    fc_path = out_dir / "endpoint_stability_fc_only.csv"

    endpoint_df.to_csv(all_path, index=False)
    fc_only.to_csv(fc_path, index=False)

    recurrence_hist = {
        str(k): int((endpoint_df["fc_seed_count"] == k).sum())
        for k in range(6)
    }

    summary = {
        "n_unique_endpoints": int(len(endpoint_df)),
        "n_model_seeds": len(EXPECTED_SEEDS),
        "strict_fc_definition": {
            "D_over_s_max": FC_KAPPA,
            "H_over_s_min": FC_TAU,
        },
        "row_level_fc_count": int(df["fc_recomputed"].sum()),
        "endpoints_with_fc_in_at_least_1_seed": int(
            (endpoint_df["fc_seed_count"] >= 1).sum()
        ),
        "endpoints_with_fc_in_at_least_2_seeds": int(
            (endpoint_df["fc_seed_count"] >= 2).sum()
        ),
        "endpoints_with_fc_in_at_least_3_seeds": int(
            (endpoint_df["fc_seed_count"] >= 3).sum()
        ),
        "endpoints_with_fc_in_all_5_seeds": int(
            (endpoint_df["fc_seed_count"] == 5).sum()
        ),
        "fc_recurrence_histogram_seed_count_to_endpoint_count":
            recurrence_hist,
    }
    write_json(summary, out_dir / "endpoint_stability_summary.json")

    print()
    print("=" * 88)
    print("E3a — MLP ENDPOINT STABILITY")
    print("=" * 88)
    print(f"Unique frozen endpoints                 : {len(endpoint_df)}")
    print(f"Independent model seeds                 : {len(EXPECTED_SEEDS)}")
    print(f"Row-level strict FC events              : {int(df['fc_recomputed'].sum())}/500")
    print()
    print(
        f"Endpoints with FC in >=1 seed           : "
        f"{int((endpoint_df['fc_seed_count'] >= 1).sum())}/100"
    )
    print(
        f"Endpoints with FC in >=2 seeds          : "
        f"{int((endpoint_df['fc_seed_count'] >= 2).sum())}/100"
    )
    print(
        f"Endpoints with FC in >=3 seeds          : "
        f"{int((endpoint_df['fc_seed_count'] >= 3).sum())}/100"
    )
    print(
        f"Endpoints with FC in all 5 seeds        : "
        f"{int((endpoint_df['fc_seed_count'] == 5).sum())}/100"
    )
    print()
    print("FC recurrence histogram (# seeds -> # endpoints):")
    for k in range(6):
        print(f"  {k} -> {recurrence_hist[str(k)]}")
    print()

    if len(fc_only) == 0:
        print("No strict FC endpoint occurred in any seed.")
    else:
        print("STRICT FC ENDPOINTS")
        cols = [
            "audit_id", "dataset_index", "target",
            "fc_seed_count", "visible_agreement_seed_count",
            "material_hidden_seed_count",
            "median_D_over_s", "median_H_over_s",
            "min_H_over_s", "max_H_over_s", "median_chi",
        ]
        print(
            fc_only[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print()
    print(f"Saved full endpoint table : {all_path.resolve()}")
    print(f"Saved FC-only table       : {fc_path.resolve()}")
    print("=" * 88)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        default="./e3_tabular_mlp/mlp_audits.csv",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
