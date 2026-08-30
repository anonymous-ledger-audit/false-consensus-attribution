#!/usr/bin/env python3
"""
E3a — Paired threshold robustness surface
=========================================

The 500 MLP rows correspond to 100 frozen endpoints x 5 independently trained
model seeds. This script evaluates the already-declared threshold grid at the
correct paired unit: the endpoint.

For each (kappa, tau) pair, an endpoint is counted as robustly satisfying the
false-consensus criterion if

    D/s <= kappa  and  H/s >= tau

in at least 3 of the 5 independently trained MLPs.

This is NOT used to choose a new threshold. It reports the entire prespecified
grid:

    kappa in {0.005, 0.01, 0.02, 0.05, 0.10}
    tau   in {0.01, 0.02, 0.05, 0.10, 0.20}

Run from the repository root:
    python ICLR/e3_tabular_mlp_threshold_surface.py

Input:
    ./e3_tabular_mlp/mlp_audits.csv

Outputs:
    ./e3_tabular_mlp/paired_threshold_surface.csv
    ./e3_tabular_mlp/paired_threshold_surface.json
    ./e3_tabular_mlp/paired_threshold_surface.pdf
    ./e3_tabular_mlp/paired_threshold_surface.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPECTED_SEEDS = [20260840, 20260841, 20260842, 20260843, 20260844]
EXPECTED_ENDPOINTS = 100

KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.10]
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20]

REQUIRED_SEEDS = 3


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run(args):
    input_path = Path(args.input_csv)
    out_dir = input_path.parent

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {
        "seed",
        "audit_id",
        "dataset_index",
        "D_over_s",
        "H_over_s",
        "resolved",
        "certification_pass",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    # Normalize booleans if CSV parser did not.
    for col in ["resolved", "certification_pass"]:
        if df[col].dtype != bool:
            df[col] = (
                df[col].astype(str).str.lower()
                .map({"true": True, "false": False})
            )

    # ------------------------------------------------------------------
    # Hard paired-design checks.
    # ------------------------------------------------------------------
    seeds = sorted(df["seed"].astype(int).unique().tolist())
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"Unexpected model seeds: {seeds}")

    if df["audit_id"].nunique() != EXPECTED_ENDPOINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_ENDPOINTS} unique endpoints, "
            f"found {df['audit_id'].nunique()}."
        )

    if len(df) != EXPECTED_ENDPOINTS * len(EXPECTED_SEEDS):
        raise RuntimeError(f"Expected 500 rows, found {len(df)}.")

    if not (df.groupby("audit_id").size() == len(EXPECTED_SEEDS)).all():
        raise RuntimeError("Some endpoints do not have exactly 5 seed rows.")

    if not (df.groupby("audit_id")["dataset_index"].nunique() == 1).all():
        raise RuntimeError("dataset_index changes within at least one audit_id.")

    if not df["resolved"].all():
        raise RuntimeError("At least one audit is unresolved.")
    if not df["certification_pass"].all():
        raise RuntimeError("At least one audit failed certification.")

    # ------------------------------------------------------------------
    # Evaluate the entire frozen grid.
    # ------------------------------------------------------------------
    rows = []
    robust_count_matrix = np.zeros((len(TAUS), len(KAPPAS)), dtype=int)
    at_least_one_matrix = np.zeros_like(robust_count_matrix)
    all_five_matrix = np.zeros_like(robust_count_matrix)

    for ti, tau in enumerate(TAUS):
        for ki, kappa in enumerate(KAPPAS):
            tmp = df.copy()
            tmp["event"] = (
                (tmp["D_over_s"] <= kappa)
                & (tmp["H_over_s"] >= tau)
            )

            recurrence = tmp.groupby("audit_id")["event"].sum()

            robust_count = int((recurrence >= REQUIRED_SEEDS).sum())
            at_least_one = int((recurrence >= 1).sum())
            all_five = int((recurrence == 5).sum())

            robust_count_matrix[ti, ki] = robust_count
            at_least_one_matrix[ti, ki] = at_least_one
            all_five_matrix[ti, ki] = all_five

            rows.append({
                "kappa_D_over_s_max": float(kappa),
                "tau_H_over_s_min": float(tau),
                "endpoints_event_in_at_least_1_of_5_seeds": at_least_one,
                "endpoints_event_in_at_least_3_of_5_seeds": robust_count,
                "endpoints_event_in_all_5_seeds": all_five,
                "majority_endpoint_rate": robust_count / EXPECTED_ENDPOINTS,
            })

    surface_df = pd.DataFrame(rows)

    csv_path = out_dir / "paired_threshold_surface.csv"
    json_path = out_dir / "paired_threshold_surface.json"
    pdf_path = out_dir / "paired_threshold_surface.pdf"
    png_path = out_dir / "paired_threshold_surface.png"

    surface_df.to_csv(csv_path, index=False)

    summary = {
        "n_unique_endpoints": EXPECTED_ENDPOINTS,
        "n_model_seeds": len(EXPECTED_SEEDS),
        "robust_endpoint_definition": (
            "criterion holds in at least 3 of the 5 independently trained models"
        ),
        "kappa_grid": KAPPAS,
        "tau_grid": TAUS,
        "majority_count_matrix_rows_tau_cols_kappa":
            robust_count_matrix.tolist(),
        "at_least_one_count_matrix_rows_tau_cols_kappa":
            at_least_one_matrix.tolist(),
        "all_five_count_matrix_rows_tau_cols_kappa":
            all_five_matrix.tolist(),
    }
    write_json(summary, json_path)

    # ------------------------------------------------------------------
    # Figure: primary majority-recurrence surface.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    im = ax.imshow(
        robust_count_matrix,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(KAPPAS)))
    ax.set_xticklabels([f"{x:g}" for x in KAPPAS])
    ax.set_yticks(np.arange(len(TAUS)))
    ax.set_yticklabels([f"{x:g}" for x in TAUS])

    ax.set_xlabel(r"aggregate-agreement tolerance $\kappa$ in $D/s \leq \kappa$")
    ax.set_ylabel(r"hidden-mass threshold $\tau$ in $H/s \geq \tau$")
    ax.set_title(
        "Endpoint-level robustness: criterion holds in at least 3/5 MLP fits"
    )

    for i in range(len(TAUS)):
        for j in range(len(KAPPAS)):
            ax.text(
                j,
                i,
                str(robust_count_matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
            )

    # Mark the original prespecified operating point (kappa=.02, tau=.05).
    orig_j = KAPPAS.index(0.02)
    orig_i = TAUS.index(0.05)
    ax.scatter(
        [orig_j],
        [orig_i],
        marker="s",
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("number of robust endpoints out of 100")

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Terminal report.
    # ------------------------------------------------------------------
    print()
    print("=" * 88)
    print("E3a — PAIRED THRESHOLD ROBUSTNESS SURFACE")
    print("=" * 88)
    print("Entry = # of 100 endpoints satisfying the criterion in >=3/5 MLP seeds.")
    print()
    print("Rows: tau = hidden-mass threshold H/s >=")
    print("Cols: kappa = visible-discrepancy tolerance D/s <=")
    print()

    header = "tau \\ kappa | " + " | ".join(f"{k:>7g}" for k in KAPPAS)
    print(header)
    print("-" * len(header))

    for i, tau in enumerate(TAUS):
        vals = " | ".join(f"{robust_count_matrix[i, j]:7d}" for j in range(len(KAPPAS)))
        print(f"{tau:>11g} | {vals}")

    print()
    print(
        "Original prespecified point "
        "(kappa=0.02, tau=0.05): "
        f"{robust_count_matrix[orig_i, orig_j]}/100 endpoints "
        "in >=3/5 seeds"
    )

    print()
    print("For context at the same original point:")
    print(
        f"  >=1/5 seeds : "
        f"{at_least_one_matrix[orig_i, orig_j]}/100"
    )
    print(
        f"  >=3/5 seeds : "
        f"{robust_count_matrix[orig_i, orig_j]}/100"
    )
    print(
        f"  5/5 seeds   : "
        f"{all_five_matrix[orig_i, orig_j]}/100"
    )

    print()
    print(f"CSV   : {csv_path.resolve()}")
    print(f"JSON  : {json_path.resolve()}")
    print(f"PDF   : {pdf_path.resolve()}")
    print(f"PNG   : {png_path.resolve()}")
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
