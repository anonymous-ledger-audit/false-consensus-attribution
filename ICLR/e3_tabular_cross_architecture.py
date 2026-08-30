#!/usr/bin/env python3
"""
E3a — Cross-architecture endpoint analysis
==========================================

Compares the same 100 frozen California Housing endpoints across:
  - Softplus MLP
  - Smooth CrossNet

Each endpoint has 5 independent model seeds per architecture.

Run from the repository root:
    python ICLR/e3_tabular_cross_architecture.py
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_SEEDS = [20260840, 20260841, 20260842, 20260843, 20260844]
EXPECTED_ENDPOINTS = 100
FC_KAPPA = 0.02
FC_TAU = 0.05


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_bool(df, col):
    if df[col].dtype == bool:
        return
    v = df[col].astype(str).str.lower().map({"true": True, "false": False})
    if v.isna().any():
        raise RuntimeError(f"Could not parse {col}")
    df[col] = v


def validate(df, name):
    required = {
        "seed","audit_id","dataset_index","target","D_over_s","H_over_s",
        "chi","resolved","certification_pass"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{name}: missing {sorted(missing)}")
    parse_bool(df, "resolved")
    parse_bool(df, "certification_pass")
    if sorted(df.seed.astype(int).unique().tolist()) != EXPECTED_SEEDS:
        raise RuntimeError(f"{name}: seed mismatch")
    if df.audit_id.nunique() != EXPECTED_ENDPOINTS or len(df) != 500:
        raise RuntimeError(f"{name}: expected 100 endpoints / 500 rows")
    if not (df.groupby("audit_id").size() == 5).all():
        raise RuntimeError(f"{name}: each endpoint must have 5 rows")
    if not (df.groupby("audit_id").dataset_index.nunique() == 1).all():
        raise RuntimeError(f"{name}: dataset_index varies within audit_id")
    if not (df.groupby("audit_id").target.nunique() == 1).all():
        raise RuntimeError(f"{name}: target varies within audit_id")
    if not df.resolved.all() or not df.certification_pass.all():
        raise RuntimeError(f"{name}: unresolved or uncertified audit")


def summarize_endpoints(df, prefix):
    d = df.copy()
    d["visible"] = d.D_over_s <= FC_KAPPA
    d["material"] = d.H_over_s >= FC_TAU
    d["fc"] = d.visible & d.material
    rows = []
    for aid, g in d.groupby("audit_id", sort=True):
        rows.append({
            "audit_id": int(aid),
            "dataset_index": int(g.dataset_index.iloc[0]),
            "target": float(g.target.iloc[0]),
            f"{prefix}_median_D_over_s": float(g.D_over_s.median()),
            f"{prefix}_median_H_over_s": float(g.H_over_s.median()),
            f"{prefix}_median_chi": float(g.chi.dropna().median()),
            f"{prefix}_visible_seed_count": int(g.visible.sum()),
            f"{prefix}_material_seed_count": int(g.material.sum()),
            f"{prefix}_fc_seed_count": int(g.fc.sum()),
        })
    return pd.DataFrame(rows)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0,1])


def spearman(x, y):
    rx = pd.Series(x).rank(method="average")
    ry = pd.Series(y).rank(method="average")
    return pearson(rx, ry)


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return float(len(a & b) / len(u)) if u else 1.0


def run(args):
    mlp_path = Path(args.mlp_csv)
    cross_path = Path(args.crossnet_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp = pd.read_csv(mlp_path)
    cross = pd.read_csv(cross_path)

    validate(mlp, "MLP")
    validate(cross, "CrossNet")

    mlp_ep = summarize_endpoints(mlp, "mlp")
    cross_ep = summarize_endpoints(cross, "crossnet")

    comp = mlp_ep.merge(
        cross_ep,
        on=["audit_id","dataset_index","target"],
        how="inner",
        validate="one_to_one",
    )
    if len(comp) != 100:
        raise RuntimeError("Architectures do not contain the same 100 endpoints.")

    h_p = pearson(comp.mlp_median_H_over_s, comp.crossnet_median_H_over_s)
    h_s = spearman(comp.mlp_median_H_over_s, comp.crossnet_median_H_over_s)
    d_p = pearson(comp.mlp_median_D_over_s, comp.crossnet_median_D_over_s)
    d_s = spearman(comp.mlp_median_D_over_s, comp.crossnet_median_D_over_s)
    chi_p = pearson(comp.mlp_median_chi, comp.crossnet_median_chi)
    chi_s = spearman(comp.mlp_median_chi, comp.crossnet_median_chi)

    mlp_top10 = set(comp.nlargest(10, "mlp_median_H_over_s").audit_id)
    cross_top10 = set(comp.nlargest(10, "crossnet_median_H_over_s").audit_id)
    mlp_top20 = set(comp.nlargest(20, "mlp_median_H_over_s").audit_id)
    cross_top20 = set(comp.nlargest(20, "crossnet_median_H_over_s").audit_id)

    shared_top10 = sorted(mlp_top10 & cross_top10)
    shared_top20 = sorted(mlp_top20 & cross_top20)

    mlp_fc_any = set(comp.loc[comp.mlp_fc_seed_count >= 1, "audit_id"])
    cross_fc_any = set(comp.loc[comp.crossnet_fc_seed_count >= 1, "audit_id"])
    shared_fc_any = sorted(mlp_fc_any & cross_fc_any)

    mlp_fc_majority = set(comp.loc[comp.mlp_fc_seed_count >= 3, "audit_id"])
    cross_fc_majority = set(comp.loc[comp.crossnet_fc_seed_count >= 3, "audit_id"])
    shared_fc_majority = sorted(mlp_fc_majority & cross_fc_majority)

    mlp_material_majority = set(
        comp.loc[comp.mlp_material_seed_count >= 3, "audit_id"]
    )
    cross_material_majority = set(
        comp.loc[comp.crossnet_material_seed_count >= 3, "audit_id"]
    )
    shared_material_majority = sorted(
        mlp_material_majority & cross_material_majority
    )

    comp["delta_H_mlp_minus_crossnet"] = (
        comp.mlp_median_H_over_s - comp.crossnet_median_H_over_s
    )
    comp["delta_D_mlp_minus_crossnet"] = (
        comp.mlp_median_D_over_s - comp.crossnet_median_D_over_s
    )
    comp["shared_fc_any"] = comp.audit_id.isin(shared_fc_any)

    comp_path = out_dir / "endpoint_comparison.csv"
    shared_path = out_dir / "shared_fc_endpoints.csv"
    comp.to_csv(comp_path, index=False)
    comp.loc[comp.shared_fc_any].to_csv(shared_path, index=False)

    summary = {
        "n_endpoints": 100,
        "n_seeds_per_architecture": 5,
        "continuous_association": {
            "median_H_over_s": {"pearson": h_p, "spearman": h_s},
            "median_D_over_s": {"pearson": d_p, "spearman": d_s},
            "median_chi": {"pearson": chi_p, "spearman": chi_s},
        },
        "high_H_overlap": {
            "top10_shared_count": len(shared_top10),
            "top10_jaccard": jaccard(mlp_top10, cross_top10),
            "top10_shared_audit_ids": shared_top10,
            "top20_shared_count": len(shared_top20),
            "top20_jaccard": jaccard(mlp_top20, cross_top20),
            "top20_shared_audit_ids": shared_top20,
        },
        "strict_fc_overlap": {
            "mlp_any_seed_count": len(mlp_fc_any),
            "crossnet_any_seed_count": len(cross_fc_any),
            "shared_any_seed_count": len(shared_fc_any),
            "shared_any_seed_audit_ids": shared_fc_any,
            "mlp_majority_count": len(mlp_fc_majority),
            "crossnet_majority_count": len(cross_fc_majority),
            "shared_majority_count": len(shared_fc_majority),
        },
        "material_hidden_majority_overlap": {
            "mlp_count": len(mlp_material_majority),
            "crossnet_count": len(cross_material_majority),
            "shared_count": len(shared_material_majority),
            "shared_audit_ids": shared_material_majority,
        },
        "architecture_contrast": {
            "fraction_mlp_H_greater": float(
                (comp.mlp_median_H_over_s > comp.crossnet_median_H_over_s).mean()
            ),
            "fraction_crossnet_D_lower": float(
                (comp.crossnet_median_D_over_s < comp.mlp_median_D_over_s).mean()
            ),
            "median_delta_H_mlp_minus_crossnet": float(
                comp.delta_H_mlp_minus_crossnet.median()
            ),
            "median_delta_D_mlp_minus_crossnet": float(
                comp.delta_D_mlp_minus_crossnet.median()
            ),
        },
    }
    write_json(summary, out_dir / "cross_architecture_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4))

    ax = axes[0]
    ax.scatter(comp.mlp_median_H_over_s, comp.crossnet_median_H_over_s, s=18, alpha=.65)
    lim = max(comp.mlp_median_H_over_s.max(), comp.crossnet_median_H_over_s.max())
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlabel(r"MLP median $H/s$")
    ax.set_ylabel(r"CrossNet median $H/s$")
    ax.set_title(f"(a) Hidden mass\nSpearman={h_s:.2f}", fontsize=10)

    ax = axes[1]
    ax.scatter(comp.mlp_median_D_over_s, comp.crossnet_median_D_over_s, s=18, alpha=.65)
    lim = max(comp.mlp_median_D_over_s.max(), comp.crossnet_median_D_over_s.max())
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlabel(r"MLP median $D/s$")
    ax.set_ylabel(r"CrossNet median $D/s$")
    ax.set_title(f"(b) Visible discrepancy\nSpearman={d_s:.2f}", fontsize=10)

    ax = axes[2]
    ax.scatter(comp.mlp_fc_seed_count, comp.crossnet_fc_seed_count, s=24, alpha=.65)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel("MLP strict-FC seed count")
    ax.set_ylabel("CrossNet strict-FC seed count")
    ax.set_title("(c) Endpoint recurrence", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "cross_architecture_three_panel.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cross_architecture_three_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 92)
    print("E3a — CROSS-ARCHITECTURE ENDPOINT ANALYSIS")
    print("=" * 92)
    print(f"Frozen endpoints compared              : {len(comp)}")
    print()
    print("Continuous endpoint association:")
    print(f"  median H/s: Pearson={h_p:.4f}, Spearman={h_s:.4f}")
    print(f"  median D/s: Pearson={d_p:.4f}, Spearman={d_s:.4f}")
    print(f"  median chi: Pearson={chi_p:.4f}, Spearman={chi_s:.4f}")
    print()
    print("High-H rank overlap:")
    print(f"  top 10: shared={len(shared_top10)}/10, Jaccard={jaccard(mlp_top10, cross_top10):.3f}")
    print(f"  top 20: shared={len(shared_top20)}/20, Jaccard={jaccard(mlp_top20, cross_top20):.3f}")
    print()
    print("Strict false-consensus overlap:")
    print(f"  any seed: MLP={len(mlp_fc_any)}, CrossNet={len(cross_fc_any)}, shared={len(shared_fc_any)}")
    print(f"  >=3/5 seeds: MLP={len(mlp_fc_majority)}, CrossNet={len(cross_fc_majority)}, shared={len(shared_fc_majority)}")
    print()
    print("Material hidden H/s >= 0.05 in >=3/5 seeds:")
    print(f"  MLP={len(mlp_material_majority)}, CrossNet={len(cross_material_majority)}, shared={len(shared_material_majority)}")
    print()
    print("Architecture contrast:")
    print(f"  fraction endpoints with MLP median H/s > CrossNet : {summary['architecture_contrast']['fraction_mlp_H_greater']:.3f}")
    print(f"  fraction endpoints with CrossNet median D/s < MLP : {summary['architecture_contrast']['fraction_crossnet_D_lower']:.3f}")
    print()

    if shared_fc_any:
        print("Endpoints strict-FC in at least one seed of BOTH architectures:")
        view = comp.loc[
            comp.audit_id.isin(shared_fc_any),
            [
                "audit_id","dataset_index",
                "mlp_fc_seed_count","crossnet_fc_seed_count",
                "mlp_median_D_over_s","mlp_median_H_over_s",
                "crossnet_median_D_over_s","crossnet_median_H_over_s",
            ]
        ].sort_values(
            ["crossnet_fc_seed_count","mlp_fc_seed_count"],
            ascending=False
        )
        print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        print("No endpoint has strict FC in both architectures.")

    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 92)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-csv", default="./e3_tabular_mlp/mlp_audits.csv")
    p.add_argument("--crossnet-csv", default="./e3_tabular_crossnet/crossnet_audits.csv")
    p.add_argument("--out-dir", default="./e3_tabular_cross_architecture")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
