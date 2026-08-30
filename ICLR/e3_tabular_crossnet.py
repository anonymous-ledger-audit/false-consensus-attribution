#!/usr/bin/env python3
"""
E3a — Smooth CrossNet fitted-model audit
========================================

Second substantive fitted nonlinear architecture for E3a.

The experiment reuses, unchanged:
    - the frozen California Housing split;
    - train-only standardization;
    - the frozen observed training baseline;
    - the same 100 held-out endpoints;
    - the same five model seeds;
    - the same exhaustive 2^8 endpoint game;
    - the same potwise BShap / straight-line IG ledger engine;
    - the same numerical certification;
    - the same descriptive threshold grid.

Scientific role
---------------
The Softplus MLP learns interactions implicitly. This model uses an explicit
Deep & Cross-style interaction tower, giving an architecture-level replication
with a substantially different inductive bias.

Cross layer:
    x_{l+1} = x_l + x_0 * (w_l^T x_l) + b_l

The cross representation is concatenated with a two-layer Softplus tower and
mapped to the scalar regression output.

Hyperparameters are selected by validation MSE only. No attribution quantity
is used for selection, early stopping, endpoint selection, or model acceptance.

Prespecified predictive pathology gate (same as the MLP stage):
    - finite predictions;
    - test R^2 >= 0.50;
    - maximum test prediction distance from the training-prediction median
      <= 10 training-output 5--95% ranges.

Primary descriptive false-consensus point:
    D/s <= 0.02 and H/s >= 0.05

Paired robustness grid:
    kappa in {0.005, 0.01, 0.02, 0.05, 0.10}
    tau   in {0.01, 0.02, 0.05, 0.10, 0.20}

Run from the repository root:
    python ICLR/e3_tabular_crossnet.py
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import e3_tabular_controls as ctl
import e3_tabular_mlp as mlp_audit


MODEL_SEEDS = ctl.MODEL_SEEDS
D = ctl.D

FC_KAPPA = 0.02
FC_TAU = 0.05

KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.10]
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20]

MIN_TEST_R2 = 0.50
MAX_TEST_EXTRAPOLATION_S = 10.0

# Small prespecified validation-only architecture grid.
# The explicit cross tower is capped at two layers to limit polynomial degree
# while retaining nontrivial higher-order interactions.
ARCH_GRID = [
    {"cross_layers": 1, "deep_width": 64,  "weight_decay": 1e-5},
    {"cross_layers": 1, "deep_width": 128, "weight_decay": 1e-5},
    {"cross_layers": 2, "deep_width": 64,  "weight_decay": 1e-5},
    {"cross_layers": 2, "deep_width": 128, "weight_decay": 1e-5},
]


class CrossLayer(nn.Module):
    """
    Original vector-form cross layer:
        x_{l+1} = x_l + x_0 (w_l^T x_l) + b_l
    """
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        nn.init.normal_(self.w, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        scalar = torch.sum(xl * self.w[None, :], dim=1, keepdim=True)
        return xl + x0 * scalar + self.b[None, :]


class SmoothCrossNet(nn.Module):
    """
    Explicit cross tower + smooth deep tower.

    The deep branch is intentionally simple and identical in depth across the
    prespecified grid; architecture search varies only cross depth and width.
    """
    def __init__(self, d: int, cross_layers: int, deep_width: int):
        super().__init__()

        self.cross = nn.ModuleList([
            CrossLayer(d) for _ in range(cross_layers)
        ])

        self.deep = nn.Sequential(
            nn.Linear(d, deep_width),
            nn.Softplus(),
            nn.Linear(deep_width, deep_width),
            nn.Softplus(),
        )

        self.out = nn.Linear(d + deep_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)

        xd = self.deep(x)
        z = torch.cat([xc, xd], dim=1)
        return self.out(z).squeeze(1)


class OriginalScaleWrapper(nn.Module):
    def __init__(self, base, y_mean, y_std):
        super().__init__()
        self.base = base
        self.register_buffer(
            "y_mean",
            torch.tensor(float(y_mean), dtype=torch.float64),
        )
        self.register_buffer(
            "y_std",
            torch.tensor(float(y_std), dtype=torch.float64),
        )

    def forward(self, x):
        return self.y_mean + self.y_std * self.base(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X, y, batch_size, seed):
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float64)),
        torch.from_numpy(np.asarray(y, dtype=np.float64)),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )


@torch.no_grad()
def predict_np(model, X, batch_size=4096):
    model.eval()
    X = np.asarray(X, dtype=np.float64)
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size])
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def fit_candidate(
    X_train,
    y_train_std,
    X_val,
    y_val_std,
    seed,
    spec,
    max_epochs,
    patience,
    batch_size,
    lr,
):
    set_seed(seed)

    model = SmoothCrossNet(
        d=D,
        cross_layers=spec["cross_layers"],
        deep_width=spec["deep_width"],
    ).double()

    loader = make_loader(X_train, y_train_std, batch_size, seed)

    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float64))
    yv = torch.from_numpy(np.asarray(y_val_std, dtype=np.float64))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=spec["weight_decay"],
    )
    mse = nn.MSELoss()

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(xb), yb)
            loss.backward()

            # Mild gradient clipping is a fixed optimizer safeguard, not an
            # attribution-dependent intervention.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(mse(model(Xv), yv).item())

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val


def regression_metrics(y_true, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def paired_surface(audit_df):
    """
    Endpoint-level robustness: count an endpoint when the criterion holds in
    at least 3 of the 5 independently trained CrossNet fits.
    """
    matrix = np.zeros((len(TAUS), len(KAPPAS)), dtype=int)
    one_matrix = np.zeros_like(matrix)
    five_matrix = np.zeros_like(matrix)

    for ti, tau in enumerate(TAUS):
        for ki, kappa in enumerate(KAPPAS):
            event = (
                (audit_df["D_over_s"] <= kappa)
                & (audit_df["H_over_s"] >= tau)
            )
            tmp = audit_df[["audit_id"]].copy()
            tmp["event"] = event.to_numpy()
            recurrence = tmp.groupby("audit_id")["event"].sum()

            matrix[ti, ki] = int((recurrence >= 3).sum())
            one_matrix[ti, ki] = int((recurrence >= 1).sum())
            five_matrix[ti, ki] = int((recurrence == 5).sum())

    return matrix, one_matrix, five_matrix


def run(args):
    torch.set_default_dtype(torch.float64)

    protocol_dir = Path(args.protocol_dir)
    controls_dir = Path(args.controls_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(
        as_frame=True,
        download_if_missing=True,
    )
    X_raw = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)

    protocol = ctl.verify_protocol(protocol_dir, X_raw, y)

    control_summary_path = controls_dir / "control_summary.json"
    if not control_summary_path.exists():
        raise FileNotFoundError(
            f"Missing fatal-control summary: {control_summary_path}"
        )

    control_summary = read_json(control_summary_path)
    if not control_summary.get("all_pass", False):
        raise RuntimeError("Fatal-control stage did not pass.")

    split = np.load(protocol_dir / "split_indices.npz")
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)
    test_idx = split["test_idx"].astype(np.int64)
    audit_idx = split["audit_dataset_idx"].astype(np.int64)
    baseline_idx = int(split["baseline_dataset_index"][0])

    prep = read_json(protocol_dir / "preprocessing.json")
    names = prep["feature_names"]

    mean = np.asarray(
        [prep["scaler_mean"][n] for n in names],
        dtype=np.float64,
    )
    scale = np.asarray(
        [prep["scaler_scale"][n] for n in names],
        dtype=np.float64,
    )

    X = (X_raw - mean[None, :]) / scale[None, :]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    baseline = X[baseline_idx].copy()
    X_audit = X[audit_idx]
    y_audit = y[audit_idx]

    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx], ddof=0))

    y_train_std = (y[train_idx] - y_mean) / y_std
    y_val_std = (y[val_idx] - y_mean) / y_std

    fit_rows = []
    audit_rows = []

    print()
    print("=" * 84)
    print("E3a — SMOOTH CROSSNET FITTED-MODEL AUDIT")
    print("=" * 84)
    print(f"Architecture grid               : {ARCH_GRID}")
    print(f"Frozen model seeds              : {MODEL_SEEDS}")
    print(f"Frozen audit endpoints          : {len(audit_idx)}")
    print(f"Primary FC point                : D/s <= {FC_KAPPA}, H/s >= {FC_TAU}")
    print(f"Predictive pathology gate       : test R2 >= {MIN_TEST_R2}")
    print()

    for seed in MODEL_SEEDS:
        print(f"[seed {seed}] validation-only architecture search")
        candidates = []

        for spec in ARCH_GRID:
            base, epoch, val_std_mse = fit_candidate(
                X_train=X_train,
                y_train_std=y_train_std,
                X_val=X_val,
                y_val_std=y_val_std,
                seed=seed,
                spec=spec,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
            )

            candidates.append({
                "spec": spec,
                "model": base,
                "epoch": int(epoch),
                "val_std_mse": float(val_std_mse),
            })

            print(
                f"    cross={spec['cross_layers']} "
                f"width={spec['deep_width']:3d} | "
                f"epoch={epoch:4d} | "
                f"val_std_MSE={val_std_mse:.6f}"
            )

        # Predictive validation loss only.
        candidates.sort(
            key=lambda z: (
                z["val_std_mse"],
                z["spec"]["cross_layers"],
                z["spec"]["deep_width"],
            )
        )
        chosen = candidates[0]
        spec = chosen["spec"]

        model = OriginalScaleWrapper(
            chosen["model"],
            y_mean=y_mean,
            y_std=y_std,
        ).double()
        model.eval()

        train_pred = predict_np(model, X_train)
        val_pred = predict_np(model, X_val)
        test_pred = predict_np(model, X_test)

        train_m = regression_metrics(y[train_idx], train_pred)
        val_m = regression_metrics(y[val_idx], val_pred)
        test_m = regression_metrics(y[test_idx], test_pred)

        output_scale = float(
            np.quantile(train_pred, 0.95)
            - np.quantile(train_pred, 0.05)
        )
        if output_scale <= 1e-12:
            raise RuntimeError("Degenerate CrossNet training-output scale.")

        train_med = float(np.median(train_pred))
        max_test_distance_s = float(
            np.max(np.abs(test_pred - train_med)) / output_scale
        )

        finite = bool(
            np.isfinite(train_pred).all()
            and np.isfinite(val_pred).all()
            and np.isfinite(test_pred).all()
        )

        gate = bool(
            finite
            and test_m["r2"] >= MIN_TEST_R2
            and max_test_distance_s <= MAX_TEST_EXTRAPOLATION_S
        )

        fit_rows.append({
            "family": "smooth_crossnet",
            "seed": seed,
            "selected_cross_layers": int(spec["cross_layers"]),
            "selected_deep_width": int(spec["deep_width"]),
            "best_epoch": int(chosen["epoch"]),
            "best_validation_mse_standardized_target":
                float(chosen["val_std_mse"]),
            "output_scale_s": output_scale,
            "train_r2": train_m["r2"],
            "val_r2": val_m["r2"],
            "test_r2": test_m["r2"],
            "train_rmse": train_m["rmse"],
            "val_rmse": val_m["rmse"],
            "test_rmse": test_m["rmse"],
            "max_test_prediction_distance_from_train_median_in_s":
                max_test_distance_s,
            "predictive_gate": gate,
        })

        print(
            f"  selected cross={spec['cross_layers']} "
            f"width={spec['deep_width']} | "
            f"val_R2={val_m['r2']:.4f} | "
            f"test_R2={test_m['r2']:.4f} | "
            f"s={output_scale:.4f} | gate={gate}"
        )

        if not gate:
            pd.DataFrame(fit_rows).to_csv(
                out_dir / "fit_metrics_partial.csv",
                index=False,
            )
            raise RuntimeError(
                f"CrossNet seed {seed} failed the prespecified predictive "
                "pathology gate. Attribution audit not interpreted."
            )

        torch.save(
            {
                "family": "smooth_crossnet",
                "seed": seed,
                "selected_cross_layers": int(spec["cross_layers"]),
                "selected_deep_width": int(spec["deep_width"]),
                "feature_names": names,
                "y_mean": y_mean,
                "y_std": y_std,
                "state_dict": model.state_dict(),
            },
            out_dir / f"crossnet_seed_{seed}.pt",
        )

        seed_rows = []

        for audit_id, (idx, endpoint, target) in enumerate(
            zip(audit_idx, X_audit, y_audit)
        ):
    # Reuse the detailed audit function from the MLP stage.
            result = mlp_audit.detailed_audit(
                model=model,
                baseline=baseline,
                endpoint=endpoint,
                output_scale=output_scale,
            )

            row = {
                "family": "smooth_crossnet",
                "seed": seed,
                "audit_id": int(audit_id),
                "dataset_index": int(idx),
                "target": float(target),
                "output_scale_s": output_scale,
                **result,
            }

            audit_rows.append(row)
            seed_rows.append(row)

            if (audit_id + 1) % 20 == 0:
                g = pd.DataFrame(seed_rows)
                print(
                    f"    audits {audit_id + 1:3d}/100 | "
                    f"median H/s={g['H_over_s'].median():.4f} | "
                    f"max H/s={g['H_over_s'].max():.4f} | "
                    f"FC={int(g['false_consensus_primary'].sum())}"
                )

    fit_df = pd.DataFrame(fit_rows)
    audit_df = pd.DataFrame(audit_rows)

    fit_df.to_csv(out_dir / "fit_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "crossnet_audits.csv", index=False)

    pass_flags = {
        "all_predictive_gates_pass": bool(fit_df["predictive_gate"].all()),
        "all_audits_resolved": bool(audit_df["resolved"].all()),
        "all_audits_certified": bool(
            audit_df["certification_pass"].all()
        ),
    }
    all_pass = bool(all(pass_flags.values()))

    seed_summary = []
    for seed in MODEL_SEEDS:
        g = audit_df[audit_df["seed"] == seed]
        seed_summary.append({
            "seed": int(seed),
            "median_D_over_s": float(g["D_over_s"].median()),
            "median_H_over_s": float(g["H_over_s"].median()),
            "q90_H_over_s": float(g["H_over_s"].quantile(0.90)),
            "max_H_over_s": float(g["H_over_s"].max()),
            "median_chi": float(g["chi"].dropna().median()),
            "false_consensus_count": int(
                g["false_consensus_primary"].sum()
            ),
            "false_consensus_rate": float(
                g["false_consensus_primary"].mean()
            ),
        })

    majority_matrix, one_matrix, five_matrix = paired_surface(audit_df)

    orig_i = TAUS.index(0.05)
    orig_j = KAPPAS.index(0.02)

    # Interaction-order anatomy.
    order_summary = {}
    for k in range(2, D + 1):
        col = f"R_order{k}_over_s"
        order_summary[str(k)] = {
            "median": float(audit_df[col].median()),
            "q90": float(audit_df[col].quantile(0.90)),
            "max": float(audit_df[col].max()),
        }

    summary = {
        "experiment": "E3a Smooth CrossNet fitted-model audit",
        "architecture_grid": ARCH_GRID,
        "model_seeds": MODEL_SEEDS,
        "n_audits": int(len(audit_df)),
        "primary_false_consensus_definition": {
            "D_over_s_max": FC_KAPPA,
            "H_over_s_min": FC_TAU,
        },
        "hard_pass_flags": pass_flags,
        "all_hard_checks_pass": all_pass,
        "overall": {
            "median_D_over_s": float(audit_df["D_over_s"].median()),
            "median_H_over_s": float(audit_df["H_over_s"].median()),
            "q90_H_over_s": float(
                audit_df["H_over_s"].quantile(0.90)
            ),
            "max_H_over_s": float(audit_df["H_over_s"].max()),
            "median_chi": float(audit_df["chi"].dropna().median()),
            "false_consensus_count": int(
                audit_df["false_consensus_primary"].sum()
            ),
            "false_consensus_rate": float(
                audit_df["false_consensus_primary"].mean()
            ),
            "median_signed_cosine": float(
                audit_df["signed_cosine"].dropna().median()
            ),
            "median_spearman_abs": float(
                audit_df["spearman_abs"].dropna().median()
            ),
            "median_top3_jaccard": float(
                audit_df["top3_jaccard"].dropna().median()
            ),
            "median_H_over_M": float(
                audit_df["H_over_M"].dropna().median()
            ),
        },
        "by_seed": seed_summary,
        "redistribution_by_interaction_order": order_summary,
        "paired_threshold_surface": {
            "kappas": KAPPAS,
            "taus": TAUS,
            "majority_3_of_5_matrix_rows_tau_cols_kappa":
                majority_matrix.tolist(),
            "at_least_1_of_5_matrix_rows_tau_cols_kappa":
                one_matrix.tolist(),
            "all_5_of_5_matrix_rows_tau_cols_kappa":
                five_matrix.tolist(),
            "original_point_majority_count":
                int(majority_matrix[orig_i, orig_j]),
            "original_point_at_least_one_count":
                int(one_matrix[orig_i, orig_j]),
            "original_point_all_five_count":
                int(five_matrix[orig_i, orig_j]),
        },
    }

    write_json(summary, out_dir / "crossnet_summary.json")

    # ------------------------------------------------------------------
    # Diagnostic figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.2))

    ax = axes[0]
    ax.scatter(
        audit_df["D_over_s"],
        audit_df["H_over_s"],
        s=12,
        alpha=0.55,
    )
    ax.axvline(FC_KAPPA, linestyle="--", linewidth=1.0)
    ax.axhline(FC_TAU, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$D/s$")
    ax.set_ylabel(r"$H/s$")
    ax.set_title("(a) CrossNet regime map", fontsize=10)

    ax = axes[1]
    vals = [
        audit_df.loc[
            audit_df["seed"] == seed,
            "H_over_s"
        ].to_numpy()
        for seed in MODEL_SEEDS
    ]
    ax.boxplot(
        vals,
        tick_labels=[str(s)[-2:] for s in MODEL_SEEDS],
        showfliers=False,
    )
    ax.set_xlabel("seed")
    ax.set_ylabel(r"$H/s$")
    ax.set_title("(b) Seed stability", fontsize=10)

    ax = axes[2]
    im = ax.imshow(
        majority_matrix,
        origin="lower",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(KAPPAS)))
    ax.set_xticklabels([f"{x:g}" for x in KAPPAS])
    ax.set_yticks(np.arange(len(TAUS)))
    ax.set_yticklabels([f"{x:g}" for x in TAUS])

    for i in range(len(TAUS)):
        for j in range(len(KAPPAS)):
            ax.text(
                j,
                i,
                str(majority_matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.scatter(
        [orig_j],
        [orig_i],
        marker="s",
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
    )
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\tau$")
    ax.set_title("(c) Robust endpoints (>=3/5)", fontsize=10)

    fig.tight_layout()
    fig.savefig(
        out_dir / "crossnet_diagnostic_three_panel.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "crossnet_diagnostic_three_panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------
    print()
    print("=" * 84)
    print("E3a — SMOOTH CROSSNET SUMMARY")
    print("=" * 84)

    print("Predictive fits:")
    for row in fit_rows:
        print(
            f"  seed={row['seed']} | "
            f"cross={row['selected_cross_layers']} "
            f"width={row['selected_deep_width']} | "
            f"val_R2={row['val_r2']:.4f} | "
            f"test_R2={row['test_r2']:.4f} | "
            f"test_RMSE={row['test_rmse']:.4f}"
        )

    print()
    print(f"Audits                        : {len(audit_df)}")
    print(f"Resolved rate                 : {audit_df['resolved'].mean():.3f}")
    print(
        f"Certification pass rate       : "
        f"{audit_df['certification_pass'].mean():.3f}"
    )
    print(
        f"Median D/s                    : "
        f"{audit_df['D_over_s'].median():.4f}"
    )
    print(
        f"Median H/s                    : "
        f"{audit_df['H_over_s'].median():.4f}"
    )
    print(
        f"90th percentile H/s           : "
        f"{audit_df['H_over_s'].quantile(0.90):.4f}"
    )
    print(
        f"Maximum H/s                   : "
        f"{audit_df['H_over_s'].max():.4f}"
    )
    print(
        f"Median hidden fraction chi    : "
        f"{audit_df['chi'].dropna().median():.4f}"
    )
    print(
        f"Primary FC row count          : "
        f"{int(audit_df['false_consensus_primary'].sum())}/{len(audit_df)}"
    )

    print()
    print("By seed:")
    for row in seed_summary:
        print(
            f"  {row['seed']}: "
            f"median H/s={row['median_H_over_s']:.4f}, "
            f"q90={row['q90_H_over_s']:.4f}, "
            f"max={row['max_H_over_s']:.4f}, "
            f"FC={row['false_consensus_count']}/100"
        )

    print()
    print("Paired threshold surface")
    print("Entry = # endpoints satisfying criterion in >=3/5 CrossNet fits.")
    print()

    header = "tau \\ kappa | " + " | ".join(
        f"{k:>7g}" for k in KAPPAS
    )
    print(header)
    print("-" * len(header))

    for i, tau in enumerate(TAUS):
        vals = " | ".join(
            f"{majority_matrix[i, j]:7d}"
            for j in range(len(KAPPAS))
        )
        print(f"{tau:>11g} | {vals}")

    print()
    print(
        f"Original point (0.02, 0.05): "
        f">=1/5={one_matrix[orig_i, orig_j]}/100, "
        f">=3/5={majority_matrix[orig_i, orig_j]}/100, "
        f"5/5={five_matrix[orig_i, orig_j]}/100"
    )

    print()
    print("Hard checks:")
    for name, value in pass_flags.items():
        print(f"  {name}: {value}")
    print(f"  all_hard_checks_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 84)

    if not all_pass:
        raise RuntimeError(
            "CrossNet stage failed a hard predictive/numerical check."
        )


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-dir",
        default="./e3_tabular_protocol",
    )
    p.add_argument(
        "--controls-dir",
        default="./e3_tabular_controls",
    )
    p.add_argument(
        "--out-dir",
        default="./e3_tabular_crossnet",
    )
    p.add_argument("--max-epochs", type=int, default=700)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
