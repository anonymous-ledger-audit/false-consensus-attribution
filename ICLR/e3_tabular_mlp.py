#!/usr/bin/env python3
"""
E3a — Softplus MLP fitted-model audit
=====================================

Uses the frozen California Housing protocol and the already-passed Additive +
Quadratic fatal controls. Data split, preprocessing, baseline, audit endpoints,
model seeds, explanation contract, and exhaustive ledger machinery are reused
unchanged.

No attribution quantity is used for model selection, early stopping, endpoint
selection, or predictive acceptance. Hyperparameters are selected by validation
MSE only.

Prespecified predictive pathology gate:
    - finite train/val/test predictions;
    - test R^2 >= 0.50;
    - no test prediction farther than 10 train-output 5--95% ranges from the
      training prediction median.

Primary descriptive false-consensus point:
    D/s <= 0.02 and H/s >= 0.05.

Run from the repository root:
    python ICLR/e3_tabular_mlp.py
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

MODEL_SEEDS = ctl.MODEL_SEEDS
D = ctl.D
N_MASKS = ctl.N_MASKS
FULL_MASK = ctl.FULL_MASK
MASK_POPCOUNT = ctl.MASK_POPCOUNT
QUAD_ORDERS = ctl.QUAD_ORDERS

FC_KAPPA = 0.02
FC_TAU = 0.05

ARCH_GRID = [
    {"width": 64, "depth": 2, "weight_decay": 1e-5},
    {"width": 64, "depth": 3, "weight_decay": 1e-5},
    {"width": 128, "depth": 2, "weight_decay": 1e-5},
    {"width": 128, "depth": 3, "weight_decay": 1e-5},
]

MIN_TEST_R2 = 0.50
MAX_TEST_EXTRAPOLATION_S = 10.0


class SoftplusMLP(nn.Module):
    def __init__(self, d: int, width: int, depth: int):
        super().__init__()
        layers = []
        in_dim = d
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.Softplus()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class OriginalScaleWrapper(nn.Module):
    def __init__(self, base, y_mean, y_std):
        super().__init__()
        self.base = base
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float64))
        self.register_buffer("y_std", torch.tensor(float(y_std), dtype=torch.float64))

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
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


@torch.no_grad()
def predict_np(model, X, batch_size=4096):
    model.eval()
    X = np.asarray(X, dtype=np.float64)
    out = []
    for start in range(0, len(X), batch_size):
        out.append(model(torch.from_numpy(X[start:start+batch_size])).cpu().numpy())
    return np.concatenate(out)


def fit_candidate(X_train, y_train_std, X_val, y_val_std, seed, spec,
                  max_epochs, patience, batch_size, lr):
    set_seed(seed)
    model = SoftplusMLP(D, spec["width"], spec["depth"]).double()

    loader = make_loader(X_train, y_train_std, batch_size, seed)
    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float64))
    yv = torch.from_numpy(np.asarray(y_val_std, dtype=np.float64))

    opt = torch.optim.AdamW(
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
            opt.zero_grad(set_to_none=True)
            loss = mse(model(xb), yb)
            loss.backward()
            opt.step()

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


def signed_cosine(a, b, eps=1e-15):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def spearman_abs(a, b):
    ra = pd.Series(np.abs(a)).rank(method="average").to_numpy()
    rb = pd.Series(np.abs(b)).rank(method="average").to_numpy()
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def top3_jaccard(a, b):
    ia = set(np.argsort(-np.abs(a))[:3].tolist())
    ib = set(np.argsort(-np.abs(b))[:3].tolist())
    u = ia | ib
    return float(len(ia & ib) / len(u)) if u else 1.0


def detailed_audit(model, baseline, endpoint, output_scale):
    game = ctl.evaluate_endpoint_game(model, baseline, endpoint)
    phi_pots = ctl.mobius_transform(game, d=D)

    E = ctl.equal_split_pot_allocation(phi_pots)
    bshap_from_pots = E.sum(axis=1)
    bshap_direct = ctl.exact_shapley_from_game(game, d=D)

    num_tol = max(1e-10, 1e-8 * max(float(output_scale), 1.0))

    previous_L = None
    resolved = False
    nested_error = float("inf")
    used_order = None
    J = L = None

    for order in QUAD_ORDERS:
        J_now = ctl.integrated_mask_gradients(model, baseline, endpoint, order=order)
        L_now = ctl.potwise_ig_from_J(J_now)

        if previous_L is not None:
            nested_error = float(np.max(np.abs(L_now - previous_L)))
            if nested_error <= num_tol:
                J, L, used_order, resolved = J_now, L_now, order, True
                break

        previous_L = L_now
        J, L, used_order = J_now, L_now, order

    direct_ig = J[FULL_MASK, :]
    potwise_ig_sum = L.sum(axis=1)
    T = L - E

    R = 0.5 * float(np.abs(T[:, 1:]).sum())
    margins = T[:, 1:].sum(axis=1)
    D_vis = 0.5 * float(np.abs(margins).sum())
    H = R - D_vis
    chi = float(H / R) if R > num_tol else np.nan

    R_by_order = {}
    for k in range(2, D + 1):
        masks_k = np.flatnonzero(MASK_POPCOUNT == k)
        R_by_order[k] = 0.5 * float(np.abs(T[:, masks_k]).sum())

    interaction_masks = np.flatnonzero(MASK_POPCOUNT >= 2)
    M = float(np.abs(phi_pots[interaction_masks]).sum())

    pot_conservation_error = 0.0
    for mask in range(1, N_MASKS):
        pot_conservation_error = max(
            pot_conservation_error,
            abs(float(L[:, mask].sum()) - float(phi_pots[mask])),
        )

    bshap_reconstruction_error = float(np.max(np.abs(bshap_from_pots - bshap_direct)))
    ig_reconstruction_error = float(np.max(np.abs(potwise_ig_sum - direct_ig)))
    endpoint_change = float(game[FULL_MASK])
    bshap_completeness_error = abs(float(bshap_direct.sum()) - endpoint_change)
    ig_completeness_error = abs(float(direct_ig.sum()) - endpoint_change)
    margin_gap_error = float(np.max(np.abs(margins - (direct_ig - bshap_direct))))
    interior_error = ctl.interior_mobius_reconstruction_error(model, baseline, endpoint)

    cert_errors = [
        pot_conservation_error,
        bshap_reconstruction_error,
        ig_reconstruction_error,
        bshap_completeness_error,
        ig_completeness_error,
        margin_gap_error,
        interior_error,
    ]
    certification_pass = bool(
        resolved and nested_error <= num_tol and max(cert_errors) <= 10.0 * num_tol
    )

    row = {
        "resolved": bool(resolved),
        "certification_pass": certification_pass,
        "quadrature_order": int(used_order),
        "nested_quadrature_error": float(nested_error),
        "endpoint_change": endpoint_change,
        "R": R, "D": D_vis, "H": H, "chi": chi,
        "R_over_s": float(R / output_scale),
        "D_over_s": float(D_vis / output_scale),
        "H_over_s": float(H / output_scale),
        "interaction_pot_mass_M": M,
        "R_over_M": float(R / M) if M > num_tol else np.nan,
        "H_over_M": float(H / M) if M > num_tol else np.nan,
        "signed_cosine": signed_cosine(bshap_direct, direct_ig),
        "spearman_abs": spearman_abs(bshap_direct, direct_ig),
        "top3_jaccard": top3_jaccard(bshap_direct, direct_ig),
        "false_consensus_primary": bool(
            D_vis / output_scale <= FC_KAPPA and H / output_scale >= FC_TAU
        ),
        "max_abs_transfer_entry": float(np.max(np.abs(T))),
        "pot_conservation_error": pot_conservation_error,
        "bshap_reconstruction_error": bshap_reconstruction_error,
        "ig_reconstruction_error": ig_reconstruction_error,
        "bshap_completeness_error": bshap_completeness_error,
        "ig_completeness_error": ig_completeness_error,
        "margin_gap_error": margin_gap_error,
        "interior_mobius_reconstruction_error": interior_error,
    }

    for k in range(2, D + 1):
        row[f"R_order{k}"] = float(R_by_order[k])
        row[f"R_order{k}_over_s"] = float(R_by_order[k] / output_scale)

    for i in range(D):
        row[f"bshap_{i}"] = float(bshap_direct[i])
        row[f"ig_{i}"] = float(direct_ig[i])

    return row


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run(args):
    torch.set_default_dtype(torch.float64)

    protocol_dir = Path(args.protocol_dir)
    controls_dir = Path(args.controls_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(as_frame=True, download_if_missing=True)
    X_raw = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)

    protocol = ctl.verify_protocol(protocol_dir, X_raw, y)

    control_summary = read_json(controls_dir / "control_summary.json")
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
    mean = np.array([prep["scaler_mean"][n] for n in names], dtype=np.float64)
    scale = np.array([prep["scaler_scale"][n] for n in names], dtype=np.float64)
    X = (X_raw - mean[None, :]) / scale[None, :]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    baseline = X[baseline_idx].copy()
    X_audit, y_audit = X[audit_idx], y[audit_idx]

    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx], ddof=0))
    y_train_std = (y[train_idx] - y_mean) / y_std
    y_val_std = (y[val_idx] - y_mean) / y_std

    fit_rows, audit_rows = [], []

    print()
    print("=" * 80)
    print("E3a — SOFTPLUS MLP FITTED-MODEL AUDIT")
    print("=" * 80)
    print(f"Architecture grid              : {ARCH_GRID}")
    print(f"Frozen model seeds            : {MODEL_SEEDS}")
    print(f"Frozen audit endpoints        : {len(audit_idx)}")
    print(f"Primary FC point              : D/s <= {FC_KAPPA}, H/s >= {FC_TAU}")
    print()

    for seed in MODEL_SEEDS:
        print(f"[seed {seed}] validation-only architecture search")
        candidates = []

        for spec in ARCH_GRID:
            base, epoch, val_std_mse = fit_candidate(
                X_train, y_train_std, X_val, y_val_std,
                seed, spec, args.max_epochs, args.patience,
                args.batch_size, args.lr,
            )
            candidates.append({
                "spec": spec, "model": base,
                "epoch": epoch, "val_std_mse": val_std_mse,
            })
            print(
                f"    width={spec['width']:3d} depth={spec['depth']} | "
                f"epoch={epoch:4d} | val_std_MSE={val_std_mse:.6f}"
            )

        candidates.sort(
            key=lambda z: (z["val_std_mse"], z["spec"]["depth"], z["spec"]["width"])
        )
        chosen = candidates[0]
        spec = chosen["spec"]

        model = OriginalScaleWrapper(chosen["model"], y_mean, y_std).double()
        model.eval()

        train_pred = predict_np(model, X_train)
        val_pred = predict_np(model, X_val)
        test_pred = predict_np(model, X_test)

        train_m = regression_metrics(y[train_idx], train_pred)
        val_m = regression_metrics(y[val_idx], val_pred)
        test_m = regression_metrics(y[test_idx], test_pred)

        output_scale = float(
            np.quantile(train_pred, 0.95) - np.quantile(train_pred, 0.05)
        )
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
            "family": "softplus_mlp",
            "seed": seed,
            "selected_width": spec["width"],
            "selected_depth": spec["depth"],
            "best_epoch": chosen["epoch"],
            "output_scale_s": output_scale,
            "train_r2": train_m["r2"], "val_r2": val_m["r2"], "test_r2": test_m["r2"],
            "train_rmse": train_m["rmse"], "val_rmse": val_m["rmse"], "test_rmse": test_m["rmse"],
            "max_test_prediction_distance_from_train_median_in_s": max_test_distance_s,
            "predictive_gate": gate,
        })

        print(
            f"  selected width={spec['width']} depth={spec['depth']} | "
            f"val_R2={val_m['r2']:.4f} | test_R2={test_m['r2']:.4f} | "
            f"s={output_scale:.4f} | gate={gate}"
        )

        if not gate:
            pd.DataFrame(fit_rows).to_csv(out_dir / "fit_metrics_partial.csv", index=False)
            raise RuntimeError(
                f"Seed {seed} failed the prespecified predictive pathology gate; "
                "attribution audits were not interpreted."
            )

        torch.save({
            "family": "softplus_mlp",
            "seed": seed,
            "selected_width": spec["width"],
            "selected_depth": spec["depth"],
            "feature_names": names,
            "y_mean": y_mean,
            "y_std": y_std,
            "state_dict": model.state_dict(),
        }, out_dir / f"mlp_seed_{seed}.pt")

        seed_rows = []
        for audit_id, (idx, endpoint, target) in enumerate(zip(audit_idx, X_audit, y_audit)):
            result = detailed_audit(model, baseline, endpoint, output_scale)
            row = {
                "family": "softplus_mlp",
                "seed": seed,
                "audit_id": audit_id,
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
                    f"    audits {audit_id+1:3d}/100 | "
                    f"median H/s={g['H_over_s'].median():.4f} | "
                    f"max H/s={g['H_over_s'].max():.4f} | "
                    f"FC={int(g['false_consensus_primary'].sum())}"
                )

    fit_df = pd.DataFrame(fit_rows)
    audit_df = pd.DataFrame(audit_rows)

    fit_df.to_csv(out_dir / "fit_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "mlp_audits.csv", index=False)

    pass_flags = {
        "all_predictive_gates_pass": bool(fit_df["predictive_gate"].all()),
        "all_audits_resolved": bool(audit_df["resolved"].all()),
        "all_audits_certified": bool(audit_df["certification_pass"].all()),
    }
    all_pass = bool(all(pass_flags.values()))

    seed_summary = []
    for seed in MODEL_SEEDS:
        g = audit_df[audit_df["seed"] == seed]
        seed_summary.append({
            "seed": seed,
            "median_D_over_s": float(g["D_over_s"].median()),
            "median_H_over_s": float(g["H_over_s"].median()),
            "q90_H_over_s": float(g["H_over_s"].quantile(0.90)),
            "max_H_over_s": float(g["H_over_s"].max()),
            "median_chi": float(g["chi"].dropna().median()),
            "false_consensus_count": int(g["false_consensus_primary"].sum()),
            "false_consensus_rate": float(g["false_consensus_primary"].mean()),
        })

    kappas = [0.005, 0.01, 0.02, 0.05, 0.10]
    taus = [0.01, 0.02, 0.05, 0.10, 0.20]
    sensitivity = {
        str(tau): {
            str(kappa): float(
                ((audit_df["D_over_s"] <= kappa) & (audit_df["H_over_s"] >= tau)).mean()
            )
            for kappa in kappas
        }
        for tau in taus
    }

    order_summary = {
        str(k): {
            "median": float(audit_df[f"R_order{k}_over_s"].median()),
            "q90": float(audit_df[f"R_order{k}_over_s"].quantile(0.90)),
            "max": float(audit_df[f"R_order{k}_over_s"].max()),
        }
        for k in range(2, D + 1)
    }

    summary = {
        "experiment": "E3a Softplus MLP fitted-model audit",
        "model_seeds": MODEL_SEEDS,
        "architecture_grid": ARCH_GRID,
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
            "q90_H_over_s": float(audit_df["H_over_s"].quantile(0.90)),
            "max_H_over_s": float(audit_df["H_over_s"].max()),
            "median_chi": float(audit_df["chi"].dropna().median()),
            "false_consensus_count": int(audit_df["false_consensus_primary"].sum()),
            "false_consensus_rate": float(audit_df["false_consensus_primary"].mean()),
            "median_signed_cosine": float(audit_df["signed_cosine"].dropna().median()),
            "median_spearman_abs": float(audit_df["spearman_abs"].dropna().median()),
            "median_top3_jaccard": float(audit_df["top3_jaccard"].dropna().median()),
            "median_H_over_M": float(audit_df["H_over_M"].dropna().median()),
        },
        "by_seed": seed_summary,
        "redistribution_by_interaction_order": order_summary,
        "threshold_sensitivity_rate": sensitivity,
    }
    write_json(summary, out_dir / "mlp_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.2))

    ax = axes[0]
    ax.scatter(audit_df["D_over_s"], audit_df["H_over_s"], s=12, alpha=0.55)
    ax.axvline(FC_KAPPA, linestyle="--", linewidth=1.0)
    ax.axhline(FC_TAU, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$D/s$")
    ax.set_ylabel(r"$H/s$")
    ax.set_title("(a) MLP regime map", fontsize=10)

    ax = axes[1]
    vals = [
        audit_df.loc[audit_df["seed"] == seed, "H_over_s"].to_numpy()
        for seed in MODEL_SEEDS
    ]
    ax.boxplot(vals, tick_labels=[str(s)[-2:] for s in MODEL_SEEDS], showfliers=False)
    ax.set_xlabel("seed")
    ax.set_ylabel(r"$H/s$")
    ax.set_title("(b) Seed stability", fontsize=10)

    ax = axes[2]
    med = [float(audit_df[f"R_order{k}_over_s"].median()) for k in range(2, D + 1)]
    ax.bar(np.arange(2, D + 1), med)
    ax.set_xlabel("interaction order")
    ax.set_ylabel(r"median $R_k/s$")
    ax.set_title("(c) Redistribution anatomy", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "mlp_diagnostic_three_panel.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "mlp_diagnostic_three_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 80)
    print("E3a — SOFTPLUS MLP SUMMARY")
    print("=" * 80)
    print("Predictive fits:")
    for row in fit_rows:
        print(
            f"  seed={row['seed']} | width={row['selected_width']} depth={row['selected_depth']} | "
            f"val_R2={row['val_r2']:.4f} | test_R2={row['test_r2']:.4f} | "
            f"test_RMSE={row['test_rmse']:.4f}"
        )
    print()
    print(f"Audits                        : {len(audit_df)}")
    print(f"Resolved rate                 : {audit_df['resolved'].mean():.3f}")
    print(f"Certification pass rate       : {audit_df['certification_pass'].mean():.3f}")
    print(f"Median D/s                    : {audit_df['D_over_s'].median():.4f}")
    print(f"Median H/s                    : {audit_df['H_over_s'].median():.4f}")
    print(f"90th percentile H/s           : {audit_df['H_over_s'].quantile(0.90):.4f}")
    print(f"Maximum H/s                   : {audit_df['H_over_s'].max():.4f}")
    print(f"Median hidden fraction chi    : {audit_df['chi'].dropna().median():.4f}")
    print(
        f"Primary false-consensus count : "
        f"{int(audit_df['false_consensus_primary'].sum())}/{len(audit_df)}"
    )
    print(f"Primary false-consensus rate  : {audit_df['false_consensus_primary'].mean():.3f}")
    print()
    print("By seed:")
    for row in seed_summary:
        print(
            f"  {row['seed']}: median H/s={row['median_H_over_s']:.4f}, "
            f"q90={row['q90_H_over_s']:.4f}, max={row['max_H_over_s']:.4f}, "
            f"FC={row['false_consensus_count']}/100"
        )
    print()
    print("Hard checks:")
    for name, value in pass_flags.items():
        print(f"  {name}: {value}")
    print(f"  all_hard_checks_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 80)

    if not all_pass:
        raise RuntimeError("MLP stage failed a hard predictive/numerical check.")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol-dir", default="./e3_tabular_protocol")
    p.add_argument("--controls-dir", default="./e3_tabular_controls")
    p.add_argument("--out-dir", default="./e3_tabular_mlp")
    p.add_argument("--max-epochs", type=int, default=700)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
