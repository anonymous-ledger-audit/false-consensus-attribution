#!/usr/bin/env python3
"""
E3a — Fatal controls + reusable exhaustive ledger audit
========================================================

This stage evaluates the two null-control families for the frozen California
Housing experiment.

It MUST consume the already-frozen protocol directory created by
e3_tabular_protocol.py. It does not resplit data, change the baseline, or
resample audit endpoints.

Controls
--------
1) Additive Softplus GAM
       f(x) = b + sum_i f_i(x_i)
   There are no cross-feature interaction pots, so the transfer ledger must be
   zero up to numerical precision.

2) Quadratic control
       f(x) = b + linear + squares + pairwise products
   Genuine pairwise interactions are present, but every pure pair pot is
   bilinear in the slider variables. Baseline Shapley and straight-line IG
   therefore split every pair 1/2--1/2, so the transfer ledger must again be
   zero up to numerical precision.

Both models are trained in float64 on the frozen train split and selected only
by validation predictive loss. The audit engine then evaluates the complete
2^8 endpoint game and every anchored pot for each frozen audit endpoint.

Locked model seeds
------------------
20260840, 20260841, 20260842, 20260843, 20260844

Numerical certification
-----------------------
Adaptive Gauss-Legendre orders: 16 -> 32 -> 64 -> 128 -> 256.
An audit is resolved only if nested potwise IG allocations stabilize and all
reconstruction/completeness checks pass.

No substantive false-consensus threshold is used in this control experiment.
The only question is whether the mathematically required null ledger is
recovered.

Outputs
-------
e3_tabular_controls/
    additive_seed_<seed>.pt
    quadratic_seed_<seed>.pt
    fit_metrics.csv
    control_audits.csv
    control_summary.json
    control_null_diagnostic.pdf
    control_null_diagnostic.png

Run
---
python ICLR/e3_tabular_controls.py --protocol-dir ./e3_tabular_protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# =====================================================================
# Frozen constants
# =====================================================================

MODEL_SEEDS = [20260840, 20260841, 20260842, 20260843, 20260844]
QUAD_ORDERS = [16, 32, 64, 128, 256]
D = 8
N_MASKS = 1 << D
FULL_MASK = N_MASKS - 1


# =====================================================================
# Utilities
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    payload = (
        str(arr.dtype).encode("utf-8")
        + str(arr.shape).encode("utf-8")
        + arr.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mobius_transform(values: np.ndarray, d: int = D) -> np.ndarray:
    """
    Fast subset Möbius transform.

    Input shape can be (2^d,) or (2^d, p).
    Output m satisfies:
        m[S] = sum_{T subset S} (-1)^{|S|-|T|} values[T].
    """
    out = np.array(values, dtype=np.float64, copy=True)
    for bit in range(d):
        step = 1 << bit
        for mask in range(1 << d):
            if mask & step:
                out[mask] -= out[mask ^ step]
    return out


def exact_shapley_from_game(game: np.ndarray, d: int = D) -> np.ndarray:
    phi = np.zeros(d, dtype=np.float64)
    for i in range(d):
        bit = 1 << i
        for mask in range(1 << d):
            if mask & bit:
                continue
            s = int(mask.bit_count())
            w = 1.0 / (d * math.comb(d - 1, s))
            phi[i] += w * (game[mask | bit] - game[mask])
    return phi


def mask_matrix(d: int = D) -> np.ndarray:
    M = np.zeros((1 << d, d), dtype=np.float64)
    for mask in range(1 << d):
        for i in range(d):
            M[mask, i] = 1.0 if mask & (1 << i) else 0.0
    return M


MASKS = mask_matrix(D)
MASK_POPCOUNT = np.asarray([m.bit_count() for m in range(N_MASKS)], dtype=int)


# =====================================================================
# Models
# =====================================================================

class AdditiveSoftplusGAM(nn.Module):
    """
    Sum of eight independent 1-D smooth subnetworks.
    No cross-feature interactions are possible by construction.
    """
    def __init__(self, d: int = D, width: int = 16):
        super().__init__()
        self.d = d
        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, width),
                nn.Softplus(),
                nn.Linear(width, width),
                nn.Softplus(),
                nn.Linear(width, 1),
            )
            for _ in range(d)
        ])
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i, net in enumerate(self.parts):
            pieces.append(net(x[:, i:i+1]))
        return torch.stack(pieces, dim=0).sum(dim=0).squeeze(1) + self.bias


class QuadraticControl(nn.Module):
    """
    Exact degree-2 polynomial in the standardized input coordinates.

    Random initialization + stochastic fitting gives seed-specific fitted
    coefficients, while the structural null T=0 remains exact for every fit.
    """
    def __init__(self, d: int = D):
        super().__init__()
        self.d = d
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.linear = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        self.square = nn.Parameter(torch.zeros(d, dtype=torch.float64))

        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs.append((i, j))
        self.pairs = pairs
        self.pair_coeff = nn.Parameter(
            torch.zeros(len(pairs), dtype=torch.float64)
        )

        with torch.no_grad():
            self.linear.normal_(0.0, 0.02)
            self.square.normal_(0.0, 0.02)
            self.pair_coeff.normal_(0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bias + x @ self.linear + (x * x) @ self.square
        if self.pairs:
            pair_terms = torch.stack(
                [x[:, i] * x[:, j] for i, j in self.pairs],
                dim=1,
            )
            y = y + pair_terms @ self.pair_coeff
        return y


class OriginalScaleWrapper(nn.Module):
    """
    Base model predicts standardized y; wrapper returns original target units.
    """
    def __init__(self, base: nn.Module, y_mean: float, y_std: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.y_mean + self.y_std * self.base(x)


# =====================================================================
# Training
# =====================================================================

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float64)),
        torch.from_numpy(np.asarray(y, dtype=np.float64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def predict_np(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    X = np.asarray(X, dtype=np.float64)
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size])
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train_std: np.ndarray,
    X_val: np.ndarray,
    y_val_std: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
):
    set_seed(seed)
    model = model.double()

    loader = make_loader(
        X_train,
        y_train_std,
        batch_size=batch_size,
        seed=seed,
    )

    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float64))
    yv = torch.from_numpy(np.asarray(y_val_std, dtype=np.float64))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = float(loss_fn(val_pred, yv).item())

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


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


# =====================================================================
# Exhaustive endpoint game + ledger audit
# =====================================================================

@torch.no_grad()
def evaluate_endpoint_game(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
) -> np.ndarray:
    """
    F(t) = f(b + t*(x-b)) - f(b), evaluated at all Boolean corners t=1^S.
    """
    model.eval()

    b = np.asarray(baseline, dtype=np.float64)
    x = np.asarray(endpoint, dtype=np.float64)
    delta = x - b

    corners = b[None, :] + MASKS * delta[None, :]

    xb = torch.from_numpy(corners)
    bb = torch.from_numpy(b[None, :])

    vals = model(xb).cpu().numpy()
    base_val = float(model(bb).item())

    return vals - base_val


def integrated_mask_gradients(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    order: int,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    For every mask T and feature i, compute

        J[T,i] = int_0^1 dF(t^T)/dt_i ds,

    with the derivative set to zero when i notin T because t^T clamps
    coordinates outside T.

    Output shape: (2^d, d).
    """
    model.eval()

    b = np.asarray(baseline, dtype=np.float64)
    x = np.asarray(endpoint, dtype=np.float64)
    delta = x - b

    nodes, weights = np.polynomial.legendre.leggauss(order)
    s = 0.5 * (nodes + 1.0)
    w = 0.5 * weights

    # shape: (order, n_masks, d)
    t_all = s[:, None, None] * MASKS[None, :, :]
    t_flat = t_all.reshape(-1, D)

    grad_flat = np.zeros_like(t_flat, dtype=np.float64)

    b_t = torch.from_numpy(b)
    delta_t = torch.from_numpy(delta)

    for start in range(0, len(t_flat), chunk_size):
        stop = min(len(t_flat), start + chunk_size)

        t = torch.tensor(
            t_flat[start:stop],
            dtype=torch.float64,
            requires_grad=True,
        )

        x_batch = b_t[None, :] + t * delta_t[None, :]
        y_batch = model(x_batch)

        grad_t = torch.autograd.grad(
            y_batch.sum(),
            t,
            create_graph=False,
            retain_graph=False,
        )[0]

        grad_flat[start:stop] = grad_t.detach().cpu().numpy()

    grad = grad_flat.reshape(order, N_MASKS, D)

    # Coordinates outside mask T are clamped in F(t^T), hence derivative 0.
    grad *= MASKS[None, :, :]

    # Integrate in s.
    J = np.tensordot(w, grad, axes=(0, 0))
    return np.asarray(J, dtype=np.float64)


def potwise_ig_from_J(J: np.ndarray) -> np.ndarray:
    """
    Möbius-transform J[:,i] separately for each feature.

    Returns L with shape (d, 2^d), where L[i,u] is the potwise path allocation.
    """
    L = np.zeros((D, N_MASKS), dtype=np.float64)

    for i in range(D):
        h = mobius_transform(J[:, i], d=D)

        # The anchored pot u cannot allocate to feature i if i notin u.
        member = MASKS[:, i] > 0.5
        h[~member] = 0.0

        L[i, :] = h

    return L


def equal_split_pot_allocation(phi_pots: np.ndarray) -> np.ndarray:
    E = np.zeros((D, N_MASKS), dtype=np.float64)
    for mask in range(1, N_MASKS):
        k = MASK_POPCOUNT[mask]
        for i in range(D):
            if mask & (1 << i):
                E[i, mask] = phi_pots[mask] / k
    return E


@torch.no_grad()
def interior_mobius_reconstruction_error(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    s_values=(0.25, 0.5, 0.75),
) -> float:
    """
    Numerically verify anchored Möbius reconstruction at several interior
    diagonal points.
    """
    model.eval()

    b = np.asarray(baseline, dtype=np.float64)
    x = np.asarray(endpoint, dtype=np.float64)
    delta = x - b

    b_t = torch.from_numpy(b[None, :])
    base_val = float(model(b_t).item())

    max_err = 0.0

    for s in s_values:
        t = float(s) * MASKS
        pts = b[None, :] + t * delta[None, :]
        vals = model(torch.from_numpy(pts)).cpu().numpy() - base_val

        pots = mobius_transform(vals, d=D)

        # Sum of all anchored pots at the full diagonal point equals F(s*1).
        reconstructed = float(np.sum(pots[1:]))
        full_value = float(vals[FULL_MASK])

        max_err = max(max_err, abs(reconstructed - full_value))

    return float(max_err)


def audit_one(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    output_scale: float,
    quad_orders=QUAD_ORDERS,
) -> dict:
    game = evaluate_endpoint_game(model, baseline, endpoint)
    phi_pots = mobius_transform(game, d=D)

    E = equal_split_pot_allocation(phi_pots)
    bshap_from_pots = E.sum(axis=1)
    bshap_direct = exact_shapley_from_game(game, d=D)

    # Numerical tolerance is model-scale aware and remains far below every
    # substantive threshold in the frozen E3 contract.
    num_tol = max(1e-10, 1e-8 * max(float(output_scale), 1.0))

    previous_L = None
    resolved = False
    used_order = None
    nested_error = float("inf")
    J = None
    L = None

    for order in quad_orders:
        J_now = integrated_mask_gradients(
            model,
            baseline,
            endpoint,
            order=order,
        )
        L_now = potwise_ig_from_J(J_now)

        if previous_L is not None:
            nested_error = float(np.max(np.abs(L_now - previous_L)))
            if nested_error <= num_tol:
                J = J_now
                L = L_now
                used_order = order
                resolved = True
                break

        previous_L = L_now
        J = J_now
        L = L_now
        used_order = order

    if J is None or L is None:
        raise RuntimeError("Internal audit error: no quadrature result.")

    direct_ig = J[FULL_MASK, :]
    potwise_ig_sum = L.sum(axis=1)

    T = L - E

    R = 0.5 * float(np.abs(T[:, 1:]).sum())
    margins = T[:, 1:].sum(axis=1)
    D_vis = 0.5 * float(np.abs(margins).sum())
    H = R - D_vis
    chi = float(H / R) if R > num_tol else np.nan

    # Certification checks.
    pot_conservation_error = 0.0
    for mask in range(1, N_MASKS):
        err = abs(float(L[:, mask].sum()) - float(phi_pots[mask]))
        pot_conservation_error = max(pot_conservation_error, err)

    bshap_reconstruction_error = float(
        np.max(np.abs(bshap_from_pots - bshap_direct))
    )
    ig_reconstruction_error = float(
        np.max(np.abs(potwise_ig_sum - direct_ig))
    )

    endpoint_change = float(game[FULL_MASK])
    bshap_completeness_error = abs(float(bshap_direct.sum()) - endpoint_change)
    ig_completeness_error = abs(float(direct_ig.sum()) - endpoint_change)

    # Row margins of T must equal the final vector discrepancy.
    margin_gap_error = float(
        np.max(np.abs(margins - (direct_ig - bshap_direct)))
    )

    interior_error = interior_mobius_reconstruction_error(
        model,
        baseline,
        endpoint,
    )

    all_cert_errors = [
        pot_conservation_error,
        bshap_reconstruction_error,
        ig_reconstruction_error,
        bshap_completeness_error,
        ig_completeness_error,
        margin_gap_error,
        interior_error,
    ]

    certification_pass = bool(
        resolved
        and nested_error <= num_tol
        and max(all_cert_errors) <= 10.0 * num_tol
    )

    return {
        "resolved": bool(resolved),
        "certification_pass": certification_pass,
        "quadrature_order": int(used_order),
        "numerical_tolerance": float(num_tol),
        "nested_quadrature_error": float(nested_error),

        "endpoint_change": endpoint_change,
        "R": R,
        "D": D_vis,
        "H": H,
        "chi": chi,
        "R_over_s": float(R / output_scale),
        "D_over_s": float(D_vis / output_scale),
        "H_over_s": float(H / output_scale),

        "max_abs_transfer_entry": float(np.max(np.abs(T))),
        "pot_conservation_error": float(pot_conservation_error),
        "bshap_reconstruction_error": bshap_reconstruction_error,
        "ig_reconstruction_error": ig_reconstruction_error,
        "bshap_completeness_error": float(bshap_completeness_error),
        "ig_completeness_error": float(ig_completeness_error),
        "margin_gap_error": margin_gap_error,
        "interior_mobius_reconstruction_error": interior_error,
    }


# =====================================================================
# Main experiment
# =====================================================================

def verify_protocol(protocol_dir: Path, X: np.ndarray, y: np.ndarray):
    required = [
        "split_indices.npz",
        "audit_indices.csv",
        "baseline.csv",
        "preprocessing.json",
        "protocol.json",
        "manifest_sha256.json",
    ]
    for name in required:
        if not (protocol_dir / name).exists():
            raise FileNotFoundError(
                f"Missing frozen protocol artifact: {protocol_dir / name}"
            )

    protocol = read_json(protocol_dir / "protocol.json")
    manifest = read_json(protocol_dir / "manifest_sha256.json")

    # Verify artifact hashes from the freeze.
    for name, expected_hash in manifest.items():
        actual = sha256_file(protocol_dir / name)
        if actual != expected_hash:
            raise RuntimeError(
                f"Frozen protocol file changed after freeze: {name}\n"
                f"expected {expected_hash}\nactual   {actual}"
            )

    # Verify dataset content against freeze.
    if sha256_array(X) != protocol["dataset"]["X_sha256"]:
        raise RuntimeError("California Housing feature matrix hash mismatch.")
    if sha256_array(y) != protocol["dataset"]["y_sha256"]:
        raise RuntimeError("California Housing target vector hash mismatch.")

    if not protocol.get("all_pass", False):
        raise RuntimeError("Frozen protocol itself did not pass its checks.")

    frozen_seeds = protocol.get("future_model_seeds", [])
    if frozen_seeds != MODEL_SEEDS:
        raise RuntimeError(
            f"Model seeds differ from frozen protocol: {frozen_seeds}"
        )

    return protocol


def run(args):
    torch.set_default_dtype(torch.float64)

    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(
        as_frame=True,
        download_if_missing=True,
    )
    X_raw = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)

    protocol = verify_protocol(protocol_dir, X_raw, y)

    split = np.load(protocol_dir / "split_indices.npz")
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)
    test_idx = split["test_idx"].astype(np.int64)
    audit_idx = split["audit_dataset_idx"].astype(np.int64)
    baseline_idx = int(split["baseline_dataset_index"][0])

    prep = read_json(protocol_dir / "preprocessing.json")
    feature_names = prep["feature_names"]

    mean = np.asarray(
        [prep["scaler_mean"][name] for name in feature_names],
        dtype=np.float64,
    )
    scale = np.asarray(
        [prep["scaler_scale"][name] for name in feature_names],
        dtype=np.float64,
    )

    X = (X_raw - mean[None, :]) / scale[None, :]

    # Re-verify that the frozen baseline/audit indices have not drifted.
    baseline = X[baseline_idx].copy()
    X_audit = X[audit_idx]
    y_audit = y[audit_idx]

    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx], ddof=0))
    y_train_std = (y[train_idx] - y_mean) / y_std
    y_val_std = (y[val_idx] - y_mean) / y_std

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    fit_rows = []
    audit_rows = []

    model_specs = [
        {
            "family": "additive",
            "factory": lambda: AdditiveSoftplusGAM(D, width=args.additive_width),
            "lr": args.additive_lr,
            "weight_decay": args.additive_weight_decay,
        },
        {
            "family": "quadratic",
            "factory": lambda: QuadraticControl(D),
            "lr": args.quadratic_lr,
            "weight_decay": args.quadratic_weight_decay,
        },
    ]

    print()
    print("=" * 78)
    print("E3a — FATAL CONTROLS + EXHAUSTIVE LEDGER AUDIT")
    print("=" * 78)
    print(f"Protocol directory               : {protocol_dir.resolve()}")
    print(f"Output directory                 : {out_dir.resolve()}")
    print(f"Frozen audit endpoints           : {len(audit_idx)}")
    print(f"Model seeds                      : {MODEL_SEEDS}")
    print(f"Quadrature orders                : {QUAD_ORDERS}")
    print()

    for spec in model_specs:
        family = spec["family"]

        for seed in MODEL_SEEDS:
            print(f"[fit] {family:10s} seed={seed}")

            base_model = spec["factory"]().double()

            base_model, best_epoch, best_val_std_mse = fit_model(
                base_model,
                X_train,
                y_train_std,
                X_val,
                y_val_std,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=spec["lr"],
                weight_decay=spec["weight_decay"],
            )

            model = OriginalScaleWrapper(
                base_model,
                y_mean=y_mean,
                y_std=y_std,
            ).double()
            model.eval()

            train_pred = predict_np(model, X_train)
            val_pred = predict_np(model, X_val)
            test_pred = predict_np(model, X_test)

            output_scale = float(
                np.quantile(train_pred, 0.95)
                - np.quantile(train_pred, 0.05)
            )
            if output_scale <= 1e-12:
                raise RuntimeError(
                    f"{family} seed={seed}: degenerate training-output scale."
                )

            train_m = regression_metrics(y[train_idx], train_pred)
            val_m = regression_metrics(y[val_idx], val_pred)
            test_m = regression_metrics(y[test_idx], test_pred)

            fit_rows.append({
                "family": family,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_mse_standardized_target": best_val_std_mse,
                "output_scale_s": output_scale,
                "train_rmse": train_m["rmse"],
                "train_mae": train_m["mae"],
                "train_r2": train_m["r2"],
                "val_rmse": val_m["rmse"],
                "val_mae": val_m["mae"],
                "val_r2": val_m["r2"],
                "test_rmse": test_m["rmse"],
                "test_mae": test_m["mae"],
                "test_r2": test_m["r2"],
            })

            checkpoint = {
                "family": family,
                "seed": seed,
                "feature_names": feature_names,
                "y_mean": y_mean,
                "y_std": y_std,
                "state_dict": model.state_dict(),
            }
            torch.save(
                checkpoint,
                out_dir / f"{family}_seed_{seed}.pt",
            )

            print(
                f"      epoch={best_epoch:4d} "
                f"val_RMSE={val_m['rmse']:.4f} "
                f"test_R2={test_m['r2']:.4f} "
                f"s={output_scale:.4f}"
            )

            for audit_id, (idx, endpoint, target) in enumerate(
                zip(audit_idx, X_audit, y_audit)
            ):
                result = audit_one(
                    model,
                    baseline=baseline,
                    endpoint=endpoint,
                    output_scale=output_scale,
                )

                audit_rows.append({
                    "family": family,
                    "seed": seed,
                    "audit_id": audit_id,
                    "dataset_index": int(idx),
                    "target": float(target),
                    **result,
                })

                if (audit_id + 1) % 20 == 0:
                    print(
                        f"      audits {audit_id + 1:3d}/100 | "
                        f"max current H/s={max(r['H_over_s'] for r in audit_rows if r['family']==family and r['seed']==seed):.3e}"
                    )

    fit_df = pd.DataFrame(fit_rows)
    audit_df = pd.DataFrame(audit_rows)

    fit_df.to_csv(out_dir / "fit_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "control_audits.csv", index=False)

    # -----------------------------------------------------------------
    # Fatal-control pass conditions
    # -----------------------------------------------------------------
    family_summary = {}

    for family in ["additive", "quadratic"]:
        g = audit_df[audit_df["family"] == family]

        family_summary[family] = {
            "n_audits": int(len(g)),
            "resolved_rate": float(g["resolved"].mean()),
            "certification_pass_rate": float(g["certification_pass"].mean()),
            "max_R": float(g["R"].max()),
            "max_D": float(g["D"].max()),
            "max_H": float(g["H"].max()),
            "max_R_over_s": float(g["R_over_s"].max()),
            "max_D_over_s": float(g["D_over_s"].max()),
            "max_H_over_s": float(g["H_over_s"].max()),
            "max_abs_transfer_entry": float(g["max_abs_transfer_entry"].max()),
            "max_pot_conservation_error": float(g["pot_conservation_error"].max()),
            "max_bshap_reconstruction_error": float(g["bshap_reconstruction_error"].max()),
            "max_ig_reconstruction_error": float(g["ig_reconstruction_error"].max()),
            "max_bshap_completeness_error": float(g["bshap_completeness_error"].max()),
            "max_ig_completeness_error": float(g["ig_completeness_error"].max()),
            "max_margin_gap_error": float(g["margin_gap_error"].max()),
            "max_interior_mobius_reconstruction_error": float(
                g["interior_mobius_reconstruction_error"].max()
            ),
            "max_nested_quadrature_error": float(g["nested_quadrature_error"].max()),
            "quadrature_order_counts": {
                str(int(k)): int(v)
                for k, v in g["quadrature_order"].value_counts().sort_index().items()
            },
        }

    # The fatal-control null threshold is numerical, not substantive.
    # H/s must remain many orders below the materiality thresholds (e.g. 0.05).
    control_null_tol_normalized = 1e-7

    pass_flags = {
        "all_audits_resolved": bool(audit_df["resolved"].all()),
        "all_audits_certified": bool(audit_df["certification_pass"].all()),
        "additive_null": bool(
            family_summary["additive"]["max_H_over_s"]
            <= control_null_tol_normalized
        ),
        "quadratic_null": bool(
            family_summary["quadratic"]["max_H_over_s"]
            <= control_null_tol_normalized
        ),
    }
    all_pass = bool(all(pass_flags.values()))

    summary = {
        "experiment": "E3a fatal controls",
        "protocol_dir": str(protocol_dir.resolve()),
        "protocol_manifest_sha256": sha256_file(
            protocol_dir / "manifest_sha256.json"
        ),
        "model_seeds": MODEL_SEEDS,
        "quadrature_orders": QUAD_ORDERS,
        "n_fits": int(len(fit_df)),
        "n_control_audits": int(len(audit_df)),
        "control_null_tolerance_H_over_s": control_null_tol_normalized,
        "families": family_summary,
        "pass_flags": pass_flags,
        "all_pass": all_pass,
    }

    write_json(
        summary,
        out_dir / "control_summary.json",
    )

    # -----------------------------------------------------------------
    # Control-stage diagnostic figure.
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.8, 3.3))

    plot_df = audit_df.copy()
    eps = 1e-18
    data_plot = [
        np.log10(
            np.maximum(
                plot_df.loc[plot_df["family"] == fam, "H_over_s"].to_numpy(),
                eps,
            )
        )
        for fam in ["additive", "quadratic"]
    ]

    ax.boxplot(
        data_plot,
        labels=["Additive", "Quadratic"],
        showfliers=False,
        widths=0.5,
    )
    ax.axhline(
        np.log10(control_null_tol_normalized),
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_ylabel(r"$\log_{10}\!\left(\max(H/s,10^{-18})\right)$")
    ax.set_title("Fatal controls: hidden redistribution remains numerical")
    ax.text(
        0.03,
        0.96,
        rf"null threshold $H/s\leq {control_null_tol_normalized:.0e}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "control_null_diagnostic.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "control_null_diagnostic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("E3a — FATAL CONTROL SUMMARY")
    print("=" * 78)

    for family in ["additive", "quadratic"]:
        s = family_summary[family]
        print(f"{family.upper()}")
        print(f"  audits                           : {s['n_audits']}")
        print(f"  resolved rate                    : {s['resolved_rate']:.3f}")
        print(f"  certification pass rate          : {s['certification_pass_rate']:.3f}")
        print(f"  max H/s                          : {s['max_H_over_s']:.3e}")
        print(f"  max R/s                          : {s['max_R_over_s']:.3e}")
        print(f"  max D/s                          : {s['max_D_over_s']:.3e}")
        print(f"  max |transfer entry|             : {s['max_abs_transfer_entry']:.3e}")
        print(f"  max pot conservation error       : {s['max_pot_conservation_error']:.3e}")
        print(f"  max IG reconstruction error      : {s['max_ig_reconstruction_error']:.3e}")
        print(f"  quadrature order counts          : {s['quadrature_order_counts']}")
        print()

    print("Pre-registered control checks:")
    for name, value in pass_flags.items():
        print(f"  {name}: {value}")
    print(f"  all_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 78)

    if not all_pass:
        raise RuntimeError(
            "Fatal-control stage FAILED. Do not proceed to fitted-model "
            "audits until the control failure is understood."
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="Fit E3a fatal controls and run complete 8-feature ledger audits."
    )

    p.add_argument(
        "--protocol-dir",
        default="./e3_tabular_protocol",
        help="Frozen output directory from e3_tabular_protocol.py",
    )
    p.add_argument(
        "--out-dir",
        default="./e3_tabular_controls",
    )

    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)

    p.add_argument("--additive-width", type=int, default=16)
    p.add_argument("--additive-lr", type=float, default=2e-3)
    p.add_argument("--additive-weight-decay", type=float, default=1e-5)

    p.add_argument("--quadratic-lr", type=float, default=3e-3)
    p.add_argument("--quadratic-weight-decay", type=float, default=1e-4)

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
