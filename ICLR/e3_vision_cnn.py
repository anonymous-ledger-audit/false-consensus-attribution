#!/usr/bin/env python3
"""
E3b — Softplus CNN fitted-model exhaustive region-ledger audit
================================================================

Final fitted-model stage of the appendix-only Fashion-MNIST experiment.

This script consumes, verifies, and never modifies:
    ./e3_vision_protocol/
    ./e3_vision_controls/

It changes only the fitted model family.  The split, train-only normalization,
observed training-image baseline, 100 class-balanced official-test endpoints,
eight fixed 2 x 4 image regions, five model seeds, true-class centered-logit
scalar, straight region-slider path, all 256 Boolean mosaics, complete Moebius
game, potwise Baseline Shapley / IG ledger, scale, quadrature orders, numerical
certificates, and descriptive threshold grid are inherited unchanged.

The CNN architecture and optimization contract were frozen in V1:

    Conv(1,32,3,pad=1) -> Softplus -> AvgPool(2)
    Conv(32,64,3,pad=1) -> Softplus -> AvgPool(2)
    Linear(64*7*7,128) -> Softplus -> Linear(128,10)

Training uses float32 for practical CPU runtime.  The validation-selected
weights are cast to float64 before output-scale calculation and every audit.
No attribution quantity affects training, early stopping, endpoint selection,
or model acceptance.

The frozen predictive pathology gate is evaluated before auditing:
    * all train/validation/test logits are finite;
    * official-test accuracy is at least 0.85;
    * no test true-class centered logit is farther than 10 training-output
      5--95% ranges from the training centered-logit median.

Run from the repository root:
    python ICLR/e3_vision_cnn.py

Outputs:
    ./e3_vision_cnn/
        cnn_seed_<seed>.pt
        fit_metrics.csv
        cnn_audits.csv
        endpoint_stability.csv
        class_summary.csv
        cnn_summary.json
        cnn_diagnostic_four_panel.pdf
        cnn_diagnostic_four_panel.png
        cnn_manifest_sha256.json

The output directory must be absent or empty.  This is a substantial CPU run:
five CNN fits and 500 complete, float64, 256-corner audits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

import e3_vision_controls as ctl


# =============================================================================
# Quantities already frozen by the protocol
# =============================================================================

D = ctl.D
N_MASKS = ctl.N_MASKS
FULL_MASK = ctl.FULL_MASK
N_CLASSES = ctl.N_CLASSES
MODEL_SEEDS = ctl.MODEL_SEEDS
QUAD_ORDERS = ctl.QUAD_ORDERS
MASK_POPCOUNT = ctl.MASK_POPCOUNT

FC_KAPPA = 0.02
FC_TAU = 0.05
KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.10]
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20]
MAJORITY_SEEDS = 3

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# =============================================================================
# Model and float32 training
# =============================================================================


class SoftplusCNN(nn.Module):
    """Exactly the architecture frozen in e3_vision_protocol v1.0."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Softplus(beta=1.0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Softplus(beta=1.0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.Softplus(beta=1.0),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(images, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def batched_nll(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 1024,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        x = torch.from_numpy(np.asarray(images[start:stop], dtype=np.float32))
        y = torch.from_numpy(np.asarray(labels[start:stop], dtype=np.int64))
        loss_sum = nn.functional.cross_entropy(model(x), y, reduction="sum")
        total += float(loss_sum.item())
        n += int(stop - start)
    return total / float(n)


def fit_cnn(
    model: nn.Module,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, int, float]:
    set_seed(seed)
    model = model.float()
    loader = make_loader(train_images, train_labels, batch_size, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CNN training loss.")
            loss.backward()
            optimizer.step()

        val_nll = batched_nll(model, val_images, val_labels)
        if not np.isfinite(val_nll):
            raise RuntimeError("Non-finite CNN validation loss.")

        if val_nll < best_val - 1e-10:
            best_val = float(val_nll)
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"      epoch={epoch:2d} val_NLL={val_nll:.5f} "
                f"best={best_val:.5f} stale={stale}"
            )

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val


@torch.no_grad()
def predict_logits_np(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    parameter = next(model.parameters())
    dtype = np.float64 if parameter.dtype == torch.float64 else np.float32
    outputs: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = torch.from_numpy(np.asarray(images[start:stop], dtype=dtype))
        outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


# =============================================================================
# Detailed version of the already-validated exhaustive ledger audit
# =============================================================================


def signed_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= eps or norm_b <= eps:
        return np.nan
    return float(np.dot(a, b) / (norm_a * norm_b))


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = pd.Series(np.abs(a)).rank(method="average").to_numpy()
    rank_b = pd.Series(np.abs(b)).rank(method="average").to_numpy()
    if np.std(rank_a) <= 0.0 or np.std(rank_b) <= 0.0:
        return np.nan
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def top3_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    selected_a = set(np.argsort(-np.abs(a))[:3].tolist())
    selected_b = set(np.argsort(-np.abs(b))[:3].tolist())
    union = selected_a | selected_b
    return float(len(selected_a & selected_b) / len(union)) if union else 1.0


def detailed_audit(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    true_class: int,
    region_masks: torch.Tensor,
    output_scale: float,
    chunk_size: int,
) -> dict[str, Any]:
    # Identical endpoint game, Moebius transform, equal-split allocation,
    # adaptive quadrature, and certificates used by the passed controls.
    game = ctl.evaluate_endpoint_game(
        model,
        baseline,
        endpoint,
        true_class,
        region_masks,
    )
    pot_values = ctl.mobius_transform(game, d=D)
    equal_split = ctl.equal_split_pot_allocation(pot_values)
    bshap_from_pots = equal_split.sum(axis=1)
    bshap_direct = ctl.exact_shapley_from_game(game, d=D)

    numerical_tolerance = max(1e-10, 1e-8 * max(float(output_scale), 1.0))
    previous_allocation = None
    resolved = False
    nested_error = float("inf")
    used_order = QUAD_ORDERS[-1]
    integrated_masks = None
    path_allocation = None

    for order in QUAD_ORDERS:
        integrated_now = ctl.integrated_mask_gradients(
            model,
            baseline,
            endpoint,
            true_class,
            region_masks,
            order=order,
            chunk_size=chunk_size,
        )
        allocation_now = ctl.potwise_ig_from_integrated_masks(integrated_now)

        if previous_allocation is not None:
            nested_error = float(
                np.max(np.abs(allocation_now - previous_allocation))
            )
            if nested_error <= numerical_tolerance:
                integrated_masks = integrated_now
                path_allocation = allocation_now
                used_order = int(order)
                resolved = True
                break

        previous_allocation = allocation_now
        integrated_masks = integrated_now
        path_allocation = allocation_now
        used_order = int(order)

    if integrated_masks is None or path_allocation is None:
        raise RuntimeError("CNN audit produced no quadrature allocation.")

    direct_ig = integrated_masks[FULL_MASK]
    potwise_ig_sum = path_allocation.sum(axis=1)
    transfer = path_allocation - equal_split

    gross = 0.5 * float(np.abs(transfer[:, 1:]).sum())
    margins = transfer[:, 1:].sum(axis=1)
    visible = 0.5 * float(np.abs(margins).sum())
    hidden = gross - visible
    concealed_fraction = (
        float(hidden / gross) if gross > numerical_tolerance else np.nan
    )

    redistribution_by_order: dict[int, float] = {}
    for order in range(2, D + 1):
        masks_at_order = np.flatnonzero(MASK_POPCOUNT == order)
        redistribution_by_order[order] = 0.5 * float(
            np.abs(transfer[:, masks_at_order]).sum()
        )

    interaction_masks = np.flatnonzero(MASK_POPCOUNT >= 2)
    interaction_pot_mass = float(np.abs(pot_values[interaction_masks]).sum())

    pot_conservation_error = 0.0
    for mask in range(1, N_MASKS):
        current = abs(
            float(path_allocation[:, mask].sum())
            - float(pot_values[mask])
        )
        pot_conservation_error = max(pot_conservation_error, current)

    bshap_reconstruction_error = float(
        np.max(np.abs(bshap_from_pots - bshap_direct))
    )
    ig_reconstruction_error = float(
        np.max(np.abs(potwise_ig_sum - direct_ig))
    )
    endpoint_change = float(game[FULL_MASK])
    bshap_completeness_error = abs(float(bshap_direct.sum()) - endpoint_change)
    ig_completeness_error = abs(float(direct_ig.sum()) - endpoint_change)
    margin_gap_error = float(
        np.max(np.abs(margins - (direct_ig - bshap_direct)))
    )
    interior_error = ctl.interior_mobius_reconstruction_error(
        model,
        baseline,
        endpoint,
        true_class,
        region_masks,
    )

    certificate_errors = [
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
        and nested_error <= numerical_tolerance
        and max(certificate_errors) <= 10.0 * numerical_tolerance
    )

    row: dict[str, Any] = {
        "resolved": bool(resolved),
        "certification_pass": certification_pass,
        "quadrature_order": int(used_order),
        "numerical_tolerance": float(numerical_tolerance),
        "nested_quadrature_error": float(nested_error),
        "endpoint_change": endpoint_change,
        "R": gross,
        "D": visible,
        "H": hidden,
        "chi": concealed_fraction,
        "R_over_s": float(gross / output_scale),
        "D_over_s": float(visible / output_scale),
        "H_over_s": float(hidden / output_scale),
        "interaction_pot_mass_M": interaction_pot_mass,
        "R_over_M": (
            float(gross / interaction_pot_mass)
            if interaction_pot_mass > numerical_tolerance
            else np.nan
        ),
        "H_over_M": (
            float(hidden / interaction_pot_mass)
            if interaction_pot_mass > numerical_tolerance
            else np.nan
        ),
        "signed_cosine": signed_cosine(bshap_direct, direct_ig),
        "spearman_abs": spearman_abs(bshap_direct, direct_ig),
        "top3_jaccard": top3_jaccard(bshap_direct, direct_ig),
        "false_consensus_primary": bool(
            visible / output_scale <= FC_KAPPA
            and hidden / output_scale >= FC_TAU
        ),
        "max_abs_transfer_entry": float(np.max(np.abs(transfer))),
        "pot_conservation_error": float(pot_conservation_error),
        "bshap_reconstruction_error": bshap_reconstruction_error,
        "ig_reconstruction_error": ig_reconstruction_error,
        "bshap_completeness_error": float(bshap_completeness_error),
        "ig_completeness_error": float(ig_completeness_error),
        "margin_gap_error": margin_gap_error,
        "interior_mobius_reconstruction_error": interior_error,
    }

    row.update(ctl.pot_order_diagnostics(pot_values, output_scale))
    for order in range(2, D + 1):
        value = redistribution_by_order[order]
        row[f"R_order{order}"] = float(value)
        row[f"R_order{order}_over_s"] = float(value / output_scale)
    for region in range(D):
        row[f"bshap_region{region}"] = float(bshap_direct[region])
        row[f"ig_region{region}"] = float(direct_ig[region])
        row[f"margin_region{region}"] = float(margins[region])

    return row


# =============================================================================
# Frozen-stage verification and paired summaries
# =============================================================================


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def require_empty_output_directory(path: Path) -> None:
    if path.exists():
        contents = list(path.iterdir())
        if contents:
            names = ", ".join(sorted(item.name for item in contents[:8]))
            if len(contents) > 8:
                names += ", ..."
            raise RuntimeError(
                f"Refusing to overwrite nonempty output directory: {path.resolve()}\n"
                f"Existing entries: {names}\n"
                "Preserve the run or choose a new --out-dir."
            )
    else:
        path.mkdir(parents=True, exist_ok=False)


def verify_passed_controls(controls_dir: Path, protocol_dir: Path) -> dict[str, Any]:
    summary_path = controls_dir / "control_summary.json"
    manifest_path = controls_dir / "control_manifest_sha256.json"
    if not summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Missing the frozen V2 control summary or manifest in "
            f"{controls_dir.resolve()}."
        )

    manifest = read_json(manifest_path)
    for name, expected_hash in manifest["files"].items():
        path = controls_dir / name
        if not path.exists():
            raise RuntimeError(f"Manifest-listed V2 artifact is missing: {path}")
        actual_hash = ctl.sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen V2 control artifact changed: {name}\n"
                f"expected {expected_hash}\nactual   {actual_hash}"
            )

    protocol_manifest_hash = ctl.sha256_file(
        protocol_dir / "manifest_sha256.json"
    )
    if manifest["protocol_manifest_sha256"] != protocol_manifest_hash:
        raise RuntimeError("V2 controls do not point to this frozen V1 protocol.")

    summary = read_json(summary_path)
    if not summary.get("all_pass", False):
        raise RuntimeError("The V2 fatal-control stage did not pass.")
    if summary.get("n_control_audits") != 1000:
        raise RuntimeError("The V2 fatal-control stage does not contain 1,000 audits.")
    if summary.get("model_seeds") != MODEL_SEEDS:
        raise RuntimeError("The V2 control seeds differ from the frozen seeds.")
    return summary


def paired_surface(
    audit_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    majority = np.zeros((len(TAUS), len(KAPPAS)), dtype=int)
    at_least_one = np.zeros_like(majority)
    all_five = np.zeros_like(majority)

    for tau_i, tau in enumerate(TAUS):
        for kappa_i, kappa in enumerate(KAPPAS):
            event = (
                (audit_df["D_over_s"] <= kappa)
                & (audit_df["H_over_s"] >= tau)
            )
            temporary = audit_df[["audit_id"]].copy()
            temporary["event"] = event.to_numpy()
            recurrence = temporary.groupby("audit_id")["event"].sum()
            majority[tau_i, kappa_i] = int(
                (recurrence >= MAJORITY_SEEDS).sum()
            )
            at_least_one[tau_i, kappa_i] = int((recurrence >= 1).sum())
            all_five[tau_i, kappa_i] = int((recurrence == 5).sum())

    return majority, at_least_one, all_five


def endpoint_stability_table(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for audit_id, group in audit_df.groupby("audit_id", sort=True):
        if len(group) != len(MODEL_SEEDS):
            raise RuntimeError(f"Audit endpoint {audit_id} does not have five fits.")
        if group["official_test_index"].nunique() != 1:
            raise RuntimeError("Official test index changes within an audit endpoint.")
        if group["true_class_id"].nunique() != 1:
            raise RuntimeError("True class changes within an audit endpoint.")

        rows.append(
            {
                "audit_id": int(audit_id),
                "official_test_index": int(group["official_test_index"].iloc[0]),
                "true_class_id": int(group["true_class_id"].iloc[0]),
                "true_class_name": str(group["true_class_name"].iloc[0]),
                "strict_fc_seed_count": int(
                    group["false_consensus_primary"].sum()
                ),
                "visible_agreement_seed_count": int(
                    (group["D_over_s"] <= FC_KAPPA).sum()
                ),
                "material_hidden_seed_count": int(
                    (group["H_over_s"] >= FC_TAU).sum()
                ),
                "correct_classification_seed_count": int(
                    group["endpoint_correct"].sum()
                ),
                "median_D_over_s": float(group["D_over_s"].median()),
                "median_H_over_s": float(group["H_over_s"].median()),
                "min_H_over_s": float(group["H_over_s"].min()),
                "max_H_over_s": float(group["H_over_s"].max()),
                "median_chi": float(group["chi"].dropna().median()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main experiment
# =============================================================================


def run(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir)
    controls_dir = Path(args.controls_dir)
    out_dir = Path(args.out_dir)

    official_train = FashionMNIST(root=args.data_dir, train=True, download=True)
    official_test = FashionMNIST(root=args.data_dir, train=False, download=True)
    train_images_all = ctl.to_numpy_uint8_images(official_train.data)
    train_targets_all = ctl.to_numpy_int64(official_train.targets)
    test_images_all = ctl.to_numpy_uint8_images(official_test.data)
    test_targets_all = ctl.to_numpy_int64(official_test.targets)

    protocol, preprocessing = ctl.verify_protocol(
        protocol_dir,
        train_images_all,
        train_targets_all,
        test_images_all,
        test_targets_all,
    )
    control_summary = verify_passed_controls(controls_dir, protocol_dir)
    require_empty_output_directory(out_dir)

    split = np.load(protocol_dir / "split_indices.npz")
    train_idx = split["official_train_train_idx"].astype(np.int64)
    val_idx = split["official_train_validation_idx"].astype(np.int64)
    test_idx = split["canonical_test_idx"].astype(np.int64)
    audit_idx = split["audit_official_test_idx"].astype(np.int64)
    baseline_idx = int(split["baseline_official_train_index"][0])

    mean_01 = float(preprocessing["global_train_pixel_mean_0_1"])
    std_01 = float(preprocessing["global_train_pixel_std_0_1"])
    train_images = ctl.normalize_images(
        train_images_all[train_idx], mean_01, std_01, dtype=np.float32
    )
    val_images = ctl.normalize_images(
        train_images_all[val_idx], mean_01, std_01, dtype=np.float32
    )
    test_images = ctl.normalize_images(
        test_images_all[test_idx], mean_01, std_01, dtype=np.float32
    )
    baseline = ctl.normalize_images(
        train_images_all[baseline_idx:baseline_idx + 1],
        mean_01,
        std_01,
        dtype=np.float64,
    )[0]
    audit_images = ctl.normalize_images(
        test_images_all[audit_idx], mean_01, std_01, dtype=np.float64
    )

    train_labels = train_targets_all[train_idx]
    val_labels = train_targets_all[val_idx]
    test_labels = test_targets_all[test_idx]
    audit_labels = test_targets_all[audit_idx]

    region_map = np.load(protocol_dir / "region_map.npy").astype(np.int16)
    region_masks = ctl.region_masks_tensor(region_map).double()
    contract = protocol["future_model_contract"]["softplus_cnn"]
    gate_contract = contract["predictive_gate"]

    descriptive = protocol["descriptive_false_consensus"]
    frozen_point = descriptive["fixed_operating_point"]
    if (
        float(frozen_point["D_over_s_max"]) != FC_KAPPA
        or float(frozen_point["H_over_s_min"]) != FC_TAU
        or [float(value) for value in descriptive["complete_kappa_grid"]]
        != KAPPAS
        or [float(value) for value in descriptive["complete_tau_grid"]]
        != TAUS
    ):
        raise RuntimeError("The CNN threshold contract differs from frozen V1.")
    if (
        contract["conv_channels"] != [32, 64]
        or int(contract["kernel_size"]) != 3
        or int(contract["padding"]) != 1
        or int(contract["fully_connected_width"]) != 128
    ):
        raise RuntimeError("The CNN implementation differs from frozen V1.")
    model_parameter_count = int(
        sum(parameter.numel() for parameter in SoftplusCNN().parameters())
    )

    fit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    print()
    print("=" * 88)
    print("E3b — SOFTPLUS CNN FITTED-MODEL EXHAUSTIVE REGION-LEDGER AUDIT")
    print("=" * 88)
    print(f"Frozen protocol                 : {protocol_dir.resolve()}")
    print(f"Passed fatal controls           : {controls_dir.resolve()}")
    print(f"Output directory                : {out_dir.resolve()}")
    print(f"Frozen model seeds              : {MODEL_SEEDS}")
    print(f"Frozen audit endpoints          : {len(audit_idx)}")
    print(f"Complete mosaics per audit      : {N_MASKS}")
    print(f"CNN trainable parameters        : {model_parameter_count}")
    print(f"Primary FC point                : D/s <= {FC_KAPPA}, H/s >= {FC_TAU}")
    print(
        "Predictive gate                : "
        f"test accuracy >= {gate_contract['minimum_test_accuracy']}, "
        "finite logits, extrapolation <= 10s"
    )
    print(f"Training / audit dtype          : float32 / float64")
    print(f"Audit chunk size                : {args.audit_chunk_size}")
    print()

    for seed in MODEL_SEEDS:
        print(f"[fit] softplus_cnn seed={seed}")
        set_seed(seed)
        model = SoftplusCNN()
        model, best_epoch, best_val_nll = fit_cnn(
            model,
            train_images,
            train_labels,
            val_images,
            val_labels,
            seed=seed,
            max_epochs=int(contract["max_epochs"]),
            patience=int(contract["patience"]),
            batch_size=int(contract["batch_size"]),
            learning_rate=float(contract["learning_rate"]),
            weight_decay=float(contract["weight_decay"]),
        )

        model = model.double().eval()
        train_logits = predict_logits_np(model, train_images)
        val_logits = predict_logits_np(model, val_images)
        test_logits = predict_logits_np(model, test_images)
        train_metrics = ctl.classification_metrics(train_labels, train_logits)
        val_metrics = ctl.classification_metrics(val_labels, val_logits)
        test_metrics = ctl.classification_metrics(test_labels, test_logits)

        train_evidence = ctl.centered_logits_numpy(train_logits, train_labels)
        test_evidence = ctl.centered_logits_numpy(test_logits, test_labels)
        output_scale = float(
            np.quantile(train_evidence, 0.95)
            - np.quantile(train_evidence, 0.05)
        )
        if not np.isfinite(output_scale) or output_scale <= 1e-12:
            raise RuntimeError(f"CNN seed {seed}: degenerate training output scale.")
        train_median = float(np.median(train_evidence))
        max_test_distance_s = float(
            np.max(np.abs(test_evidence - train_median)) / output_scale
        )
        finite_logits = bool(
            np.isfinite(train_logits).all()
            and np.isfinite(val_logits).all()
            and np.isfinite(test_logits).all()
        )
        predictive_gate = bool(
            finite_logits
            and test_metrics["accuracy"]
            >= float(gate_contract["minimum_test_accuracy"])
            and max_test_distance_s
            <= float(
                gate_contract[
                    "maximum_test_centered_logit_distance_from_training_median_in_s"
                ]
            )
        )

        fit_row = {
            "family": "softplus_cnn",
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_nll": float(best_val_nll),
            "output_scale_s": output_scale,
            "train_accuracy": train_metrics["accuracy"],
            "train_nll": train_metrics["negative_log_likelihood"],
            "val_accuracy": val_metrics["accuracy"],
            "val_nll": val_metrics["negative_log_likelihood"],
            "test_accuracy": test_metrics["accuracy"],
            "test_nll": test_metrics["negative_log_likelihood"],
            "max_test_centered_logit_distance_from_train_median_in_s": (
                max_test_distance_s
            ),
            "finite_logits": finite_logits,
            "predictive_gate": predictive_gate,
        }
        test_predictions = np.argmax(test_logits, axis=1)
        for class_id in range(N_CLASSES):
            class_rows_mask = test_labels == class_id
            fit_row[f"test_accuracy_class{class_id}"] = float(
                np.mean(test_predictions[class_rows_mask] == class_id)
            )
        fit_rows.append(fit_row)

        print(
            f"      selected epoch={best_epoch:2d} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f} | "
            f"test_NLL={test_metrics['negative_log_likelihood']:.4f} | "
            f"s={output_scale:.4f} | gate={predictive_gate}"
        )

        if not predictive_gate:
            pd.DataFrame(fit_rows).to_csv(
                out_dir / "fit_metrics_partial.csv", index=False
            )
            raise RuntimeError(
                f"CNN seed {seed} failed the frozen predictive pathology gate. "
                "No attribution result from this stage may be interpreted."
            )

        torch.save(
            {
                "experiment": "E3b Softplus CNN fitted-model audit",
                "family": "softplus_cnn",
                "seed": int(seed),
                "protocol_manifest_sha256": ctl.sha256_file(
                    protocol_dir / "manifest_sha256.json"
                ),
                "controls_manifest_sha256": ctl.sha256_file(
                    controls_dir / "control_manifest_sha256.json"
                ),
                "model_contract": contract,
                "normalization_mean_0_1": mean_01,
                "normalization_std_0_1": std_01,
                "state_dict": model.state_dict(),
            },
            out_dir / f"cnn_seed_{seed}.pt",
        )

        # Only slider derivatives are needed from here onward.
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        official_to_test_position = {
            int(index): position for position, index in enumerate(test_idx)
        }
        seed_rows: list[dict[str, Any]] = []
        audit_start = time.perf_counter()

        for audit_id, (official_index, endpoint, true_class) in enumerate(
            zip(audit_idx, audit_images, audit_labels)
        ):
            test_position = official_to_test_position[int(official_index)]
            endpoint_logits = test_logits[test_position]
            predicted_class = int(np.argmax(endpoint_logits))

            result = detailed_audit(
                model,
                baseline,
                endpoint,
                int(true_class),
                region_masks,
                output_scale,
                chunk_size=args.audit_chunk_size,
            )
            row = {
                "family": "softplus_cnn",
                "seed": int(seed),
                "audit_id": int(audit_id),
                "official_test_index": int(official_index),
                "true_class_id": int(true_class),
                "true_class_name": CLASS_NAMES[int(true_class)],
                "endpoint_predicted_class": predicted_class,
                "endpoint_correct": bool(predicted_class == int(true_class)),
                "endpoint_true_class_centered_logit": float(
                    ctl.centered_logits_numpy(
                        endpoint_logits[None, :],
                        np.asarray([true_class], dtype=np.int64),
                    )[0]
                ),
                "output_scale_s": output_scale,
                **result,
            }
            audit_rows.append(row)
            seed_rows.append(row)

            if (audit_id + 1) % 10 == 0:
                current = pd.DataFrame(seed_rows)
                elapsed = time.perf_counter() - audit_start
                rate = elapsed / float(audit_id + 1)
                remaining_minutes = rate * (100 - audit_id - 1) / 60.0
                print(
                    f"      audits {audit_id + 1:3d}/100 | "
                    f"median H/s={current['H_over_s'].median():.4f} | "
                    f"max H/s={current['H_over_s'].max():.4f} | "
                    f"FC={int(current['false_consensus_primary'].sum())} | "
                    f"ETA~{remaining_minutes:.1f} min"
                )

        # Durable local progress after every completed fit and audit block.
        pd.DataFrame(fit_rows).to_csv(out_dir / "fit_metrics_partial.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out_dir / "cnn_audits_partial.csv", index=False)

    fit_df = pd.DataFrame(fit_rows)
    audit_df = pd.DataFrame(audit_rows)
    fit_df.to_csv(out_dir / "fit_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "cnn_audits.csv", index=False)

    # The two partial-progress files are deliberately retained and hashed: they
    # are independently useful evidence that the final rows were accumulated
    # seed by seed rather than silently regenerated with changed endpoints.

    endpoint_df = endpoint_stability_table(audit_df)
    endpoint_df.to_csv(out_dir / "endpoint_stability.csv", index=False)

    class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        group = audit_df[audit_df["true_class_id"] == class_id]
        endpoint_group = endpoint_df[endpoint_df["true_class_id"] == class_id]
        class_rows.append(
            {
                "true_class_id": class_id,
                "true_class_name": class_name,
                "audit_rows": int(len(group)),
                "unique_endpoints": int(group["audit_id"].nunique()),
                "median_D_over_s": float(group["D_over_s"].median()),
                "median_H_over_s": float(group["H_over_s"].median()),
                "q90_H_over_s": float(group["H_over_s"].quantile(0.90)),
                "median_chi": float(group["chi"].dropna().median()),
                "strict_fc_rows": int(group["false_consensus_primary"].sum()),
                "strict_fc_endpoints_in_at_least_3_of_5": int(
                    (endpoint_group["strict_fc_seed_count"] >= 3).sum()
                ),
            }
        )
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(out_dir / "class_summary.csv", index=False)

    if len(audit_df) != 500 or audit_df["audit_id"].nunique() != 100:
        raise RuntimeError("Final CNN audit table does not have the frozen 500-row design.")
    if not (audit_df.groupby("audit_id").size() == 5).all():
        raise RuntimeError("At least one frozen endpoint lacks exactly five CNN fits.")

    pass_flags = {
        "passed_controls_verified": bool(control_summary.get("all_pass", False)),
        "all_predictive_gates_pass": bool(fit_df["predictive_gate"].all()),
        "exactly_500_audits": bool(len(audit_df) == 500),
        "exactly_100_endpoints_with_5_fits_each": bool(
            audit_df["audit_id"].nunique() == 100
            and (audit_df.groupby("audit_id").size() == 5).all()
        ),
        "all_audits_resolved": bool(audit_df["resolved"].all()),
        "all_audits_certified": bool(audit_df["certification_pass"].all()),
    }
    all_pass = bool(all(pass_flags.values()))

    seed_summary: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        group = audit_df[audit_df["seed"] == seed]
        seed_summary.append(
            {
                "seed": int(seed),
                "median_D_over_s": float(group["D_over_s"].median()),
                "median_H_over_s": float(group["H_over_s"].median()),
                "q90_H_over_s": float(group["H_over_s"].quantile(0.90)),
                "max_H_over_s": float(group["H_over_s"].max()),
                "median_chi": float(group["chi"].dropna().median()),
                "false_consensus_count": int(
                    group["false_consensus_primary"].sum()
                ),
                "false_consensus_rate": float(
                    group["false_consensus_primary"].mean()
                ),
            }
        )

    majority_matrix, one_matrix, five_matrix = paired_surface(audit_df)
    original_tau_i = TAUS.index(FC_TAU)
    original_kappa_i = KAPPAS.index(FC_KAPPA)

    order_summary = {
        str(order): {
            "median": float(audit_df[f"R_order{order}_over_s"].median()),
            "q90": float(
                audit_df[f"R_order{order}_over_s"].quantile(0.90)
            ),
            "max": float(audit_df[f"R_order{order}_over_s"].max()),
        }
        for order in range(2, D + 1)
    }

    recurrence_histogram = {
        str(count): int((endpoint_df["strict_fc_seed_count"] == count).sum())
        for count in range(6)
    }

    summary = {
        "experiment": "E3b Softplus CNN fitted-model exhaustive audit",
        "protocol_manifest_sha256": ctl.sha256_file(
            protocol_dir / "manifest_sha256.json"
        ),
        "controls_manifest_sha256": ctl.sha256_file(
            controls_dir / "control_manifest_sha256.json"
        ),
        "model_contract": contract,
        "trainable_parameter_count": model_parameter_count,
        "model_seeds": MODEL_SEEDS,
        "n_audits": int(len(audit_df)),
        "n_unique_endpoints": int(audit_df["audit_id"].nunique()),
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
            "median_H_over_M": float(audit_df["H_over_M"].dropna().median()),
            "audit_endpoint_accuracy": float(audit_df["endpoint_correct"].mean()),
        },
        "fatal_control_reference": {
            "additive_max_H_over_s": float(
                control_summary["families"]["additive"]["max_H_over_s"]
            ),
            "quadratic_max_H_over_s": float(
                control_summary["families"]["quadratic"]["max_H_over_s"]
            ),
            "quadratic_median_pair_pot_L1_over_s": float(
                control_summary["families"]["quadratic"][
                    "median_pair_pot_L1_over_s"
                ]
            ),
        },
        "by_seed": seed_summary,
        "by_true_class": class_rows,
        "redistribution_by_interaction_order": order_summary,
        "strict_false_consensus_endpoint_recurrence_histogram": (
            recurrence_histogram
        ),
        "paired_threshold_surface": {
            "kappas": KAPPAS,
            "taus": TAUS,
            "majority_3_of_5_matrix_rows_tau_cols_kappa": (
                majority_matrix.tolist()
            ),
            "at_least_1_of_5_matrix_rows_tau_cols_kappa": one_matrix.tolist(),
            "all_5_of_5_matrix_rows_tau_cols_kappa": five_matrix.tolist(),
            "original_point_majority_count": int(
                majority_matrix[original_tau_i, original_kappa_i]
            ),
            "original_point_at_least_one_count": int(
                one_matrix[original_tau_i, original_kappa_i]
            ),
            "original_point_all_five_count": int(
                five_matrix[original_tau_i, original_kappa_i]
            ),
        },
        "numerical_diagnostics": {
            "resolved_rate": float(audit_df["resolved"].mean()),
            "certification_pass_rate": float(
                audit_df["certification_pass"].mean()
            ),
            "quadrature_order_counts": {
                str(int(order)): int(count)
                for order, count in audit_df["quadrature_order"]
                .value_counts()
                .sort_index()
                .items()
            },
            "max_nested_quadrature_error": float(
                audit_df["nested_quadrature_error"].max()
            ),
            "max_pot_conservation_error": float(
                audit_df["pot_conservation_error"].max()
            ),
            "max_ig_reconstruction_error": float(
                audit_df["ig_reconstruction_error"].max()
            ),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(summary, out_dir / "cnn_summary.json")

    # Engineering diagnostic only. Final appendix figures are generated from
    # the frozen CSV/JSON outputs by the read-only artifact script.
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 2.75))

    axes[0].plot(
        [str(seed)[-2:] for seed in MODEL_SEEDS],
        fit_df["test_accuracy"].to_numpy(),
        marker="o",
        color="#376996",
    )
    axes[0].axhline(
        float(gate_contract["minimum_test_accuracy"]),
        linestyle="--",
        linewidth=0.9,
        color="#333333",
    )
    axes[0].set_xlabel("seed suffix")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("(a) Predictive gate", fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.22)

    axes[1].scatter(
        audit_df["D_over_s"],
        audit_df["H_over_s"],
        s=9,
        alpha=0.42,
        color="#7A5195",
    )
    axes[1].axvline(FC_KAPPA, linestyle="--", linewidth=0.9, color="#333333")
    axes[1].axhline(FC_TAU, linestyle="--", linewidth=0.9, color="#333333")
    axes[1].set_xlabel(r"$D/s$")
    axes[1].set_ylabel(r"$H/s$")
    axes[1].set_title("(b) Regime map", fontsize=9)

    values = [
        audit_df.loc[audit_df["seed"] == seed, "H_over_s"].to_numpy()
        for seed in MODEL_SEEDS
    ]
    axes[2].boxplot(
        values,
        tick_labels=[str(seed)[-2:] for seed in MODEL_SEEDS],
        showfliers=False,
    )
    axes[2].set_xlabel("seed suffix")
    axes[2].set_ylabel(r"$H/s$")
    axes[2].set_title("(c) Seed stability", fontsize=9)

    image = axes[3].imshow(majority_matrix, origin="lower", aspect="auto")
    axes[3].set_xticks(np.arange(len(KAPPAS)))
    axes[3].set_xticklabels([f"{value:g}" for value in KAPPAS], fontsize=7)
    axes[3].set_yticks(np.arange(len(TAUS)))
    axes[3].set_yticklabels([f"{value:g}" for value in TAUS], fontsize=7)
    for row_i in range(len(TAUS)):
        for col_i in range(len(KAPPAS)):
            axes[3].text(
                col_i,
                row_i,
                str(majority_matrix[row_i, col_i]),
                ha="center",
                va="center",
                fontsize=7,
            )
    axes[3].scatter(
        [original_kappa_i],
        [original_tau_i],
        marker="s",
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
    )
    axes[3].set_xlabel(r"$\kappa$")
    axes[3].set_ylabel(r"$\tau$")
    axes[3].set_title("(d) Robust endpoints", fontsize=9)
    fig.colorbar(image, ax=axes[3], fraction=0.05, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "cnn_diagnostic_four_panel.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "cnn_diagnostic_four_panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    output_files = sorted(
        path for path in out_dir.iterdir() if path.name != "cnn_manifest_sha256.json"
    )
    write_json(
        {
            "hash_algorithm": "SHA-256",
            "protocol_manifest_sha256": ctl.sha256_file(
                protocol_dir / "manifest_sha256.json"
            ),
            "controls_manifest_sha256": ctl.sha256_file(
                controls_dir / "control_manifest_sha256.json"
            ),
            "files": {path.name: ctl.sha256_file(path) for path in output_files},
        },
        out_dir / "cnn_manifest_sha256.json",
    )

    print()
    print("=" * 88)
    print("E3b — SOFTPLUS CNN SUMMARY")
    print("=" * 88)
    print("Predictive fits:")
    for row in fit_rows:
        print(
            f"  seed={row['seed']} | epoch={row['best_epoch']:2d} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"test_acc={row['test_accuracy']:.4f} | "
            f"test_NLL={row['test_nll']:.4f} | gate={row['predictive_gate']}"
        )
    print()
    print(f"Audits                        : {len(audit_df)}")
    print(f"Resolved rate                 : {audit_df['resolved'].mean():.3f}")
    print(
        f"Certification pass rate       : "
        f"{audit_df['certification_pass'].mean():.3f}"
    )
    print(f"Median D/s                    : {audit_df['D_over_s'].median():.4f}")
    print(f"Median H/s                    : {audit_df['H_over_s'].median():.4f}")
    print(
        f"90th percentile H/s           : "
        f"{audit_df['H_over_s'].quantile(0.90):.4f}"
    )
    print(f"Maximum H/s                   : {audit_df['H_over_s'].max():.4f}")
    print(
        f"Median hidden fraction chi    : "
        f"{audit_df['chi'].dropna().median():.4f}"
    )
    print(
        f"Primary FC row count          : "
        f"{int(audit_df['false_consensus_primary'].sum())}/{len(audit_df)}"
    )
    print(
        "Strict FC endpoint recurrence : "
        + ", ".join(
            f"{count}/5 -> {recurrence_histogram[str(count)]}"
            for count in range(6)
        )
    )
    print()
    print("By seed:")
    for row in seed_summary:
        print(
            f"  {row['seed']}: median H/s={row['median_H_over_s']:.4f}, "
            f"q90={row['q90_H_over_s']:.4f}, "
            f"max={row['max_H_over_s']:.4f}, "
            f"FC={row['false_consensus_count']}/100"
        )

    print()
    print("Paired threshold surface")
    print("Entry = # endpoints satisfying the criterion in >=3/5 CNN fits.")
    print()
    header = "tau \\ kappa | " + " | ".join(f"{value:>7g}" for value in KAPPAS)
    print(header)
    print("-" * len(header))
    for row_i, tau in enumerate(TAUS):
        values = " | ".join(
            f"{majority_matrix[row_i, col_i]:7d}"
            for col_i in range(len(KAPPAS))
        )
        print(f"{tau:>11g} | {values}")

    print()
    print(
        f"Original point ({FC_KAPPA}, {FC_TAU}): "
        f">=1/5={one_matrix[original_tau_i, original_kappa_i]}/100, "
        f">=3/5={majority_matrix[original_tau_i, original_kappa_i]}/100, "
        f"5/5={five_matrix[original_tau_i, original_kappa_i]}/100"
    )
    print()
    print("Hard checks:")
    for name, value in pass_flags.items():
        print(f"  {name}: {value}")
    print(f"  all_hard_checks_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 88)

    if not all_pass:
        raise RuntimeError(
            "CNN stage failed a frozen predictive or numerical hard check."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final E3b Softplus CNN exhaustive audit."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--protocol-dir", default="./e3_vision_protocol")
    parser.add_argument("--controls-dir", default="./e3_vision_controls")
    parser.add_argument("--out-dir", default="./e3_vision_cnn")
    parser.add_argument(
        "--audit-chunk-size",
        type=int,
        default=512,
        help="Performance-only batch size for float64 CNN quadrature.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
