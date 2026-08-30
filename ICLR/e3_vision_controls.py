#!/usr/bin/env python3
"""
E3b — Vision fatal controls + exhaustive eight-region audit engine
=================================================================

Consumes the immutable Fashion-MNIST protocol produced by
``e3_vision_protocol.py``.  It verifies every frozen SHA-256 hash before
loading the same train/validation/test split, observed baseline image, fixed
2 x 4 region grid, 100 class-balanced audit endpoints, and five model seeds.

Fatal controls
--------------
1. Additive region Softplus model

       logits(x) = b + sum_r f_r(x_r)

   Each of the eight fixed image regions is processed independently.  The
   true-class centered logit is therefore additive across region sliders, so
   every cross-region pot and the complete transfer ledger must be numerical
   zero.

2. Quadratic region-interaction model

       e_r(x_r) = W_r vec(x_r) + b_r                 (linear)
       logits(x) = b + singleton terms
                     + sum_{r<q} B_{rq}(e_r odot e_q)

   Genuine region-pair interactions are learned.  Because every region
   embedding is linear, each anchored pair pot is proportional to t_r t_q in
   the two region sliders.  Baseline Shapley and straight-line IG split it
   exactly 1/2--1/2, so the transfer ledger must again be numerical zero.

The reusable audit engine used by the Softplus CNN evaluates all
2^8 = 256 Boolean region mosaics, the complete anchored Moebius decomposition,
potwise Baseline Shapley, potwise straight-line IG, T/R/D/H/chi, adaptive
Gauss--Legendre quadrature, and independent reconstruction certificates.

The controls train in float32 for practical CPU runtime.  Frozen selected
weights are cast to float64 before every prediction-scale calculation and
attribution audit.

Hard pass conditions
--------------------
* every one of 1,000 audits resolves;
* every audit passes every numerical certificate;
* max H/s <= 1e-7 for both families;
* Additive has no material order >= 2 anchored pots;
* Quadratic has no material order >= 3 anchored pots.

Run from the repository root
----------------------------
    python ICLR/e3_vision_controls.py

Outputs
-------
    ./e3_vision_controls/
        additive_seed_<seed>.pt
        quadratic_seed_<seed>.pt
        fit_metrics.csv
        control_audits.csv
        control_summary.json
        control_null_diagnostic.pdf
        control_null_diagnostic.png
        control_manifest_sha256.json

The output directory must be absent or empty.  A failed stage leaves its
diagnostic files in place and must be investigated rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


# =============================================================================
# Frozen representation constants
# =============================================================================

D = 8
N_MASKS = 1 << D
FULL_MASK = N_MASKS - 1
N_CLASSES = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
MODEL_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
QUAD_ORDERS = [16, 32, 64, 128, 256]
CONTROL_NULL_MAX_H_OVER_S = 1e-7


def mask_matrix(d: int = D) -> np.ndarray:
    masks = np.zeros((1 << d, d), dtype=np.float64)
    for mask in range(1 << d):
        for feature in range(d):
            masks[mask, feature] = float(bool(mask & (1 << feature)))
    return masks


MASKS = mask_matrix(D)
MASK_POPCOUNT = np.asarray([mask.bit_count() for mask in range(N_MASKS)], dtype=int)


# =============================================================================
# General utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Must match the array hash used by e3_vision_protocol.py exactly."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


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
                "Preserve the failed/completed run or choose a new --out-dir."
            )
    else:
        path.mkdir(parents=True, exist_ok=False)


def to_numpy_uint8_images(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 dataset images, got {result.dtype}.")
    return result


def to_numpy_int64(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def normalize_images(
    images_uint8: np.ndarray,
    mean_01: float,
    std_01: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    scalar_type = resolved_dtype.type
    images = images_uint8.astype(resolved_dtype, copy=False) / scalar_type(255.0)
    images = (images - scalar_type(mean_01)) / scalar_type(std_01)
    return images[:, None, :, :]


def global_pixel_moments_01(
    images_uint8: np.ndarray,
    batch_size: int = 4096,
) -> tuple[float, float]:
    total = 0.0
    total_squares = 0.0
    for start in range(0, len(images_uint8), batch_size):
        stop = min(start + batch_size, len(images_uint8))
        batch = images_uint8[start:stop].astype(np.float64)
        total += float(batch.sum())
        total_squares += float(np.einsum("nhw,nhw->", batch, batch))
    count = int(images_uint8.size)
    mean_01 = total / (255.0 * count)
    second_moment_01 = total_squares / ((255.0**2) * count)
    variance_01 = max(0.0, second_moment_01 - mean_01**2)
    return float(mean_01), float(np.sqrt(variance_01))


def centered_logit_tensor(logits: torch.Tensor, true_class: int) -> torch.Tensor:
    target = logits[:, true_class]
    alternatives = (logits.sum(dim=1) - target) / float(N_CLASSES - 1)
    return target - alternatives


def centered_logits_numpy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    target = logits[np.arange(len(logits)), labels]
    alternatives = (logits.sum(axis=1) - target) / float(N_CLASSES - 1)
    return target - alternatives


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    pred = np.argmax(logits, axis=1)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    log_prob_true = shifted[np.arange(len(labels)), labels] - logsumexp

    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "negative_log_likelihood": float(-np.mean(log_prob_true)),
    }


def mobius_transform(values: np.ndarray, d: int = D) -> np.ndarray:
    """Fast subset Moebius transform along the first (mask) dimension."""
    output = np.array(values, dtype=np.float64, copy=True)
    for bit in range(d):
        step = 1 << bit
        for mask in range(1 << d):
            if mask & step:
                output[mask] -= output[mask ^ step]
    return output


def exact_shapley_from_game(game: np.ndarray, d: int = D) -> np.ndarray:
    attribution = np.zeros(d, dtype=np.float64)
    for feature in range(d):
        bit = 1 << feature
        for mask in range(1 << d):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = 1.0 / (d * math.comb(d - 1, size))
            attribution[feature] += weight * (game[mask | bit] - game[mask])
    return attribution


# =============================================================================
# Frozen protocol verification and data loading
# =============================================================================


def verify_protocol(
    protocol_dir: Path,
    train_images: np.ndarray,
    train_targets: np.ndarray,
    test_images: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        "split_indices.npz",
        "audit_indices.csv",
        "audit_images_uint8.npz",
        "baseline.csv",
        "baseline_image_uint8.npy",
        "region_definitions.csv",
        "region_map.npy",
        "preprocessing.json",
        "protocol.json",
        "manifest_sha256.json",
    ]
    for name in required:
        if not (protocol_dir / name).exists():
            raise FileNotFoundError(f"Missing frozen protocol artifact: {protocol_dir / name}")

    protocol = read_json(protocol_dir / "protocol.json")
    manifest = read_json(protocol_dir / "manifest_sha256.json")

    for name, expected_hash in manifest["files"].items():
        path = protocol_dir / name
        if not path.exists():
            raise RuntimeError(f"Manifest-listed protocol artifact is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen protocol artifact changed: {name}\n"
                f"expected {expected_hash}\nactual   {actual_hash}"
            )

    frozen_hashes = protocol["dataset"]["content_sha256"]
    current_arrays = {
        "official_train_images_uint8": train_images,
        "official_train_targets_int64": train_targets,
        "official_test_images_uint8": test_images,
        "official_test_targets_int64": test_targets,
    }
    for name, array in current_arrays.items():
        actual = sha256_array(array)
        if actual != frozen_hashes[name]:
            raise RuntimeError(
                f"Fashion-MNIST content hash mismatch for {name}.\n"
                f"expected {frozen_hashes[name]}\nactual   {actual}"
            )

    if not protocol["protocol_checks"].get("all_pass", False):
        raise RuntimeError("The frozen vision protocol did not pass its own checks.")
    if protocol["model_seeds"] != MODEL_SEEDS:
        raise RuntimeError(
            f"Model seeds differ from protocol: {protocol['model_seeds']}"
        )
    if protocol["numerical_contract"]["adaptive_gauss_legendre_orders"] != QUAD_ORDERS:
        raise RuntimeError("Quadrature orders differ from the frozen protocol.")
    if protocol["explanation_representation"]["n_region_features"] != D:
        raise RuntimeError("Frozen explanation dimension is not eight.")

    preprocessing = read_json(protocol_dir / "preprocessing.json")
    split = np.load(protocol_dir / "split_indices.npz")
    train_idx = split["official_train_train_idx"].astype(np.int64)
    frozen_mean = float(preprocessing["global_train_pixel_mean_0_1"])
    frozen_std = float(preprocessing["global_train_pixel_std_0_1"])
    actual_mean, actual_std = global_pixel_moments_01(train_images[train_idx])
    if abs(actual_mean - frozen_mean) > 1e-14 or abs(actual_std - frozen_std) > 1e-14:
        raise RuntimeError("Train-only normalization statistics no longer reproduce.")

    baseline_idx = int(split["baseline_official_train_index"][0])
    frozen_baseline = np.load(protocol_dir / "baseline_image_uint8.npy")
    if not np.array_equal(frozen_baseline, train_images[baseline_idx]):
        raise RuntimeError("Frozen baseline image no longer matches the dataset row.")

    audit_idx = split["audit_official_test_idx"].astype(np.int64)
    frozen_audit = np.load(protocol_dir / "audit_images_uint8.npz")
    if not np.array_equal(frozen_audit["images_uint8"], test_images[audit_idx]):
        raise RuntimeError("Frozen audit images no longer match the test rows.")
    if not np.array_equal(frozen_audit["true_class_id"], test_targets[audit_idx]):
        raise RuntimeError("Frozen audit labels no longer match the test rows.")

    return protocol, preprocessing


# =============================================================================
# Models frozen by protocol v1.0
# =============================================================================


def region_slices_from_map(region_map: np.ndarray) -> list[tuple[int, int, int, int]]:
    slices: list[tuple[int, int, int, int]] = []
    for region in range(D):
        rows, cols = np.where(region_map == region)
        if len(rows) != 98:
            raise RuntimeError(f"Region {region} does not contain exactly 98 pixels.")
        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        col_start, col_stop = int(cols.min()), int(cols.max()) + 1
        expected = np.zeros_like(region_map, dtype=bool)
        expected[row_start:row_stop, col_start:col_stop] = True
        if not np.array_equal(expected, region_map == region):
            raise RuntimeError(f"Region {region} is not the frozen rectangle.")
        slices.append((row_start, row_stop, col_start, col_stop))
    return slices


class AdditiveRegionControl(nn.Module):
    """Eight independent smooth patch networks summed at logit level."""

    def __init__(
        self,
        region_slices: list[tuple[int, int, int, int]],
        width: int = 32,
    ) -> None:
        super().__init__()
        self.region_slices = region_slices
        self.parts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(98, width),
                    nn.Softplus(beta=1.0),
                    nn.Linear(width, width),
                    nn.Softplus(beta=1.0),
                    nn.Linear(width, N_CLASSES),
                )
                for _ in range(D)
            ]
        )
        self.logit_bias = nn.Parameter(torch.zeros(N_CLASSES))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.logit_bias[None, :].expand(len(images), -1)
        for part, (r0, r1, c0, c1) in zip(self.parts, self.region_slices):
            patch = images[:, :, r0:r1, c0:c1].reshape(len(images), -1)
            logits = logits + part(patch)
        return logits


class QuadraticRegionControl(nn.Module):
    """
    Exact degree <= 2 function of the eight region sliders.

    Region embeddings are deliberately linear.  Applying a nonlinearity before
    the pair products would invalidate the required quadratic null control.
    """

    def __init__(
        self,
        region_slices: list[tuple[int, int, int, int]],
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.region_slices = region_slices
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [nn.Linear(98, embedding_dim) for _ in range(D)]
        )
        self.pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
        total_features = (D + len(self.pairs)) * embedding_dim
        self.output = nn.Linear(total_features, N_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        region_embeddings = []
        for embedding, (r0, r1, c0, c1) in zip(
            self.embeddings,
            self.region_slices,
        ):
            patch = images[:, :, r0:r1, c0:c1].reshape(len(images), -1)
            region_embeddings.append(embedding(patch))

        singleton_features = torch.cat(region_embeddings, dim=1)
        pair_features = torch.cat(
            [region_embeddings[i] * region_embeddings[j] for i, j in self.pairs],
            dim=1,
        )
        return self.output(torch.cat([singleton_features, pair_features], dim=1))


# =============================================================================
# Training and prediction
# =============================================================================


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


def fit_classifier(
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

    val_x = torch.from_numpy(np.asarray(val_images, dtype=np.float32))
    val_y = torch.from_numpy(np.asarray(val_labels, dtype=np.int64))

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
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss in a fatal control.")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val


@torch.no_grad()
def predict_logits_np(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    outputs = []
    parameter = next(model.parameters())
    target_dtype = np.float64 if parameter.dtype == torch.float64 else np.float32
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = torch.from_numpy(np.asarray(images[start:stop], dtype=target_dtype))
        outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


# =============================================================================
# Exhaustive endpoint game and ledger audit
# =============================================================================


def region_masks_tensor(region_map: np.ndarray) -> torch.Tensor:
    masks = np.zeros((D, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)
    for region in range(D):
        masks[region, 0] = (region_map == region).astype(np.float64)
    if not np.allclose(masks.sum(axis=0), 1.0):
        raise RuntimeError("Frozen region masks do not form a partition.")
    return torch.from_numpy(masks)


def slider_images(
    t: torch.Tensor,
    baseline: torch.Tensor,
    region_delta: torch.Tensor,
) -> torch.Tensor:
    return baseline[None, :, :, :] + torch.einsum(
        "br,rchw->bchw",
        t,
        region_delta,
    )


@torch.no_grad()
def evaluate_endpoint_game(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    true_class: int,
    region_masks: torch.Tensor,
) -> np.ndarray:
    model.eval()
    baseline_t = torch.from_numpy(np.asarray(baseline, dtype=np.float64))
    endpoint_t = torch.from_numpy(np.asarray(endpoint, dtype=np.float64))
    delta = endpoint_t - baseline_t
    region_delta = region_masks * delta[None, :, :, :]

    t = torch.from_numpy(MASKS)
    images = slider_images(t, baseline_t, region_delta)
    values = centered_logit_tensor(model(images), true_class)
    baseline_value = centered_logit_tensor(model(baseline_t[None]), true_class)[0]
    return (values - baseline_value).cpu().numpy().astype(np.float64)


def integrated_mask_gradients(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    true_class: int,
    region_masks: torch.Tensor,
    order: int,
    chunk_size: int,
) -> np.ndarray:
    """
    For every coalition mask T and region r, compute

        J[T,r] = integral_0^1 d F(s * 1_T) / d t_r ds,

    with the derivative set to zero for r not in T because those coordinates
    are clamped by the coalition restriction.
    """
    model.eval()
    baseline_t = torch.from_numpy(np.asarray(baseline, dtype=np.float64))
    endpoint_t = torch.from_numpy(np.asarray(endpoint, dtype=np.float64))
    region_delta = region_masks * (endpoint_t - baseline_t)[None, :, :, :]

    nodes, weights = np.polynomial.legendre.leggauss(order)
    slider_s = 0.5 * (nodes + 1.0)
    quad_weights = 0.5 * weights

    t_all = slider_s[:, None, None] * MASKS[None, :, :]
    t_flat = t_all.reshape(-1, D)
    grad_flat = np.zeros_like(t_flat, dtype=np.float64)

    for start in range(0, len(t_flat), chunk_size):
        stop = min(start + chunk_size, len(t_flat))
        t = torch.tensor(
            t_flat[start:stop],
            dtype=torch.float64,
            requires_grad=True,
        )
        images = slider_images(t, baseline_t, region_delta)
        scalar = centered_logit_tensor(model(images), true_class)
        grad = torch.autograd.grad(
            scalar.sum(),
            t,
            create_graph=False,
            retain_graph=False,
        )[0]
        grad_flat[start:stop] = grad.detach().cpu().numpy()

    gradients = grad_flat.reshape(order, N_MASKS, D)
    gradients *= MASKS[None, :, :]
    integrated = np.tensordot(quad_weights, gradients, axes=(0, 0))
    return np.asarray(integrated, dtype=np.float64)


def potwise_ig_from_integrated_masks(integrated_masks: np.ndarray) -> np.ndarray:
    allocation = np.zeros((D, N_MASKS), dtype=np.float64)
    for feature in range(D):
        transformed = mobius_transform(integrated_masks[:, feature], d=D)
        transformed[MASKS[:, feature] < 0.5] = 0.0
        allocation[feature] = transformed
    return allocation


def equal_split_pot_allocation(pot_values: np.ndarray) -> np.ndarray:
    allocation = np.zeros((D, N_MASKS), dtype=np.float64)
    for mask in range(1, N_MASKS):
        size = MASK_POPCOUNT[mask]
        for feature in range(D):
            if mask & (1 << feature):
                allocation[feature, mask] = pot_values[mask] / float(size)
    return allocation


@torch.no_grad()
def interior_mobius_reconstruction_error(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    true_class: int,
    region_masks: torch.Tensor,
    slider_values: Iterable[float] = (0.25, 0.5, 0.75),
) -> float:
    model.eval()
    baseline_t = torch.from_numpy(np.asarray(baseline, dtype=np.float64))
    endpoint_t = torch.from_numpy(np.asarray(endpoint, dtype=np.float64))
    region_delta = region_masks * (endpoint_t - baseline_t)[None, :, :, :]
    base_value = float(
        centered_logit_tensor(model(baseline_t[None]), true_class)[0].item()
    )

    max_error = 0.0
    for slider in slider_values:
        t = torch.from_numpy(float(slider) * MASKS)
        images = slider_images(t, baseline_t, region_delta)
        values = (
            centered_logit_tensor(model(images), true_class).cpu().numpy()
            - base_value
        )
        pots = mobius_transform(values, d=D)
        reconstructed = float(np.sum(pots[1:]))
        full_value = float(values[FULL_MASK])
        max_error = max(max_error, abs(reconstructed - full_value))
    return float(max_error)


def pot_order_diagnostics(
    pot_values: np.ndarray,
    output_scale: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for order in range(1, D + 1):
        selected = np.flatnonzero(MASK_POPCOUNT == order)
        values = np.abs(pot_values[selected])
        result[f"pot_order{order}_L1_over_s"] = float(values.sum() / output_scale)
        result[f"pot_order{order}_max_abs_over_s"] = float(
            values.max(initial=0.0) / output_scale
        )
    higher_two = np.flatnonzero(MASK_POPCOUNT >= 2)
    higher_three = np.flatnonzero(MASK_POPCOUNT >= 3)
    result["max_abs_pot_order_ge2_over_s"] = float(
        np.abs(pot_values[higher_two]).max(initial=0.0) / output_scale
    )
    result["max_abs_pot_order_ge3_over_s"] = float(
        np.abs(pot_values[higher_three]).max(initial=0.0) / output_scale
    )
    return result


def audit_one(
    model: nn.Module,
    baseline: np.ndarray,
    endpoint: np.ndarray,
    true_class: int,
    region_masks: torch.Tensor,
    output_scale: float,
    chunk_size: int,
    quadrature_orders: Iterable[int] = QUAD_ORDERS,
) -> dict[str, Any]:
    game = evaluate_endpoint_game(
        model,
        baseline,
        endpoint,
        true_class,
        region_masks,
    )
    pot_values = mobius_transform(game, d=D)

    equal_split = equal_split_pot_allocation(pot_values)
    bshap_from_pots = equal_split.sum(axis=1)
    bshap_direct = exact_shapley_from_game(game, d=D)

    numerical_tolerance = max(1e-10, 1e-8 * max(float(output_scale), 1.0))
    previous_allocation = None
    resolved = False
    used_order = QUAD_ORDERS[-1]
    nested_error = float("inf")
    integrated_masks = None
    path_allocation = None

    for order in quadrature_orders:
        integrated_now = integrated_mask_gradients(
            model,
            baseline,
            endpoint,
            true_class,
            region_masks,
            order=order,
            chunk_size=chunk_size,
        )
        allocation_now = potwise_ig_from_integrated_masks(integrated_now)

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
        raise RuntimeError("Audit produced no quadrature allocation.")

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

    pot_conservation_error = 0.0
    for mask in range(1, N_MASKS):
        current = abs(float(path_allocation[:, mask].sum()) - float(pot_values[mask]))
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
    interior_error = interior_mobius_reconstruction_error(
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

    return {
        "resolved": bool(resolved),
        "certification_pass": certification_pass,
        "quadrature_order": used_order,
        "numerical_tolerance": float(numerical_tolerance),
        "nested_quadrature_error": nested_error,
        "endpoint_change": endpoint_change,
        "R": gross,
        "D": visible,
        "H": hidden,
        "chi": concealed_fraction,
        "R_over_s": float(gross / output_scale),
        "D_over_s": float(visible / output_scale),
        "H_over_s": float(hidden / output_scale),
        "max_abs_transfer_entry": float(np.max(np.abs(transfer))),
        "pot_conservation_error": pot_conservation_error,
        "bshap_reconstruction_error": bshap_reconstruction_error,
        "ig_reconstruction_error": ig_reconstruction_error,
        "bshap_completeness_error": float(bshap_completeness_error),
        "ig_completeness_error": float(ig_completeness_error),
        "margin_gap_error": margin_gap_error,
        "interior_mobius_reconstruction_error": interior_error,
        **pot_order_diagnostics(pot_values, output_scale),
    }


# =============================================================================
# Main control experiment
# =============================================================================


def run(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    require_empty_output_directory(out_dir)

    official_train = FashionMNIST(root=args.data_dir, train=True, download=True)
    official_test = FashionMNIST(root=args.data_dir, train=False, download=True)
    train_images_all = to_numpy_uint8_images(official_train.data)
    train_targets_all = to_numpy_int64(official_train.targets)
    test_images_all = to_numpy_uint8_images(official_test.data)
    test_targets_all = to_numpy_int64(official_test.targets)

    protocol, preprocessing = verify_protocol(
        protocol_dir,
        train_images_all,
        train_targets_all,
        test_images_all,
        test_targets_all,
    )

    split = np.load(protocol_dir / "split_indices.npz")
    train_idx = split["official_train_train_idx"].astype(np.int64)
    val_idx = split["official_train_validation_idx"].astype(np.int64)
    test_idx = split["canonical_test_idx"].astype(np.int64)
    audit_idx = split["audit_official_test_idx"].astype(np.int64)
    baseline_idx = int(split["baseline_official_train_index"][0])

    mean_01 = float(preprocessing["global_train_pixel_mean_0_1"])
    std_01 = float(preprocessing["global_train_pixel_std_0_1"])
    region_map = np.load(protocol_dir / "region_map.npy").astype(np.int16)
    region_slices = region_slices_from_map(region_map)
    region_masks = region_masks_tensor(region_map).double()

    train_images = normalize_images(train_images_all[train_idx], mean_01, std_01)
    val_images = normalize_images(train_images_all[val_idx], mean_01, std_01)
    test_images = normalize_images(test_images_all[test_idx], mean_01, std_01)
    baseline = normalize_images(
        train_images_all[baseline_idx:baseline_idx + 1],
        mean_01,
        std_01,
        dtype=np.float64,
    )[0]
    audit_images = normalize_images(
        test_images_all[audit_idx],
        mean_01,
        std_01,
        dtype=np.float64,
    )

    train_labels = train_targets_all[train_idx]
    val_labels = train_targets_all[val_idx]
    test_labels = test_targets_all[test_idx]
    audit_labels = test_targets_all[audit_idx]

    contracts = protocol["future_model_contract"]
    additive_contract = contracts["additive_region_control"]
    quadratic_contract = contracts["quadratic_region_control"]

    model_specs = [
        {
            "family": "additive",
            "factory": lambda: AdditiveRegionControl(
                region_slices,
                width=int(additive_contract["hidden_width"]),
            ),
            "contract": additive_contract,
        },
        {
            "family": "quadratic",
            "factory": lambda: QuadraticRegionControl(
                region_slices,
                embedding_dim=int(
                    quadratic_contract["region_embedding_dimension"]
                ),
            ),
            "contract": quadratic_contract,
        },
    ]

    fit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    print()
    print("=" * 84)
    print("E3b — VISION FATAL CONTROLS + EXHAUSTIVE REGION-LEDGER AUDIT")
    print("=" * 84)
    print(f"Protocol directory               : {protocol_dir.resolve()}")
    print(f"Output directory                 : {out_dir.resolve()}")
    print(f"Frozen audit images              : {len(audit_idx)}")
    print(f"Model seeds                      : {MODEL_SEEDS}")
    print(f"Quadrature orders                : {QUAD_ORDERS}")
    print(f"Audit dtype                      : float64")
    print()

    for spec in model_specs:
        family = str(spec["family"])
        contract = spec["contract"]

        for seed in MODEL_SEEDS:
            print(f"[fit] {family:10s} seed={seed}")
            # Initialization is part of the frozen model seed, so seed before
            # constructing the module (the training routine seeds again before
            # building the shuffled loader).
            set_seed(seed)
            model = spec["factory"]()
            model, best_epoch, best_val_nll = fit_classifier(
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

            # Cast the frozen selected fit to float64 before scale calculation,
            # prediction diagnostics, checkpointing, and every audit.
            model = model.double().eval()
            train_logits = predict_logits_np(model, train_images)
            val_logits = predict_logits_np(model, val_images)
            test_logits = predict_logits_np(model, test_images)

            train_metrics = classification_metrics(train_labels, train_logits)
            val_metrics = classification_metrics(val_labels, val_logits)
            test_metrics = classification_metrics(test_labels, test_logits)

            train_evidence = centered_logits_numpy(train_logits, train_labels)
            test_evidence = centered_logits_numpy(test_logits, test_labels)
            output_scale = float(
                np.quantile(train_evidence, 0.95)
                - np.quantile(train_evidence, 0.05)
            )
            if not np.isfinite(output_scale) or output_scale <= 1e-12:
                raise RuntimeError(
                    f"{family} seed {seed}: degenerate training evidence scale."
                )
            train_median = float(np.median(train_evidence))
            max_test_distance_s = float(
                np.max(np.abs(test_evidence - train_median)) / output_scale
            )
            finite_predictions = bool(
                np.isfinite(train_logits).all()
                and np.isfinite(val_logits).all()
                and np.isfinite(test_logits).all()
            )

            fit_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_nll": best_val_nll,
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
                    "finite_predictions": finite_predictions,
                }
            )

            torch.save(
                {
                    "experiment": "E3b vision fatal controls",
                    "family": family,
                    "seed": seed,
                    "protocol_manifest_sha256": sha256_file(
                        protocol_dir / "manifest_sha256.json"
                    ),
                    "model_contract": contract,
                    "region_slices": region_slices,
                    "normalization_mean_0_1": mean_01,
                    "normalization_std_0_1": std_01,
                    "state_dict": model.state_dict(),
                },
                out_dir / f"{family}_seed_{seed}.pt",
            )

            print(
                f"      epoch={best_epoch:3d} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_NLL={test_metrics['negative_log_likelihood']:.4f} "
                f"s={output_scale:.4f}"
            )

            # Auditing requires derivatives only with respect to the eight
            # region sliders. Freezing parameters reduces CPU memory without
            # changing the function or slider gradients.
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            seed_audit_rows: list[dict[str, Any]] = []
            for audit_id, (test_row, endpoint, true_class) in enumerate(
                zip(audit_idx, audit_images, audit_labels)
            ):
                result = audit_one(
                    model,
                    baseline,
                    endpoint,
                    int(true_class),
                    region_masks,
                    output_scale,
                    chunk_size=args.audit_chunk_size,
                )
                row = {
                    "family": family,
                    "seed": seed,
                    "audit_id": audit_id,
                    "official_test_index": int(test_row),
                    "true_class_id": int(true_class),
                    "output_scale_s": output_scale,
                    **result,
                }
                audit_rows.append(row)
                seed_audit_rows.append(row)

                if (audit_id + 1) % 20 == 0:
                    current_max = max(item["H_over_s"] for item in seed_audit_rows)
                    print(
                        f"      audits {audit_id + 1:3d}/100 | "
                        f"max current H/s={current_max:.3e}"
                    )

    fit_df = pd.DataFrame(fit_rows)
    audit_df = pd.DataFrame(audit_rows)
    fit_df.to_csv(out_dir / "fit_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "control_audits.csv", index=False)

    family_summary: dict[str, Any] = {}
    for family in ["additive", "quadratic"]:
        group = audit_df[audit_df["family"] == family]
        fits = fit_df[fit_df["family"] == family]
        family_summary[family] = {
            "n_fits": int(len(fits)),
            "n_audits": int(len(group)),
            "test_accuracy_min": float(fits["test_accuracy"].min()),
            "test_accuracy_max": float(fits["test_accuracy"].max()),
            "test_accuracy_median": float(fits["test_accuracy"].median()),
            "resolved_rate": float(group["resolved"].mean()),
            "certification_pass_rate": float(group["certification_pass"].mean()),
            "max_R_over_s": float(group["R_over_s"].max()),
            "max_D_over_s": float(group["D_over_s"].max()),
            "max_H_over_s": float(group["H_over_s"].max()),
            "max_abs_transfer_entry": float(group["max_abs_transfer_entry"].max()),
            "max_pot_conservation_error": float(
                group["pot_conservation_error"].max()
            ),
            "max_ig_reconstruction_error": float(
                group["ig_reconstruction_error"].max()
            ),
            "max_nested_quadrature_error": float(
                group["nested_quadrature_error"].max()
            ),
            "max_abs_pot_order_ge2_over_s": float(
                group["max_abs_pot_order_ge2_over_s"].max()
            ),
            "max_abs_pot_order_ge3_over_s": float(
                group["max_abs_pot_order_ge3_over_s"].max()
            ),
            "median_pair_pot_L1_over_s": float(
                group["pot_order2_L1_over_s"].median()
            ),
            "quadrature_order_counts": {
                str(int(order)): int(count)
                for order, count in group["quadrature_order"]
                .value_counts()
                .sort_index()
                .items()
            },
        }

    structural_tolerance = CONTROL_NULL_MAX_H_OVER_S
    pass_flags = {
        "all_predictions_finite": bool(fit_df["finite_predictions"].all()),
        "all_audits_resolved": bool(audit_df["resolved"].all()),
        "all_audits_certified": bool(audit_df["certification_pass"].all()),
        "additive_null": bool(
            family_summary["additive"]["max_H_over_s"]
            <= CONTROL_NULL_MAX_H_OVER_S
        ),
        "quadratic_null": bool(
            family_summary["quadratic"]["max_H_over_s"]
            <= CONTROL_NULL_MAX_H_OVER_S
        ),
        "additive_has_no_material_order_ge2_pots": bool(
            family_summary["additive"]["max_abs_pot_order_ge2_over_s"]
            <= structural_tolerance
        ),
        "quadratic_has_no_material_order_ge3_pots": bool(
            family_summary["quadratic"]["max_abs_pot_order_ge3_over_s"]
            <= structural_tolerance
        ),
    }
    all_pass = bool(all(pass_flags.values()))

    summary = {
        "experiment": "E3b Fashion-MNIST vision fatal controls",
        "protocol_manifest_sha256": sha256_file(
            protocol_dir / "manifest_sha256.json"
        ),
        "model_seeds": MODEL_SEEDS,
        "quadrature_orders": QUAD_ORDERS,
        "training_dtype": "float32",
        "prediction_and_audit_dtype": "float64",
        "control_null_tolerance_H_over_s": CONTROL_NULL_MAX_H_OVER_S,
        "structural_pot_tolerance_over_s": structural_tolerance,
        "n_fits": int(len(fit_df)),
        "n_control_audits": int(len(audit_df)),
        "families": family_summary,
        "pass_flags": pass_flags,
        "all_pass": all_pass,
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(summary, out_dir / "control_summary.json")

    # Compact engineering diagnostic. Paper figures are generated from the
    # frozen CSV/JSON files, never from retraining.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    accuracy_data = [
        fit_df.loc[fit_df["family"] == family, "test_accuracy"].to_numpy()
        for family in ["additive", "quadratic"]
    ]
    boxes = axes[0].boxplot(
        accuracy_data,
        tick_labels=["Additive", "Quadratic"],
        patch_artist=True,
        widths=0.52,
    )
    for patch, color in zip(boxes["boxes"], ["#A9D6E5", "#F6BD60"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("(a) Predictive controls")
    axes[0].grid(True, axis="y", alpha=0.25)

    floor = 1e-18
    null_data = [
        np.log10(
            np.maximum(
                audit_df.loc[audit_df["family"] == family, "H_over_s"].to_numpy(),
                floor,
            )
        )
        for family in ["additive", "quadratic"]
    ]
    boxes = axes[1].boxplot(
        null_data,
        tick_labels=["Additive", "Quadratic"],
        patch_artist=True,
        widths=0.52,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], ["#A9D6E5", "#F6BD60"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].axhline(
        np.log10(CONTROL_NULL_MAX_H_OVER_S),
        color="#222222",
        linestyle="--",
        linewidth=0.9,
    )
    axes[1].set_ylabel(r"$\log_{10}\max(H/s,10^{-18})$")
    axes[1].set_title("(b) Required numerical null")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "control_null_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "control_null_diagnostic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    output_files = sorted(
        path
        for path in out_dir.iterdir()
        if path.name != "control_manifest_sha256.json"
    )
    write_json(
        {
            "hash_algorithm": "SHA-256",
            "protocol_manifest_sha256": sha256_file(
                protocol_dir / "manifest_sha256.json"
            ),
            "files": {path.name: sha256_file(path) for path in output_files},
        },
        out_dir / "control_manifest_sha256.json",
    )

    print()
    print("=" * 84)
    print("E3b — VISION FATAL CONTROL SUMMARY")
    print("=" * 84)
    for family in ["additive", "quadratic"]:
        current = family_summary[family]
        print(f"{family.upper()}")
        print(
            "  test accuracy range              : "
            f"{current['test_accuracy_min']:.4f}--{current['test_accuracy_max']:.4f}"
        )
        print(f"  audits                           : {current['n_audits']}")
        print(f"  resolved rate                    : {current['resolved_rate']:.3f}")
        print(
            "  certification pass rate          : "
            f"{current['certification_pass_rate']:.3f}"
        )
        print(f"  max H/s                          : {current['max_H_over_s']:.3e}")
        print(f"  max R/s                          : {current['max_R_over_s']:.3e}")
        print(f"  max D/s                          : {current['max_D_over_s']:.3e}")
        print(
            "  max |transfer entry|             : "
            f"{current['max_abs_transfer_entry']:.3e}"
        )
        print(
            "  max pot conservation error       : "
            f"{current['max_pot_conservation_error']:.3e}"
        )
        print(
            "  max IG reconstruction error      : "
            f"{current['max_ig_reconstruction_error']:.3e}"
        )
        print(
            "  median pair-pot L1/s             : "
            f"{current['median_pair_pot_L1_over_s']:.3e}"
        )
        print(
            "  quadrature order counts          : "
            f"{current['quadrature_order_counts']}"
        )
        print()

    print("Frozen fatal-control checks:")
    for name, value in pass_flags.items():
        print(f"  {name}: {value}")
    print(f"  all_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir.resolve()}")
    print("=" * 84)

    if not all_pass:
        raise RuntimeError(
            "Vision fatal-control stage FAILED. Do not run the CNN stage until "
            "the predictive/numerical/structural failure is understood."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit E3b vision fatal controls and run 1,000 exhaustive audits."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--protocol-dir", default="./e3_vision_protocol")
    parser.add_argument("--out-dir", default="./e3_vision_controls")
    parser.add_argument(
        "--audit-chunk-size",
        type=int,
        default=2048,
        help="Performance-only batch size for quadrature image evaluations.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
