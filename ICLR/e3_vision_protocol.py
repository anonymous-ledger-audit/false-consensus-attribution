#!/usr/bin/env python3
"""
E3b — Fashion-MNIST vision protocol freeze
===========================================

This script freezes the appendix-only vision experiment BEFORE any model is
fitted and before any attribution quantity is inspected.

Frozen contract
---------------
Dataset
    Fashion-MNIST with its canonical 60,000-image training split and
    10,000-image test split.

Train / validation
    One fixed stratified 51,000 / 9,000 split of the canonical training set,
    seed 20260910.  The canonical 10,000-image test split remains untouched.

Preprocessing
    Pixel intensities are mapped to [0, 1].  A single global mean and standard
    deviation are fitted on the 51,000 training images only.

Explanation representation
    Exactly eight fixed spatial regions in a 2 x 4 grid.  Each region is
    14 x 7 pixels.  These eight region sliders are the explanation features.

Baseline
    An actual training image: the training observation closest in Euclidean
    distance to the training mean image in raw [0, 1] pixel space.

Audit panel
    Exactly 100 fixed canonical-test images, ten per true class, selected with
    seed 20260911.  The same images are used for every family and every seed.

Model seeds
    20260920, 20260921, 20260922, 20260923, 20260924.

Explained scalar
    For endpoint true class y and model logits ell(x),

        g_y(x) = ell_y(x) - mean_{c != y} ell_c(x).

Endpoint game and path
    All 2^8 = 256 Boolean baseline/endpoint region mosaics and the straight
    path on the eight region sliders.  No screening or approximation.

The script writes only data/protocol artifacts.  It does NOT instantiate,
train, or audit a predictor.

Run from the repository root
----------------------------
    python ICLR/e3_vision_protocol.py

Outputs
-------
    ./e3_vision_protocol/
        split_indices.npz
        audit_indices.csv
        audit_images_uint8.npz
        audit_panel.png
        baseline.csv
        baseline_image_uint8.npy
        baseline_region_grid.png
        region_definitions.csv
        region_map.npy
        preprocessing.json
        protocol.json
        manifest_sha256.json

The output directory must be absent or empty.  This script refuses to
overwrite a previous freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

try:
    import torch
    import torchvision
    from torchvision.datasets import FashionMNIST
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise RuntimeError(
        "E3b requires torch and torchvision. Install the matching torchvision "
        "build for your PyTorch installation before running this script."
    ) from exc


# -----------------------------------------------------------------------------
# Frozen constants.  Changing one creates a different protocol.
# -----------------------------------------------------------------------------

PROTOCOL_NAME = "E3b Fashion-MNIST exhaustive region-ledger audit"
PROTOCOL_VERSION = "1.0-freeze"

TRAIN_VAL_SPLIT_SEED = 20260910
AUDIT_SEED = 20260911
MODEL_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]

OFFICIAL_TRAIN_N = 60_000
OFFICIAL_TEST_N = 10_000
TRAIN_N = 51_000
VAL_N = 9_000
AUDIT_PER_CLASS = 10
AUDIT_N = 100

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
N_CLASSES = 10
REGION_ROWS = 2
REGION_COLS = 4
N_REGIONS = REGION_ROWS * REGION_COLS
BOOLEAN_CORNERS = 2**N_REGIONS

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

# The same descriptive operating point and full threshold grid as tabular E3.
FC_KAPPA = 0.02
FC_TAU = 0.05
KAPPA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10]
TAU_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
MAJORITY_SEEDS = 3

QUADRATURE_ORDERS = [16, 32, 64, 128, 256]
CONTROL_NULL_MAX_H_OVER_S = 1e-7

# Architecture and training contracts are recorded here and implemented by the
# stage-specific runners.
FUTURE_MODEL_CONTRACT = {
    "additive_region_control": {
        "definition": (
            "ten logits equal a bias plus the sum of eight independent "
            "98->32->32->10 Softplus region subnetworks"
        ),
        "hidden_width": 32,
        "hidden_depth": 2,
        "activation": "Softplus(beta=1)",
        "max_epochs": 60,
        "patience": 8,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    "quadratic_region_control": {
        "definition": (
            "linear 16-dimensional region embeddings; singleton linear "
            "terms plus all 28 pairwise Hadamard-product terms; linear "
            "ten-logit output"
        ),
        "region_embedding_dimension": 16,
        "n_region_pairs": 28,
        "max_epochs": 80,
        "patience": 10,
        "batch_size": 256,
        "learning_rate": 2e-3,
        "weight_decay": 1e-4,
    },
    "softplus_cnn": {
        "definition": (
            "Conv(1,32,3,pad=1)->Softplus->AvgPool(2); "
            "Conv(32,64,3,pad=1)->Softplus->AvgPool(2); "
            "Linear(64*7*7,128)->Softplus->Linear(128,10)"
        ),
        "conv_channels": [32, 64],
        "kernel_size": 3,
        "padding": 1,
        "pooling": "AveragePool2d(kernel_size=2,stride=2)",
        "fully_connected_width": 128,
        "activation": "Softplus(beta=1)",
        "max_epochs": 40,
        "patience": 7,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "predictive_gate": {
            "finite_logits": True,
            "minimum_test_accuracy": 0.85,
            "maximum_test_centered_logit_distance_from_training_median_in_s": 10.0,
            "gate_is_not_used_for_retraining_or_hyperparameter_selection": True,
        },
    },
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


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
            names = ", ".join(sorted(p.name for p in contents[:8]))
            if len(contents) > 8:
                names += ", ..."
            raise RuntimeError(
                f"Refusing to overwrite nonempty protocol directory: {path.resolve()}\n"
                f"Existing entries: {names}\n"
                "Keep the existing freeze or choose a new --out-dir."
            )
    else:
        path.mkdir(parents=True, exist_ok=False)


def to_numpy_int64(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def to_numpy_uint8_images(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 images, received {array.dtype}.")
    return array


def squared_distances_to_mean_image(
    images_uint8: np.ndarray,
    mean_image_01: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute exact float64 squared L2 distances without a large temporary."""
    result = np.empty(len(images_uint8), dtype=np.float64)
    for start in range(0, len(images_uint8), batch_size):
        stop = min(start + batch_size, len(images_uint8))
        batch = images_uint8[start:stop].astype(np.float64) / 255.0
        diff = batch - mean_image_01[None, :, :]
        result[start:stop] = np.einsum("nhw,nhw->n", diff, diff)
    return result


def global_pixel_moments_01(
    images_uint8: np.ndarray,
    batch_size: int = 4096,
) -> tuple[float, float]:
    """Train-only global mean/std without a full float64 image copy."""
    total = 0.0
    total_squares = 0.0
    for start in range(0, len(images_uint8), batch_size):
        stop = min(start + batch_size, len(images_uint8))
        batch = images_uint8[start:stop].astype(np.float64)
        total += float(batch.sum())
        total_squares += float(np.einsum("nhw,nhw->", batch, batch))

    n_pixels = int(images_uint8.size)
    mean_01 = total / (255.0 * n_pixels)
    second_moment_01 = total_squares / ((255.0**2) * n_pixels)
    variance_01 = max(0.0, second_moment_01 - mean_01**2)
    return float(mean_01), float(np.sqrt(variance_01))


def build_region_map() -> tuple[np.ndarray, pd.DataFrame]:
    if IMAGE_HEIGHT % REGION_ROWS != 0 or IMAGE_WIDTH % REGION_COLS != 0:
        raise RuntimeError("Image dimensions are not divisible by the frozen grid.")

    region_height = IMAGE_HEIGHT // REGION_ROWS
    region_width = IMAGE_WIDTH // REGION_COLS
    region_map = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), -1, dtype=np.int16)
    rows: list[dict[str, Any]] = []

    region_id = 0
    for grid_row in range(REGION_ROWS):
        row_start = grid_row * region_height
        row_stop = (grid_row + 1) * region_height
        for grid_col in range(REGION_COLS):
            col_start = grid_col * region_width
            col_stop = (grid_col + 1) * region_width
            region_map[row_start:row_stop, col_start:col_stop] = region_id
            rows.append(
                {
                    "region_id_zero_based": region_id,
                    "region_label": f"R{region_id + 1}",
                    "grid_row_zero_based": grid_row,
                    "grid_col_zero_based": grid_col,
                    "row_start_inclusive": row_start,
                    "row_stop_exclusive": row_stop,
                    "col_start_inclusive": col_start,
                    "col_stop_exclusive": col_stop,
                    "n_pixels": (row_stop - row_start) * (col_stop - col_start),
                }
            )
            region_id += 1

    return region_map, pd.DataFrame(rows)


def save_region_grid_preview(image_uint8: np.ndarray, path: Path) -> None:
    scale = 10
    image = Image.fromarray(image_uint8, mode="L").resize(
        (IMAGE_WIDTH * scale, IMAGE_HEIGHT * scale),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    draw = ImageDraw.Draw(image)

    for col in range(1, REGION_COLS):
        x = col * (IMAGE_WIDTH // REGION_COLS) * scale
        draw.line([(x, 0), (x, IMAGE_HEIGHT * scale)], fill=(220, 45, 45), width=2)
    for row in range(1, REGION_ROWS):
        y = row * (IMAGE_HEIGHT // REGION_ROWS) * scale
        draw.line([(0, y), (IMAGE_WIDTH * scale, y)], fill=(220, 45, 45), width=2)

    region_height = IMAGE_HEIGHT // REGION_ROWS
    region_width = IMAGE_WIDTH // REGION_COLS
    region_id = 0
    for grid_row in range(REGION_ROWS):
        for grid_col in range(REGION_COLS):
            x = grid_col * region_width * scale + 5
            y = grid_row * region_height * scale + 4
            draw.text(
                (x, y),
                f"R{region_id + 1}",
                fill=(255, 230, 40),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            region_id += 1

    image.save(path)


def save_audit_panel(images_uint8: np.ndarray, path: Path) -> None:
    """Save the 100 frozen endpoints as ten class rows by ten columns."""
    if images_uint8.shape != (AUDIT_N, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(f"Unexpected audit image shape: {images_uint8.shape}")

    scale = 4
    tile_w = IMAGE_WIDTH * scale
    tile_h = IMAGE_HEIGHT * scale
    panel = Image.new("L", (AUDIT_PER_CLASS * tile_w, N_CLASSES * tile_h), color=0)

    for index, array in enumerate(images_uint8):
        row = index // AUDIT_PER_CLASS
        col = index % AUDIT_PER_CLASS
        tile = Image.fromarray(array, mode="L").resize(
            (tile_w, tile_h),
            resample=Image.Resampling.NEAREST,
        )
        panel.paste(tile, (col * tile_w, row * tile_h))

    panel.save(path)


def class_count_dict(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=N_CLASSES)
    return {str(class_id): int(counts[class_id]) for class_id in range(N_CLASSES)}


# -----------------------------------------------------------------------------
# Protocol freeze
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    require_empty_output_directory(out_dir)

    # Download/load the canonical dataset.  No transforms are used because the
    # freeze stores the original uint8 observations and its own preprocessing.
    official_train = FashionMNIST(root=data_dir, train=True, download=True)
    official_test = FashionMNIST(root=data_dir, train=False, download=True)

    train_images_all = to_numpy_uint8_images(official_train.data)
    train_targets_all = to_numpy_int64(official_train.targets)
    test_images = to_numpy_uint8_images(official_test.data)
    test_targets = to_numpy_int64(official_test.targets)

    expected_train_shape = (OFFICIAL_TRAIN_N, IMAGE_HEIGHT, IMAGE_WIDTH)
    expected_test_shape = (OFFICIAL_TEST_N, IMAGE_HEIGHT, IMAGE_WIDTH)
    if train_images_all.shape != expected_train_shape:
        raise RuntimeError(
            f"Expected official train shape {expected_train_shape}, "
            f"received {train_images_all.shape}."
        )
    if test_images.shape != expected_test_shape:
        raise RuntimeError(
            f"Expected official test shape {expected_test_shape}, "
            f"received {test_images.shape}."
        )

    # Fixed stratified 51k/9k split of the canonical 60k training observations.
    official_indices = np.arange(OFFICIAL_TRAIN_N, dtype=np.int64)
    train_idx, val_idx = train_test_split(
        official_indices,
        test_size=VAL_N,
        random_state=TRAIN_VAL_SPLIT_SEED,
        shuffle=True,
        stratify=train_targets_all,
    )
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
    canonical_test_idx = np.arange(OFFICIAL_TEST_N, dtype=np.int64)

    # Train-only global normalization and train mean image.
    train_raw = train_images_all[train_idx]
    pixel_mean_01, pixel_std_01 = global_pixel_moments_01(train_raw)
    if pixel_std_01 <= 1e-12:
        raise RuntimeError("Degenerate train-only pixel standard deviation.")

    mean_image_01 = train_raw.mean(axis=0, dtype=np.float64) / 255.0

    # Observed training baseline nearest the train mean image.
    baseline_dist2 = squared_distances_to_mean_image(train_raw, mean_image_01)
    baseline_train_position = int(np.argmin(baseline_dist2))
    baseline_official_train_index = int(train_idx[baseline_train_position])
    baseline_image = train_images_all[baseline_official_train_index].copy()
    baseline_class = int(train_targets_all[baseline_official_train_index])
    baseline_l2 = float(np.sqrt(baseline_dist2[baseline_train_position]))
    baseline_rms = float(baseline_l2 / np.sqrt(IMAGE_HEIGHT * IMAGE_WIDTH))

    # Eight fixed, complete, disjoint regions.
    region_map, region_df = build_region_map()

    # Exactly ten canonical-test endpoints per true class.
    rng = np.random.default_rng(AUDIT_SEED)
    audit_records: list[dict[str, Any]] = []
    chosen_test_indices: list[int] = []

    for class_id in range(N_CLASSES):
        candidates = np.flatnonzero(test_targets == class_id)
        if len(candidates) < AUDIT_PER_CLASS:
            raise RuntimeError(
                f"Class {class_id} has only {len(candidates)} test images."
            )
        picked = np.sort(
            rng.choice(candidates, size=AUDIT_PER_CLASS, replace=False).astype(np.int64)
        )
        chosen_test_indices.extend(int(index) for index in picked)

    audit_test_idx = np.asarray(chosen_test_indices, dtype=np.int64)
    audit_images = test_images[audit_test_idx].copy()
    audit_targets = test_targets[audit_test_idx].copy()

    for audit_id, (test_index, class_id) in enumerate(
        zip(audit_test_idx, audit_targets)
    ):
        audit_records.append(
            {
                "audit_id": int(audit_id),
                "official_test_index": int(test_index),
                "true_class_id": int(class_id),
                "true_class_name": CLASS_NAMES[int(class_id)],
            }
        )
    audit_df = pd.DataFrame(audit_records)

    # ------------------------------------------------------------------
    # Hard protocol checks, all evaluated before anything is written.
    # ------------------------------------------------------------------
    split_disjoint = len(np.intersect1d(train_idx, val_idx)) == 0
    split_complete = np.array_equal(
        np.sort(np.concatenate([train_idx, val_idx])), official_indices
    )
    split_sizes_exact = len(train_idx) == TRAIN_N and len(val_idx) == VAL_N
    stratified_train_exact = np.all(
        np.bincount(train_targets_all[train_idx], minlength=N_CLASSES) == 5100
    )
    stratified_val_exact = np.all(
        np.bincount(train_targets_all[val_idx], minlength=N_CLASSES) == 900
    )

    baseline_in_train = bool(np.any(train_idx == baseline_official_train_index))
    baseline_not_in_validation = not bool(
        np.any(val_idx == baseline_official_train_index)
    )
    baseline_is_observed = bool(
        np.array_equal(
            baseline_image,
            train_images_all[baseline_official_train_index],
        )
    )

    audit_n_exact = len(audit_test_idx) == AUDIT_N
    audit_unique = len(np.unique(audit_test_idx)) == AUDIT_N
    audit_test_only = bool(
        np.all((audit_test_idx >= 0) & (audit_test_idx < OFFICIAL_TEST_N))
    )
    audit_exactly_ten_per_class = bool(
        np.all(np.bincount(audit_targets, minlength=N_CLASSES) == AUDIT_PER_CLASS)
    )

    region_ids = np.unique(region_map)
    region_complete = bool(
        np.array_equal(region_ids, np.arange(N_REGIONS, dtype=np.int16))
        and np.all(region_map >= 0)
    )
    region_counts = np.bincount(region_map.ravel(), minlength=N_REGIONS)
    regions_equal_size = bool(np.all(region_counts == 98))
    region_total_exact = int(region_counts.sum()) == IMAGE_HEIGHT * IMAGE_WIDTH

    normalized_mean = (pixel_mean_01 - pixel_mean_01) / pixel_std_01
    normalized_std = pixel_std_01 / pixel_std_01
    train_normalization_check = bool(
        abs(normalized_mean) <= 1e-14 and abs(normalized_std - 1.0) <= 1e-14
    )

    checks = {
        "canonical_train_shape": train_images_all.shape == expected_train_shape,
        "canonical_test_shape": test_images.shape == expected_test_shape,
        "split_disjoint": split_disjoint,
        "split_complete": split_complete,
        "split_sizes_exact": split_sizes_exact,
        "stratified_train_exactly_5100_per_class": bool(stratified_train_exact),
        "stratified_validation_exactly_900_per_class": bool(stratified_val_exact),
        "baseline_in_train": baseline_in_train,
        "baseline_not_in_validation": baseline_not_in_validation,
        "baseline_is_observed_image": baseline_is_observed,
        "audit_n_exact": audit_n_exact,
        "audit_unique": audit_unique,
        "audit_test_only": audit_test_only,
        "audit_exactly_10_per_class": audit_exactly_ten_per_class,
        "region_ids_exactly_0_through_7": region_complete,
        "regions_exactly_98_pixels_each": regions_equal_size,
        "region_map_covers_exactly_784_pixels": region_total_exact,
        "train_only_normalization_check": train_normalization_check,
    }
    checks["all_pass"] = bool(all(checks.values()))

    if not checks["all_pass"]:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Protocol checks failed before write: {failed}")

    # ------------------------------------------------------------------
    # Write immutable protocol artifacts.
    # ------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "split_indices.npz",
        official_train_train_idx=train_idx,
        official_train_validation_idx=val_idx,
        canonical_test_idx=canonical_test_idx,
        baseline_official_train_index=np.asarray(
            [baseline_official_train_index], dtype=np.int64
        ),
        audit_official_test_idx=audit_test_idx,
    )

    audit_df.to_csv(out_dir / "audit_indices.csv", index=False)
    np.savez_compressed(
        out_dir / "audit_images_uint8.npz",
        images_uint8=audit_images,
        true_class_id=audit_targets,
        official_test_index=audit_test_idx,
    )

    baseline_df = pd.DataFrame(
        [
            {
                "official_train_index": baseline_official_train_index,
                "train_position": baseline_train_position,
                "true_class_id": baseline_class,
                "true_class_name": CLASS_NAMES[baseline_class],
                "l2_distance_to_train_mean_image_in_0_1_space": baseline_l2,
                "rms_distance_to_train_mean_image_in_0_1_space": baseline_rms,
            }
        ]
    )
    baseline_df.to_csv(out_dir / "baseline.csv", index=False)
    np.save(out_dir / "baseline_image_uint8.npy", baseline_image)

    region_df.to_csv(out_dir / "region_definitions.csv", index=False)
    np.save(out_dir / "region_map.npy", region_map)

    save_region_grid_preview(
        baseline_image,
        out_dir / "baseline_region_grid.png",
    )
    save_audit_panel(audit_images, out_dir / "audit_panel.png")

    preprocessing = {
        "raw_image_dtype": "uint8",
        "raw_pixel_range": [0, 255],
        "model_pixel_scaling": "x_01 = x_uint8 / 255.0",
        "model_normalization": "(x_01 - global_train_pixel_mean) / global_train_pixel_std",
        "statistics_fit_split": "51,000-image training split only",
        "global_train_pixel_mean_0_1": pixel_mean_01,
        "global_train_pixel_std_0_1": pixel_std_01,
        "train_mean_image_0_1": mean_image_01.tolist(),
        "baseline_selection_space": (
            "raw [0,1] pixels; Euclidean distance to train mean image"
        ),
        "note": (
            "The affine model normalization commutes with the frozen "
            "baseline-to-endpoint region interpolation."
        ),
    }
    write_json(preprocessing, out_dir / "preprocessing.json")

    dataset_hashes = {
        "official_train_images_uint8": sha256_array(train_images_all),
        "official_train_targets_int64": sha256_array(train_targets_all),
        "official_test_images_uint8": sha256_array(test_images),
        "official_test_targets_int64": sha256_array(test_targets),
    }

    protocol = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": {
            "name": "Fashion-MNIST",
            "provider": "torchvision.datasets.FashionMNIST",
            "canonical_training_rows": OFFICIAL_TRAIN_N,
            "canonical_test_rows": OFFICIAL_TEST_N,
            "image_shape": [1, IMAGE_HEIGHT, IMAGE_WIDTH],
            "n_classes": N_CLASSES,
            "class_names": CLASS_NAMES,
            "content_sha256": dataset_hashes,
        },
        "split": {
            "source": "canonical 60,000-image training split",
            "method": "fixed stratified train/validation split",
            "seed": TRAIN_VAL_SPLIT_SEED,
            "n_train": TRAIN_N,
            "n_validation": VAL_N,
            "n_test": OFFICIAL_TEST_N,
            "train_class_counts": class_count_dict(train_targets_all[train_idx]),
            "validation_class_counts": class_count_dict(train_targets_all[val_idx]),
            "test_class_counts": class_count_dict(test_targets),
        },
        "preprocessing": {
            "fit_on_training_only": True,
            "global_train_pixel_mean_0_1": pixel_mean_01,
            "global_train_pixel_std_0_1": pixel_std_01,
        },
        "baseline": {
            "selection_rule": (
                "observed training image minimizing Euclidean distance to "
                "the training mean image in raw [0,1] pixel space"
            ),
            "official_train_index": baseline_official_train_index,
            "train_position": baseline_train_position,
            "true_class_id": baseline_class,
            "true_class_name": CLASS_NAMES[baseline_class],
            "l2_distance_to_train_mean_image": baseline_l2,
            "rms_distance_to_train_mean_image": baseline_rms,
        },
        "explanation_representation": {
            "n_region_features": N_REGIONS,
            "grid": [REGION_ROWS, REGION_COLS],
            "region_shape_pixels": [
                IMAGE_HEIGHT // REGION_ROWS,
                IMAGE_WIDTH // REGION_COLS,
            ],
            "pixels_per_region": 98,
            "region_order": "row-major",
            "endpoint_game_boolean_corners": BOOLEAN_CORNERS,
            "screening": "none",
            "coalition_game_approximation": "none",
        },
        "audit_panel": {
            "source": "canonical official test split",
            "selection_before_model_fitting": True,
            "seed": AUDIT_SEED,
            "n_images": AUDIT_N,
            "images_per_true_class": AUDIT_PER_CLASS,
            "class_counts": class_count_dict(audit_targets),
            "same_endpoints_for_every_family_and_seed": True,
        },
        "model_seeds": MODEL_SEEDS,
        "explained_scalar": {
            "name": "true-class centered logit",
            "formula": "g_y(x) = logit_y(x) - mean_{c != y} logit_c(x)",
            "endpoint_true_label_fixed_independently_of_model": True,
            "original_logit_scale": True,
        },
        "path": {
            "definition": (
                "straight line from the frozen observed baseline image to "
                "the frozen endpoint, expressed through eight region sliders"
            ),
            "region_slider_formula": (
                "x(t) = x0 + sum_r t_r * mask_r * (x1 - x0)"
            ),
        },
        "normalization_scale": {
            "definition": (
                "s = Q0.95(g_y(x) over training observations using their "
                "true labels) - Q0.05 of the same values"
            ),
            "computed_per_fitted_model": True,
            "uses_training_outputs_only": True,
        },
        "descriptive_false_consensus": {
            "fixed_operating_point": {
                "D_over_s_max": FC_KAPPA,
                "H_over_s_min": FC_TAU,
            },
            "complete_kappa_grid": KAPPA_GRID,
            "complete_tau_grid": TAU_GRID,
            "robust_endpoint_definition": (
                f"criterion holds in at least {MAJORITY_SEEDS} of 5 fitted-model seeds"
            ),
            "binary_event_is_not_the_empirical_thesis": True,
        },
        "numerical_contract": {
            "audit_dtype": "float64",
            "adaptive_gauss_legendre_orders": QUADRATURE_ORDERS,
            "numerical_tolerance": "max(1e-10, 1e-8 * max(s, 1))",
            "certificate_tolerance_multiplier": 10.0,
            "control_max_H_over_s": CONTROL_NULL_MAX_H_OVER_S,
            "all_audits_must_resolve": True,
            "all_audits_must_certify": True,
        },
        "future_model_contract": FUTURE_MODEL_CONTRACT,
        "confirmatory_audit_count_if_all_three_families_run": (
            3 * len(MODEL_SEEDS) * AUDIT_N
        ),
        "protocol_checks": checks,
        "software_versions_at_freeze": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
        },
    }
    write_json(protocol, out_dir / "protocol.json")

    # Hash every frozen artifact except the manifest that contains the hashes.
    artifact_paths = sorted(
        path for path in out_dir.iterdir() if path.name != "manifest_sha256.json"
    )
    manifest = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "hash_algorithm": "SHA-256",
        "files": {path.name: sha256_file(path) for path in artifact_paths},
    }
    write_json(manifest, out_dir / "manifest_sha256.json")

    # ------------------------------------------------------------------
    # Terminal freeze report.
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("E3b — FASHION-MNIST VISION PROTOCOL FREEZE")
    print("=" * 80)
    print(f"Canonical train / test          : {OFFICIAL_TRAIN_N} / {OFFICIAL_TEST_N}")
    print(f"Train / validation / test       : {TRAIN_N} / {VAL_N} / {OFFICIAL_TEST_N}")
    print(f"Train-validation split seed     : {TRAIN_VAL_SPLIT_SEED}")
    print()
    print(f"Train-only pixel mean [0,1]     : {pixel_mean_01:.12f}")
    print(f"Train-only pixel std [0,1]      : {pixel_std_01:.12f}")
    print()
    print(f"Baseline official train index   : {baseline_official_train_index}")
    print(f"Baseline train position         : {baseline_train_position}")
    print(f"Baseline true class             : {baseline_class} ({CLASS_NAMES[baseline_class]})")
    print(f"Baseline L2 distance to mean    : {baseline_l2:.6f}")
    print(f"Baseline RMS distance to mean   : {baseline_rms:.6f}")
    print()
    print(f"Frozen explanation regions      : {N_REGIONS} (2 x 4; 98 pixels each)")
    print(f"Complete Boolean corners        : {BOOLEAN_CORNERS}")
    print()
    print(f"Audit seed                       : {AUDIT_SEED}")
    print(f"Audit endpoints                  : {AUDIT_N}")
    print("Audit count by true class        :")
    audit_counts = np.bincount(audit_targets, minlength=N_CLASSES)
    for class_id, count in enumerate(audit_counts):
        print(f"  {class_id}: {CLASS_NAMES[class_id]:<12s} {int(count):2d}")
    print()
    print(f"Frozen future model seeds        : {MODEL_SEEDS}")
    print(f"Future complete audits           : 3 x 5 x 100 = {3 * 5 * 100}")
    print()
    print("Protocol checks:")
    for name, value in checks.items():
        print(f"  {name}: {value}")
    print()
    print(f"Outputs                          : {out_dir.resolve()}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the appendix-only E3b Fashion-MNIST protocol."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Fashion-MNIST download/cache directory (default: ./data).",
    )
    parser.add_argument(
        "--out-dir",
        default="./e3_vision_protocol",
        help="New immutable protocol directory (default: ./e3_vision_protocol).",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
