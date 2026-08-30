#!/usr/bin/env python3
"""
E3b — Publication artifacts from frozen Fashion-MNIST audits
================================================================

Read-only paper-artifact stage.  This script never imports PyTorch, loads a
checkpoint, fits a model, recomputes an attribution, changes an endpoint, or
selects a threshold.  It verifies and summarizes the already-frozen V1/V2/V3
outputs, then creates the complete appendix figure and table set.

Run from the repository root
----------------------------
    python ICLR/e3_vision_paper_artifacts.py

Default inputs
--------------
    ./e3_vision_protocol/
    ./e3_vision_controls/
    ./e3_vision_cnn/

Default outputs
---------------
    ./ICLR/paper_figures/e3b_app_protocol.{pdf,png}
    ./ICLR/paper_figures/e3b_app_result_summary.{pdf,png}
    ./ICLR/paper_figures/e3b_app_stable_witnesses.{pdf,png}
    ./ICLR/paper_figures/tab_e3b_model_summary.tex
    ./ICLR/paper_figures/tab_e3b_classwise.tex
    ./ICLR/paper_figures/tab_e3b_numerics.tex
    ./ICLR/paper_figures/tab_e3b_threshold_surface.tex
    ./ICLR/paper_figures/e3b_appendix_insert.tex
    ./ICLR/paper_figures/e3b_paper_values.json
    ./ICLR/paper_figures/e3b_artifact_manifest_sha256.json

Scientific units
----------------
There are 100 fixed endpoints and five independently fitted models.  Pooled
seed--endpoint medians are retained because they are the preregistered
descriptive summaries reported by V3.  Distributional paper panels use one
seed-median value per endpoint.  Binary robustness counts an endpoint only
when the event holds in at least three of five fits.

The two strict witnesses are not selected by visual appearance.  They are the
complete set of endpoints satisfying the original strict criterion in at
least three of five fits.  Within each endpoint, the displayed fit is the
smallest frozen seed satisfying the criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXPECTED_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
EXPECTED_ENDPOINTS = 100
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
SHORT_CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot",
]

COLORS = {
    "additive": "#7F8C8D",
    "quadratic": "#4C78A8",
    "cnn": "#8E5EA2",
    "hidden": "#8E5EA2",
    "strict": "#D55E00",
    "robust": "#111111",
    "bshap": "#4C78A8",
    "ig": "#E07A5F",
    "orders": "#2A9D8F",
}

CERTIFICATE_COLUMNS = [
    "pot_conservation_error",
    "bshap_reconstruction_error",
    "ig_reconstruction_error",
    "bshap_completeness_error",
    "ig_completeness_error",
    "margin_gap_error",
    "interior_mobius_reconstruction_error",
]


# =============================================================================
# I/O, verification, and formatting
# =============================================================================


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.55,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path.resolve())
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_text(value: str, path: Path) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def verify_manifest(stage_dir: Path, manifest_name: str) -> dict[str, Any]:
    """Verify every paper-relevant artifact; checkpoints are never consumed."""
    manifest = read_json(stage_dir / manifest_name)
    checked: dict[str, str] = {}
    skipped_checkpoints: list[str] = []
    for name, expected_hash in manifest["files"].items():
        path = stage_dir / name
        if path.suffix.lower() == ".pt":
            skipped_checkpoints.append(name)
            continue
        if not path.exists():
            raise RuntimeError(f"Manifest-listed artifact is missing: {path.resolve()}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen artifact changed: {path.resolve()}\n"
                f"expected {expected_hash}\nactual   {actual_hash}"
            )
        checked[name] = actual_hash
    return {
        "manifest_sha256": sha256_file(stage_dir / manifest_name),
        "verified_non_checkpoint_files": checked,
        "ignored_unused_checkpoints": sorted(skipped_checkpoints),
    }


def parse_bool(df: pd.DataFrame, column: str, label: str) -> None:
    if df[column].dtype == bool:
        return
    parsed = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise RuntimeError(f"{label}: cannot parse Boolean column {column}.")
    df[column] = parsed.astype(bool)


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    result = str(value)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def latex_number(value: float, decimals: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    if exponent <= -3 or exponent >= 4:
        coefficient = value / (10.0**exponent)
        return rf"${coefficient:.2f}\!\times\!10^{{{exponent}}}$"
    return f"${value:.{decimals}f}$"


def save_figure(fig: mpl.figure.Figure, base: Path, dpi: int) -> list[Path]:
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return [pdf_path, png_path]


# =============================================================================
# Data validation and paired summaries
# =============================================================================


def paired_threshold_surface(
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
            recurrence = event.groupby(audit_df["audit_id"]).sum()
            majority[tau_i, kappa_i] = int(
                (recurrence >= MAJORITY_SEEDS).sum()
            )
            at_least_one[tau_i, kappa_i] = int((recurrence >= 1).sum())
            all_five[tau_i, kappa_i] = int((recurrence == 5).sum())
    return majority, at_least_one, all_five


def validate_audits(
    df: pd.DataFrame,
    *,
    label: str,
    expected_rows: int,
    expected_families: Iterable[str],
    protocol_audit: pd.DataFrame,
) -> None:
    required = {
        "family",
        "seed",
        "audit_id",
        "official_test_index",
        "true_class_id",
        "D_over_s",
        "H_over_s",
        "resolved",
        "certification_pass",
        "quadrature_order",
        "numerical_tolerance",
        "nested_quadrature_error",
        *CERTIFICATE_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{label}: missing columns {sorted(missing)}")
    for column in ["resolved", "certification_pass"]:
        parse_bool(df, column, label)
    if len(df) != expected_rows:
        raise RuntimeError(f"{label}: expected {expected_rows} rows, found {len(df)}")
    if sorted(df["seed"].astype(int).unique().tolist()) != EXPECTED_SEEDS:
        raise RuntimeError(f"{label}: model seeds differ from the frozen contract.")
    if set(df["family"].unique()) != set(expected_families):
        raise RuntimeError(f"{label}: unexpected fitted families.")
    expected_per_pair = 1
    if df.duplicated(["family", "seed", "audit_id"]).any():
        raise RuntimeError(f"{label}: duplicate family/seed/audit rows.")
    counts = df.groupby(["family", "audit_id"]).size()
    if not (counts == len(EXPECTED_SEEDS) * expected_per_pair).all():
        raise RuntimeError(f"{label}: an endpoint does not have five fits per family.")
    if not df["resolved"].all() or not df["certification_pass"].all():
        raise RuntimeError(f"{label}: unresolved or uncertified audit.")
    numerical = df[
        ["D_over_s", "H_over_s", "quadrature_order", "numerical_tolerance"]
        + CERTIFICATE_COLUMNS
    ].to_numpy(dtype=float)
    if not np.isfinite(numerical).all():
        raise RuntimeError(f"{label}: non-finite numerical result.")

    identity = (
        df[["audit_id", "official_test_index", "true_class_id"]]
        .drop_duplicates()
        .sort_values("audit_id")
        .reset_index(drop=True)
    )
    expected_identity = protocol_audit[
        ["audit_id", "official_test_index", "true_class_id"]
    ].sort_values("audit_id").reset_index(drop=True)
    if not identity.equals(expected_identity):
        raise RuntimeError(f"{label}: endpoint identities differ from V1.")


def load_and_validate(
    protocol_dir: Path,
    controls_dir: Path,
    cnn_dir: Path,
) -> dict[str, Any]:
    verification = {
        "protocol": verify_manifest(protocol_dir, "manifest_sha256.json"),
        "controls": verify_manifest(controls_dir, "control_manifest_sha256.json"),
        "cnn": verify_manifest(cnn_dir, "cnn_manifest_sha256.json"),
    }

    protocol = read_json(protocol_dir / "protocol.json")
    controls_summary = read_json(controls_dir / "control_summary.json")
    cnn_summary = read_json(cnn_dir / "cnn_summary.json")
    if not protocol["protocol_checks"].get("all_pass", False):
        raise RuntimeError("V1 protocol did not pass.")
    if not controls_summary.get("all_pass", False):
        raise RuntimeError("V2 fatal controls did not pass.")
    if not cnn_summary.get("all_hard_checks_pass", False):
        raise RuntimeError("V3 CNN did not pass.")

    audit_index = pd.read_csv(protocol_dir / "audit_indices.csv")
    if len(audit_index) != EXPECTED_ENDPOINTS:
        raise RuntimeError("V1 does not contain exactly 100 endpoints.")
    if audit_index["audit_id"].tolist() != list(range(EXPECTED_ENDPOINTS)):
        raise RuntimeError("V1 audit IDs are not exactly 0 through 99.")
    if not (
        audit_index.groupby("true_class_id").size().to_numpy() == 10
    ).all():
        raise RuntimeError("V1 audit panel is not exactly ten images per class.")

    control_audits = pd.read_csv(controls_dir / "control_audits.csv")
    cnn_audits = pd.read_csv(cnn_dir / "cnn_audits.csv")
    control_fits = pd.read_csv(controls_dir / "fit_metrics.csv")
    cnn_fits = pd.read_csv(cnn_dir / "fit_metrics.csv")
    endpoint_stability = pd.read_csv(cnn_dir / "endpoint_stability.csv")
    class_summary = pd.read_csv(cnn_dir / "class_summary.csv")

    validate_audits(
        control_audits,
        label="V2 controls",
        expected_rows=1000,
        expected_families=["additive", "quadratic"],
        protocol_audit=audit_index,
    )
    validate_audits(
        cnn_audits,
        label="V3 CNN",
        expected_rows=500,
        expected_families=["softplus_cnn"],
        protocol_audit=audit_index,
    )
    for column in ["predictive_gate", "finite_logits"]:
        parse_bool(cnn_fits, column, "V3 CNN fits")
    if len(control_fits) != 10 or len(cnn_fits) != 5:
        raise RuntimeError("Unexpected fitted-model row counts.")
    if not cnn_fits["predictive_gate"].all():
        raise RuntimeError("At least one frozen CNN predictive gate failed.")

    for column in ["false_consensus_primary", "endpoint_correct"]:
        parse_bool(cnn_audits, column, "V3 CNN")
    recomputed_fc = (
        (cnn_audits["D_over_s"] <= FC_KAPPA)
        & (cnn_audits["H_over_s"] >= FC_TAU)
    )
    if not np.array_equal(recomputed_fc.to_numpy(), cnn_audits["false_consensus_primary"]):
        raise RuntimeError("Stored strict-FC flags do not reproduce.")

    majority, one, five = paired_threshold_surface(cnn_audits)
    stored_surface = np.asarray(
        cnn_summary["paired_threshold_surface"][
            "majority_3_of_5_matrix_rows_tau_cols_kappa"
        ],
        dtype=int,
    )
    if not np.array_equal(majority, stored_surface):
        raise RuntimeError("Stored paired threshold surface does not reproduce.")

    audit_npz = np.load(protocol_dir / "audit_images_uint8.npz")
    audit_images = audit_npz["images_uint8"]
    baseline_image = np.load(protocol_dir / "baseline_image_uint8.npy")
    region_definitions = pd.read_csv(protocol_dir / "region_definitions.csv")
    if audit_images.shape != (100, 28, 28) or baseline_image.shape != (28, 28):
        raise RuntimeError("Frozen image arrays have unexpected shapes.")
    if not np.array_equal(
        audit_npz["official_test_index"],
        audit_index["official_test_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Frozen audit image identities do not reproduce.")

    # A final independent recurrence check against V3's endpoint table.
    recurrence = (
        cnn_audits.groupby("audit_id")["false_consensus_primary"]
        .sum()
        .astype(int)
    )
    stored_recurrence = endpoint_stability.set_index("audit_id")[
        "strict_fc_seed_count"
    ].astype(int)
    if not recurrence.equals(stored_recurrence):
        raise RuntimeError("Endpoint recurrence table does not reproduce.")

    return {
        "verification": verification,
        "protocol": protocol,
        "controls_summary": controls_summary,
        "cnn_summary": cnn_summary,
        "audit_index": audit_index,
        "control_audits": control_audits,
        "cnn_audits": cnn_audits,
        "control_fits": control_fits,
        "cnn_fits": cnn_fits,
        "endpoint_stability": endpoint_stability,
        "class_summary": class_summary,
        "audit_images": audit_images,
        "baseline_image": baseline_image,
        "region_definitions": region_definitions,
        "majority_surface": majority,
        "one_surface": one,
        "five_surface": five,
    }


# =============================================================================
# Figures
# =============================================================================


def draw_region_grid(
    ax: mpl.axes.Axes,
    *,
    labels: bool,
    color: str = "#D55E00",
    linewidth: float = 0.65,
) -> None:
    ax.axvline(6.5, color=color, linewidth=linewidth)
    ax.axvline(13.5, color=color, linewidth=linewidth)
    ax.axvline(20.5, color=color, linewidth=linewidth)
    ax.axhline(13.5, color=color, linewidth=linewidth)
    if labels:
        centers = [(3, 2), (10, 2), (17, 2), (24, 2), (3, 16), (10, 16), (17, 16), (24, 16)]
        for region, (x, y) in enumerate(centers, start=1):
            ax.text(
                x,
                y,
                f"R{region}",
                color="#FFF3B0",
                fontsize=5.8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.32, "pad": 0.25, "edgecolor": "none"},
            )


def make_protocol_figure(data: dict[str, Any], out_dir: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(7.05, 1.72))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.33, 1, 1, 1, 1, 1],
        wspace=0.10,
        hspace=0.36,
    )

    baseline_ax = fig.add_subplot(grid[:, 0])
    baseline_ax.imshow(data["baseline_image"], cmap="gray", vmin=0, vmax=255)
    draw_region_grid(baseline_ax, labels=True)
    baseline_ax.set_title("Observed baseline\n(class: Shirt)", fontsize=7.3, pad=2.5)
    baseline_ax.set_xticks([])
    baseline_ax.set_yticks([])
    for spine in baseline_ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.7)

    audit_index = data["audit_index"]
    for class_id in range(10):
        row = class_id // 5
        col = class_id % 5 + 1
        ax = fig.add_subplot(grid[row, col])
        audit_id = int(
            audit_index.loc[audit_index["true_class_id"] == class_id, "audit_id"].min()
        )
        ax.imshow(data["audit_images"][audit_id], cmap="gray", vmin=0, vmax=255)
        ax.set_title(SHORT_CLASS_NAMES[class_id], fontsize=6.5, pad=1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.515,
        0.995,
        "One deterministically displayed endpoint per class (100 frozen endpoints; 10/class)",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#333333",
    )
    return save_figure(fig, out_dir / "e3b_app_protocol", dpi)


def endpoint_median_frame(df: pd.DataFrame, family: str) -> pd.DataFrame:
    selected = df[df["family"] == family].copy()
    numeric = selected.select_dtypes(include=[np.number]).columns.tolist()
    medians = selected.groupby("audit_id", as_index=False)[numeric].median()
    identity = (
        selected[["audit_id", "official_test_index", "true_class_id"]]
        .drop_duplicates("audit_id")
    )
    return identity.merge(medians, on="audit_id", how="inner", validate="one_to_one")


def make_summary_figure(data: dict[str, Any], out_dir: Path, dpi: int) -> list[Path]:
    controls = data["control_audits"]
    cnn = data["cnn_audits"]
    endpoint_stability = data["endpoint_stability"]
    majority = data["majority_surface"]

    add_ep = endpoint_median_frame(controls, "additive")
    quad_ep = endpoint_median_frame(controls, "quadratic")
    cnn_ep = endpoint_median_frame(cnn, "softplus_cnn")

    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.82))
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.86, wspace=0.48)

    # (a) Numerically forced controls versus fitted CNN, using endpoint medians.
    ax = axes[0]
    log_values = [
        np.log10(add_ep["H_over_s"].to_numpy()),
        np.log10(quad_ep["H_over_s"].to_numpy()),
        np.log10(cnn_ep["H_over_s"].to_numpy()),
    ]
    boxes = ax.boxplot(
        log_values,
        tick_labels=["Add.", "Quad.", "CNN"],
        showfliers=False,
        widths=0.56,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.0},
    )
    for patch, color in zip(
        boxes["boxes"],
        [COLORS["additive"], COLORS["quadratic"], COLORS["cnn"]],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.axhline(-7, linestyle="--", color="#444444", linewidth=0.75)
    ax.text(1.02, -6.72, "null tolerance", fontsize=5.7, color="#444444")
    ax.set_ylim(-14.5, 1.55)
    ax.set_yticks([-14, -10, -7, -3, 1])
    ax.set_yticklabels([r"$10^{-14}$", r"$10^{-10}$", r"$10^{-7}$", r"$10^{-3}$", r"$10^{1}$"])
    ax.set_ylabel(r"endpoint-median $H/s$")
    ax.set_title("(a) Null controls vs CNN", pad=3)
    ax.grid(True, axis="y")

    # (b) Continuous regime at the correct endpoint unit.
    ax = axes[1]
    ax.scatter(
        cnn_ep["D_over_s"],
        cnn_ep["H_over_s"],
        s=11,
        alpha=0.55,
        color=COLORS["hidden"],
        linewidths=0,
        label="endpoint median",
    )
    robust_ids = endpoint_stability.loc[
        endpoint_stability["strict_fc_seed_count"] >= MAJORITY_SEEDS,
        "audit_id",
    ].astype(int)
    robust = cnn_ep[cnn_ep["audit_id"].isin(robust_ids)]
    ax.scatter(
        robust["D_over_s"],
        robust["H_over_s"],
        marker="D",
        s=27,
        facecolors="none",
        edgecolors=COLORS["robust"],
        linewidths=1.0,
        label=r"strict in $\geq3/5$",
        zorder=4,
    )
    ax.axvline(FC_KAPPA, linestyle="--", color="#444444", linewidth=0.75)
    ax.axhline(FC_TAU, linestyle="--", color="#444444", linewidth=0.75)
    ax.set_yscale("log")
    ax.set_xlim(0.0, max(0.25, float(cnn_ep["D_over_s"].max()) * 1.06))
    ax.set_ylim(0.025, 23)
    ax.set_xlabel(r"endpoint-median $D/s$")
    ax.set_ylabel(r"endpoint-median $H/s$")
    ax.set_title("(b) Visible vs hidden", pad=3)
    ax.grid(True, which="major")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.25, borderpad=0.1)

    # (c) Entire frozen threshold grid, majority recurrence.
    ax = axes[2]
    cmap = LinearSegmentedColormap.from_list(
        "e3b_surface", ["#F6F3FA", "#A987BE", "#5E3C86"]
    )
    image = ax.imshow(majority, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(KAPPAS)))
    ax.set_xticklabels([f"{value:g}" for value in KAPPAS])
    ax.set_yticks(np.arange(len(TAUS)))
    ax.set_yticklabels([f"{value:g}" for value in TAUS])
    for row in range(len(TAUS)):
        for col in range(len(KAPPAS)):
            value = int(majority[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if value >= 35 else "#222222",
            )
    original_row = TAUS.index(FC_TAU)
    original_col = KAPPAS.index(FC_KAPPA)
    ax.add_patch(
        Rectangle(
            (original_col - 0.46, original_row - 0.46),
            0.92,
            0.92,
            fill=False,
            edgecolor="#111111",
            linewidth=1.05,
        )
    )
    ax.set_xlabel(r"agreement tolerance $\kappa$")
    ax.set_ylabel(r"hidden threshold $\tau$")
    ax.set_title(r"(c) Endpoints in $\geq3/5$ fits", pad=3)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.8)

    # (d) Interaction-order anatomy at endpoint level.
    ax = axes[3]
    orders = np.arange(2, 9)
    order_columns = [f"R_order{order}_over_s" for order in orders]
    order_endpoint = cnn.groupby("audit_id")[order_columns].median()
    order_medians = np.asarray([order_endpoint[column].median() for column in order_columns])
    order_q90 = np.asarray([order_endpoint[column].quantile(0.90) for column in order_columns])
    ax.bar(
        orders,
        order_medians,
        width=0.72,
        color=COLORS["orders"],
        alpha=0.82,
        label="median",
    )
    ax.plot(
        orders,
        order_q90,
        color="#D55E00",
        marker="o",
        markersize=2.7,
        linewidth=0.95,
        label="90th pct.",
    )
    ax.set_xticks(orders)
    ax.set_xlabel("interaction order")
    ax.set_ylabel(r"endpoint-median $R_k/s$")
    ax.set_title("(d) Redistribution anatomy", pad=3)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)

    return save_figure(fig, out_dir / "e3b_app_result_summary", dpi)


def make_witness_figure(data: dict[str, Any], out_dir: Path, dpi: int) -> list[Path]:
    cnn = data["cnn_audits"]
    stability = data["endpoint_stability"]
    robust = stability[
        stability["strict_fc_seed_count"] >= MAJORITY_SEEDS
    ].sort_values(["strict_fc_seed_count", "audit_id"], ascending=[False, True])
    if len(robust) != 2:
        raise RuntimeError(
            "Expected exactly two majority-stable strict witnesses under the frozen point."
        )

    fig = plt.figure(figsize=(7.05, 3.25))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.82, 2.30, 1.36, 1.48],
        hspace=0.62,
        wspace=0.47,
        left=0.035,
        right=0.995,
        bottom=0.14,
        top=0.91,
    )

    for row_index, witness in enumerate(robust.itertuples(index=False)):
        audit_id = int(witness.audit_id)
        group = cnn[cnn["audit_id"] == audit_id].sort_values("seed")
        strict_group = group[group["false_consensus_primary"]]
        chosen = strict_group.sort_values("seed").iloc[0]
        output_scale = float(chosen["output_scale_s"])

        image_ax = fig.add_subplot(grid[row_index, 0])
        image_ax.imshow(data["audit_images"][audit_id], cmap="gray", vmin=0, vmax=255)
        draw_region_grid(image_ax, labels=True, linewidth=0.55)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.set_title(
            f"audit {audit_id}: {witness.true_class_name}\n"
            f"strict in {int(witness.strict_fc_seed_count)}/5 fits",
            fontsize=6.7,
            pad=2.0,
        )

        vector_ax = fig.add_subplot(grid[row_index, 1])
        regions = np.arange(1, 9)
        bshap = np.asarray(
            [chosen[f"bshap_region{region}"] / output_scale for region in range(8)]
        )
        ig = np.asarray(
            [chosen[f"ig_region{region}"] / output_scale for region in range(8)]
        )
        width = 0.37
        vector_ax.bar(
            regions - width / 2,
            bshap,
            width=width,
            color=COLORS["bshap"],
            alpha=0.88,
            label="BShap",
        )
        vector_ax.bar(
            regions + width / 2,
            ig,
            width=width,
            color=COLORS["ig"],
            alpha=0.88,
            label="IG",
        )
        vector_ax.axhline(0.0, color="#333333", linewidth=0.55)
        vector_ax.set_xticks(regions)
        vector_ax.set_xticklabels([f"R{region}" for region in regions])
        vector_ax.set_ylabel("attribution / s")
        vector_ax.set_title(
            f"Reported vectors (seed ...{str(int(chosen['seed']))[-2:]})",
            fontsize=7.2,
            pad=2.2,
        )
        vector_ax.grid(True, axis="y")
        if row_index == 0:
            vector_ax.legend(frameon=False, ncol=2, loc="upper left")

        order_ax = fig.add_subplot(grid[row_index, 2])
        orders = np.arange(2, 9)
        order_values = np.asarray(
            [chosen[f"R_order{order}_over_s"] for order in orders]
        )
        order_ax.bar(
            orders,
            order_values,
            width=0.72,
            color=COLORS["orders"],
            alpha=0.85,
        )
        order_ax.set_xticks(orders)
        order_ax.set_xlabel("order $k$")
        order_ax.set_ylabel(r"$R_k/s$")
        order_ax.set_title("Redistribution anatomy", fontsize=7.2, pad=2.2)
        order_ax.grid(True, axis="y")
        order_ax.text(
            0.98,
            0.96,
            rf"$D/s={chosen['D_over_s']:.3f}$" + "\n" +
            rf"$H/s={chosen['H_over_s']:.3f}$" + "\n" +
            rf"$\chi={chosen['chi']:.3f}$",
            transform=order_ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.1,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2},
        )

        recurrence_ax = fig.add_subplot(grid[row_index, 3])
        strict_mask = group["false_consensus_primary"].to_numpy(dtype=bool)
        recurrence_ax.scatter(
            group.loc[~strict_mask, "D_over_s"],
            group.loc[~strict_mask, "H_over_s"],
            s=22,
            color="#B8B8B8",
            edgecolors="#555555",
            linewidths=0.4,
            label="other fit",
        )
        recurrence_ax.scatter(
            group.loc[strict_mask, "D_over_s"],
            group.loc[strict_mask, "H_over_s"],
            s=28,
            color=COLORS["strict"],
            edgecolors="white",
            linewidths=0.45,
            label="strict FC",
            zorder=3,
        )
        for point in group.itertuples(index=False):
            recurrence_ax.annotate(
                str(int(point.seed))[-2:],
                (point.D_over_s, point.H_over_s),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5.5,
            )
        recurrence_ax.axvline(FC_KAPPA, linestyle="--", color="#444444", linewidth=0.7)
        recurrence_ax.axhline(FC_TAU, linestyle="--", color="#444444", linewidth=0.7)
        recurrence_ax.set_xlabel(r"$D/s$")
        recurrence_ax.set_ylabel(r"$H/s$")
        recurrence_ax.set_title("Across five fitted CNNs", fontsize=7.2, pad=2.2)
        recurrence_ax.margins(x=0.16, y=0.18)
        recurrence_ax.grid(True, alpha=0.35)
        if row_index == 0:
            recurrence_ax.legend(frameon=False, loc="best", handletextpad=0.3)

    fig.text(
        0.5,
        0.985,
        "Complete set of majority-stable strict false-consensus endpoints",
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="semibold",
    )
    return save_figure(fig, out_dir / "e3b_app_stable_witnesses", dpi)


# =============================================================================
# LaTeX tables and insertion block
# =============================================================================


def interaction_mass_over_s(df: pd.DataFrame) -> pd.Series:
    columns = [f"pot_order{order}_L1_over_s" for order in range(2, 9)]
    return df[columns].sum(axis=1)


def model_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    controls = data["control_audits"].copy()
    cnn = data["cnn_audits"].copy()
    controls["M_over_s"] = interaction_mass_over_s(controls)
    cnn["M_over_s"] = cnn["interaction_pot_mass_M"] / cnn["output_scale_s"]
    fits = {
        "additive": data["control_fits"].query("family == 'additive'"),
        "quadratic": data["control_fits"].query("family == 'quadratic'"),
        "softplus_cnn": data["cnn_fits"],
    }
    audits = {
        "additive": controls.query("family == 'additive'"),
        "quadratic": controls.query("family == 'quadratic'"),
        "softplus_cnn": cnn,
    }
    labels = {
        "additive": "Additive region",
        "quadratic": "Quadratic region",
        "softplus_cnn": "Softplus CNN",
    }
    result: list[dict[str, Any]] = []
    for family in ["additive", "quadratic", "softplus_cnn"]:
        group = audits[family]
        fit_group = fits[family]
        material_recurrence = (
            (group["H_over_s"] >= FC_TAU).groupby(group["audit_id"]).sum()
        )
        strict_recurrence = (
            (
                (group["D_over_s"] <= FC_KAPPA)
                & (group["H_over_s"] >= FC_TAU)
            )
            .groupby(group["audit_id"])
            .sum()
        )
        result.append(
            {
                "family": family,
                "label": labels[family],
                "test_accuracy_min": float(fit_group["test_accuracy"].min()),
                "test_accuracy_max": float(fit_group["test_accuracy"].max()),
                "M_over_s_median": float(group["M_over_s"].median()),
                "D_over_s_median": float(group["D_over_s"].median()),
                "H_over_s_median": float(group["H_over_s"].median()),
                "H_over_s_max": float(group["H_over_s"].max()),
                "chi_median": (
                    float(group["chi"].dropna().median())
                    if family == "softplus_cnn"
                    else None
                ),
                "material_majority_endpoints": int(
                    (material_recurrence >= MAJORITY_SEEDS).sum()
                ),
                "strict_majority_endpoints": int(
                    (strict_recurrence >= MAJORITY_SEEDS).sum()
                ),
            }
        )
    return result


def make_model_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen E3b model results. Test accuracy is the range over five fits. Ledger summaries are descriptive medians over the $500$ seed--endpoint rows; control $H/s$ values are numerical nulls. Material and strict columns count endpoints satisfying $H/s\geq0.05$, respectively together with $D/s\leq0.02$, in at least three of five fits.}",
        r"\label{tab:e3b-model-summary}",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Family & Test acc. & $\operatorname{med} M/s$ & $\operatorname{med} D/s$ & $\operatorname{med} H/s$ & $\max H/s$ & $\operatorname{med}\chi$ & Material & Strict \\",
        r"\midrule",
    ]
    for row in rows:
        chi = "--" if row["chi_median"] is None else f"${row['chi_median']:.3f}$"
        lines.append(
            f"{latex_escape(row['label'])} & "
            f"${row['test_accuracy_min']:.3f}$--${row['test_accuracy_max']:.3f}$ & "
            f"{latex_number(row['M_over_s_median'])} & "
            f"{latex_number(row['D_over_s_median'])} & "
            f"{latex_number(row['H_over_s_median'])} & "
            f"{latex_number(row['H_over_s_max'])} & "
            f"{chi} & "
            f"${row['material_majority_endpoints']}/100$ & "
            f"${row['strict_majority_endpoints']}/100$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def make_class_table(class_summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CNN audit summaries by true Fashion-MNIST class. Each class contributes ten frozen endpoints and $50$ seed--endpoint rows. Maj. strict counts endpoints satisfying the original strict criterion in at least three of five fits.}",
        r"\label{tab:e3b-classwise}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.3pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Class & $\operatorname{med}D/s$ & $\operatorname{med}H/s$ & $q_{.9}(H/s)$ & $\operatorname{med}\chi$ & Maj. strict \\",
        r"\midrule",
    ]
    for row in class_summary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.true_class_name)} & "
            f"${row.median_D_over_s:.3f}$ & "
            f"${row.median_H_over_s:.3f}$ & "
            f"${row.q90_H_over_s:.3f}$ & "
            f"${row.median_chi:.3f}$ & "
            f"${int(row.strict_fc_endpoints_in_at_least_3_of_5)}/10$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def certification_ratio(group: pd.DataFrame) -> float:
    nested_ratio = group["nested_quadrature_error"] / group["numerical_tolerance"]
    certificate_ratio = (
        group[CERTIFICATE_COLUMNS].max(axis=1)
        / (10.0 * group["numerical_tolerance"])
    )
    return float(np.maximum(nested_ratio, certificate_ratio).max())


def make_numerics_table(data: dict[str, Any]) -> str:
    controls = data["control_audits"]
    cnn = data["cnn_audits"]
    groups = [
        ("Additive region", controls.query("family == 'additive'")),
        ("Quadratic region", controls.query("family == 'quadratic'")),
        ("Softplus CNN", cnn),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Numerical certification for all E3b audits. The final column is the largest error divided by its applicable frozen tolerance, so values below one certify every audit.}",
        r"\label{tab:e3b-numerics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrlc}",
        r"\toprule",
        r"Family & Resolved & Certified & Quadrature orders & Max tol. ratio \\",
        r"\midrule",
    ]
    for label, group in groups:
        order_counts = group["quadrature_order"].value_counts().sort_index()
        order_text = ", ".join(
            f"{int(order)}:{int(count)}" for order, count in order_counts.items()
        )
        lines.append(
            f"{latex_escape(label)} & ${int(group['resolved'].sum())}/500$ & "
            f"${int(group['certification_pass'].sum())}/500$ & "
            f"{order_text} & {latex_number(certification_ratio(group))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def make_surface_table(majority: np.ndarray) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Complete frozen E3b threshold surface. Each entry counts endpoints for which $D/s\leq\kappa$ and $H/s\geq\tau$ in at least three of five CNN fits. The preregistered operating point is $(\kappa,\tau)=(0.02,0.05)$.}",
        r"\label{tab:e3b-threshold-surface}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.2pt}",
        r"\begin{tabular}{c|rrrrr}",
        r"\toprule",
        r"$\tau\backslash\kappa$ & $0.005$ & $0.01$ & $0.02$ & $0.05$ & $0.10$ \\",
        r"\midrule",
    ]
    for row_index, tau in enumerate(TAUS):
        values = []
        for col_index in range(len(KAPPAS)):
            value = int(majority[row_index, col_index])
            if tau == FC_TAU and KAPPAS[col_index] == FC_KAPPA:
                values.append(rf"\textbf{{{value}}}")
            else:
                values.append(str(value))
        lines.append(f"${tau:g}$ & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_appendix_insert(data: dict[str, Any]) -> str:
    summary = data["cnn_summary"]["overall"]
    class_summary = data["class_summary"]
    order = data["cnn_summary"]["redistribution_by_interaction_order"]
    return rf"""
\section{{Appendix-only vision replication}}
\label{{app:e3b-vision}}

\paragraph{{Frozen contract.}}
To test whether the fitted-model conclusion extends beyond tabular regression,
we froze an appendix-only Fashion-MNIST replication before fitting any vision
model.  The canonical training set is split into $51{{,}}000$ training and
$9{{,}}000$ validation images with seed $20260910$; the official $10{{,}}000$
test images remain held out.  Pixel normalization is fitted on training data
only.  The explanation representation consists of eight fixed $14\times7$
regions in a $2\times4$ grid, so every endpoint game is again evaluated at all
$2^8=256$ Boolean corners.  The baseline is official training image $37961$,
the observed training image nearest the train mean.  Before model fitting we
froze $100$ official-test endpoints, ten per class, and five model seeds
$20260920$--$20260924$.  For endpoint label $y$, the scalar output is the
true-class centered logit
\[
g_y(x)=\operatorname{{logit}}_y(x)
-\frac{{1}}{{9}}\sum_{{c\ne y}}\operatorname{{logit}}_c(x).
\]
The path changes the eight regions linearly from the observed baseline to the
endpoint.  The scale $s$ is the training-only $95$th--$5$th percentile range
of $g_y$ evaluated using each training image's true label.

\begin{{figure*}}[t]
    \centering
    \includegraphics[width=\textwidth]{{paper_figures/e3b_app_protocol.pdf}}
    \caption{{Frozen E3b visual contract. Left: the observed training baseline
    and the eight explanation regions. Right: the smallest frozen audit ID in
    each class, displayed by a deterministic rule; the complete panel contains
    ten endpoints per class.}}
    \label{{fig:e3b-protocol}}
\end{{figure*}}

\paragraph{{Fatal controls and fitted CNN.}}
We first fit five additive-region Softplus classifiers and five quadratic
region-interaction classifiers.  The quadratic family contains genuine learned
pair interactions: its median pair-pot $\ell_1$ mass is $0.657s$.  Nevertheless,
the theory requires both controls to have zero transfer ledger, and the
exhaustive engine returns maximum $H/s$ values of
$6.38\times10^{{-14}}$ and $5.01\times10^{{-14}}$, respectively.  All
$1{{,}}000$ control audits resolve and certify.

The confirmatory vision model is the prespecified $421{{,}}642$-parameter
Softplus CNN.  Across five fits, official-test accuracy ranges from $0.892$ to
$0.902$, so every frozen predictive gate passes.  All $500$ fitted-model audits
also resolve and certify.  Across the $500$ seed--endpoint rows,
\[
\begin{{aligned}}
\operatorname{{median}}(D/s)&={summary['median_D_over_s']:.3f}, &
\operatorname{{median}}(H/s)&={summary['median_H_over_s']:.3f},\\
\operatorname{{median}}(\chi)&={summary['median_chi']:.3f}. &&
\end{{aligned}}
\]
The $90$th percentile of $H/s$ is {summary['q90_H_over_s']:.3f}, its maximum
is {summary['max_H_over_s']:.3f}, and $99\%$ of audit rows satisfy
$H/s\geq0.05$.  Thus the CNN median exceeds the worst control by roughly
fourteen orders of magnitude.

\input{{paper_figures/tab_e3b_model_summary.tex}}

\begin{{figure*}}[t]
    \centering
    \includegraphics[width=\textwidth]{{paper_figures/e3b_app_result_summary.pdf}}
    \caption{{E3b confirmatory summary. \textbf{{(a)}} Each distribution uses
    one median over five fitted models per endpoint; both fatal controls remain
    at numerical zero while the CNN is separated by roughly fourteen orders of
    magnitude. \textbf{{(b)}} Seed-median visible discrepancy and hidden mass
    for all $100$ endpoints; diamonds mark the two endpoints satisfying the
    original strict criterion in at least three fits. \textbf{{(c)}} Complete
    prespecified threshold surface, reported at the paired endpoint unit.
    \textbf{{(d)}} Redistribution by interaction order after taking a median
    across seeds within each endpoint.}}
    \label{{fig:e3b-summary}}
\end{{figure*}}

\paragraph{{Continuous and strict conclusions.}}
The hidden signal is nearly ubiquitous: $99/100$ endpoints have
$H/s\geq0.05$ in at least three fits, and for $99$ of them it holds in all five.
The strict preregistered conjunction $D/s\leq0.02$, $H/s\geq0.05$ holds in
$13/500$ seed--endpoint rows, for $8/100$ endpoints at least once, and for
$2/100$ endpoints in a majority of fits.  We therefore do not describe strict
false consensus as common.  The entire fixed grid nevertheless shows a stable
region: at $D/s\leq0.05$, $H/s\geq0.20$, $10/100$ endpoints qualify in a
majority of fits; at $D/s\leq0.10$ the count is $43/100$.

The conventional aggregate screens are simultaneously reassuring and
incomplete.  Their medians are signed cosine
${summary['median_signed_cosine']:.3f}$, absolute-rank Spearman
${summary['median_spearman_abs']:.3f}$, and top-three Jaccard
${summary['median_top3_jaccard']:.3f}$, while the median hidden share of total
interaction-pot mass is $H/M={summary['median_H_over_M']:.3f}$.  The ledger
anatomy is not a pairwise artifact: median redistribution rises from
$R_2/s={order['2']['median']:.3f}$ to $R_5/s={order['5']['median']:.3f}$ before
declining at orders six through eight.

\begin{{figure*}}[t]
    \centering
    \includegraphics[width=\textwidth]{{paper_figures/e3b_app_stable_witnesses.pdf}}
    \caption{{The complete set of majority-stable strict E3b witnesses.  For
    each endpoint, the displayed fitted model is the smallest frozen seed that
    satisfies the preregistered strict criterion.  The reported BShap and IG
    region vectors remain close, while the order-resolved ledger reveals
    substantial redistribution that cancels under aggregation.  The rightmost
    panels show recurrence across all five independently fitted CNNs.}}
    \label{{fig:e3b-witnesses}}
\end{{figure*}}

\input{{paper_figures/tab_e3b_classwise.tex}}
\input{{paper_figures/tab_e3b_numerics.tex}}
\input{{paper_figures/tab_e3b_threshold_surface.tex}}

\paragraph{{Scope.}}
This replication is conditional on the frozen Fashion-MNIST task, observed
baseline, coarse eight-region representation, centered-logit scalar, and
straight region-slider path.  It is not evidence about causal image regions or
all vision architectures.  Its role is narrower: under a second modality and a
convolutional model, the same exhaustive ledger engine remains null in the two
theoretically required controls and detects large concealed redistribution in
a competitively fitted nonlinear predictor.
"""


# =============================================================================
# Main
# =============================================================================


def run(args: argparse.Namespace) -> None:
    configure_style()
    protocol_dir = Path(args.protocol_dir)
    controls_dir = Path(args.controls_dir)
    cnn_dir = Path(args.cnn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_validate(protocol_dir, controls_dir, cnn_dir)
    outputs: list[Path] = []
    outputs.extend(make_protocol_figure(data, out_dir, args.dpi))
    outputs.extend(make_summary_figure(data, out_dir, args.dpi))
    outputs.extend(make_witness_figure(data, out_dir, args.dpi))

    rows = model_rows(data)
    text_outputs = {
        "tab_e3b_model_summary.tex": make_model_table(rows),
        "tab_e3b_classwise.tex": make_class_table(data["class_summary"]),
        "tab_e3b_numerics.tex": make_numerics_table(data),
        "tab_e3b_threshold_surface.tex": make_surface_table(
            data["majority_surface"]
        ),
        "e3b_appendix_insert.tex": make_appendix_insert(data),
    }
    for name, content in text_outputs.items():
        path = out_dir / name
        write_text(content, path)
        outputs.append(path)

    cnn = data["cnn_audits"]
    endpoint_medians = cnn.groupby("audit_id")[
        ["D_over_s", "H_over_s", "chi", "H_over_M"]
    ].median()
    material_recurrence = (
        (cnn["H_over_s"] >= FC_TAU).groupby(cnn["audit_id"]).sum()
    )
    strict_recurrence = (
        cnn["false_consensus_primary"].groupby(cnn["audit_id"]).sum()
    )
    paper_values = {
        "experiment": "E3b appendix-only Fashion-MNIST replication",
        "verification": data["verification"],
        "model_table_rows": rows,
        "pooled_seed_endpoint_summary": data["cnn_summary"]["overall"],
        "endpoint_median_summary": {
            "median_D_over_s": float(endpoint_medians["D_over_s"].median()),
            "median_H_over_s": float(endpoint_medians["H_over_s"].median()),
            "q90_H_over_s": float(endpoint_medians["H_over_s"].quantile(0.90)),
            "median_chi": float(endpoint_medians["chi"].median()),
            "median_H_over_M": float(endpoint_medians["H_over_M"].median()),
        },
        "paired_endpoint_counts": {
            "material_H_at_least_3_of_5": int(
                (material_recurrence >= MAJORITY_SEEDS).sum()
            ),
            "material_H_all_5_of_5": int((material_recurrence == 5).sum()),
            "strict_at_least_1_of_5": int((strict_recurrence >= 1).sum()),
            "strict_at_least_3_of_5": int(
                (strict_recurrence >= MAJORITY_SEEDS).sum()
            ),
            "strict_all_5_of_5": int((strict_recurrence == 5).sum()),
        },
        "majority_threshold_surface_rows_tau_cols_kappa": (
            data["majority_surface"].tolist()
        ),
        "strict_witness_audit_ids": data["endpoint_stability"].loc[
            data["endpoint_stability"]["strict_fc_seed_count"] >= MAJORITY_SEEDS,
            "audit_id",
        ].astype(int).tolist(),
        "classwise": data["class_summary"].to_dict(orient="records"),
    }
    values_path = out_dir / "e3b_paper_values.json"
    write_json(paper_values, values_path)
    outputs.append(values_path)

    manifest_path = out_dir / "e3b_artifact_manifest_sha256.json"
    manifest = {
        "hash_algorithm": "SHA-256",
        "read_only_generation": True,
        "input_manifests": {
            stage: value["manifest_sha256"]
            for stage, value in data["verification"].items()
        },
        "outputs": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(outputs, key=lambda item: item.name)
        },
    }
    write_json(manifest, manifest_path)

    print()
    print("=" * 88)
    print("E3b — FROZEN PAPER ARTIFACTS")
    print("=" * 88)
    print("Verified protocol/control/CNN manifests : True")
    print("Model training or attribution calls     : 0")
    print("Frozen CNN audit rows                   : 500")
    print("Frozen unique endpoints                 : 100")
    print("Resolved / certified                    : 500 / 500")
    print(
        "Pooled median D/s, H/s, chi           : "
        f"{cnn['D_over_s'].median():.4f}, "
        f"{cnn['H_over_s'].median():.4f}, "
        f"{cnn['chi'].median():.4f}"
    )
    print(
        "Material H in >=3/5 fits              : "
        f"{int((material_recurrence >= MAJORITY_SEEDS).sum())}/100"
    )
    print(
        "Strict FC in >=3/5 fits               : "
        f"{int((strict_recurrence >= MAJORITY_SEEDS).sum())}/100"
    )
    print(f"Output directory                        : {out_dir.resolve()}")
    print("Generated:")
    for path in sorted(outputs + [manifest_path], key=lambda item: item.name):
        print(f"  {path.name}")
    print("=" * 88)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all E3b appendix artifacts without retraining."
    )
    parser.add_argument("--protocol-dir", default="./e3_vision_protocol")
    parser.add_argument("--controls-dir", default="./e3_vision_controls")
    parser.add_argument("--cnn-dir", default="./e3_vision_cnn")
    parser.add_argument("--out-dir", default="./ICLR/paper_figures")
    parser.add_argument("--dpi", type=int, default=400)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
