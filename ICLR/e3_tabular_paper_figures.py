#!/usr/bin/env python3
"""
E3a — Paper figure generator for the frozen tabular experiment
================================================================

This script reads the already-completed E3a outputs and creates the complete
main-paper and appendix figure set. It never trains a model, changes an audit
endpoint, searches over a threshold, or recomputes an attribution.

Run from the repository root
----------------------------
    python ICLR/e3_tabular_paper_figures.py

Default inputs
--------------
    ./e3_tabular_controls/control_audits.csv
    ./e3_tabular_controls/fit_metrics.csv
    ./e3_tabular_mlp/mlp_audits.csv
    ./e3_tabular_mlp/fit_metrics.csv
    ./e3_tabular_crossnet/crossnet_audits.csv
    ./e3_tabular_crossnet/fit_metrics.csv

Default outputs
---------------
    ./ICLR/paper_figures/e3_main_fitted_audit.{pdf,png}
    ./ICLR/paper_figures/e3_app_threshold_surfaces.{pdf,png}
    ./ICLR/paper_figures/e3_app_predictive_numerical.{pdf,png}
    ./ICLR/paper_figures/e3_app_cross_architecture.{pdf,png}
    ./ICLR/paper_figures/e3_figure_values.json
    ./ICLR/paper_figures/e3_figure_manifest_sha256.json

Scientific unit
---------------
The 500 rows for a fitted architecture are 100 fixed endpoints evaluated under
five independently trained fits. Main-paper distributional and association
panels therefore use one seed-median value per endpoint. Seed-level recurrence
is retained only where recurrence itself is the estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Frozen descriptive contract. These are reported as a complete grid; they are
# not selected by this script.
# -----------------------------------------------------------------------------
EXPECTED_SEEDS = [20260840, 20260841, 20260842, 20260843, 20260844]
EXPECTED_ENDPOINTS = 100

FC_KAPPA = 0.02
FC_TAU = 0.05
KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.10]
TAUS = [0.01, 0.02, 0.05, 0.10, 0.20]
MAJORITY_SEEDS = 3

MODEL_ORDER = ["Additive", "Quadratic", "Softplus MLP", "CrossNet"]
FAMILY_TO_MODEL = {
    "additive": "Additive",
    "quadratic": "Quadratic",
    "softplus_mlp": "Softplus MLP",
    "smooth_crossnet": "CrossNet",
}
MODEL_COLORS = {
    "Additive": "#7F8C8D",
    "Quadratic": "#4C78A8",
    "Softplus MLP": "#D66B4D",
    "CrossNet": "#2A9D8F",
}

CERTIFICATE_ERROR_COLUMNS = [
    "pot_conservation_error",
    "bshap_reconstruction_error",
    "ig_reconstruction_error",
    "bshap_completeness_error",
    "ig_completeness_error",
    "margin_gap_error",
    "interior_mobius_reconstruction_error",
]


def configure_style() -> None:
    """Paper-oriented, colorblind-safe Matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path.resolve()}")
    return pd.read_csv(path)


def parse_bool_column(df: pd.DataFrame, column: str, label: str) -> None:
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
        bad = sorted(df.loc[parsed.isna(), column].astype(str).unique().tolist())
        raise RuntimeError(f"{label}: cannot parse {column}; values={bad}")
    df[column] = parsed.astype(bool)


def validate_audit_family(df: pd.DataFrame, model: str) -> None:
    required = {
        "seed", "audit_id", "dataset_index", "target", "D_over_s",
        "H_over_s", "chi", "resolved", "certification_pass",
        "quadrature_order", "nested_quadrature_error",
        *CERTIFICATE_ERROR_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{model}: missing audit columns {sorted(missing)}")

    for column in ["resolved", "certification_pass"]:
        parse_bool_column(df, column, model)

    seeds = sorted(df["seed"].astype(int).unique().tolist())
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"{model}: unexpected seeds {seeds}")
    if len(df) != EXPECTED_ENDPOINTS * len(EXPECTED_SEEDS):
        raise RuntimeError(f"{model}: expected 500 audit rows, found {len(df)}")
    if df["audit_id"].nunique() != EXPECTED_ENDPOINTS:
        raise RuntimeError(
            f"{model}: expected {EXPECTED_ENDPOINTS} endpoints, "
            f"found {df['audit_id'].nunique()}"
        )
    if df.duplicated(["seed", "audit_id"]).any():
        raise RuntimeError(f"{model}: duplicate (seed, audit_id) rows")
    if not (df.groupby("audit_id").size() == len(EXPECTED_SEEDS)).all():
        raise RuntimeError(f"{model}: some endpoints do not have five fits")
    if not (df.groupby("audit_id")["dataset_index"].nunique() == 1).all():
        raise RuntimeError(f"{model}: dataset_index varies within audit_id")
    if not (df.groupby("audit_id")["target"].nunique() == 1).all():
        raise RuntimeError(f"{model}: target varies within audit_id")
    if not df["resolved"].all():
        raise RuntimeError(f"{model}: at least one audit is unresolved")
    if not df["certification_pass"].all():
        raise RuntimeError(f"{model}: at least one audit is uncertified")

    for column in ["D_over_s", "H_over_s", "quadrature_order"]:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{model}: non-finite values in {column}")
    if (df["D_over_s"] < -1e-12).any() or (df["H_over_s"] < -1e-10).any():
        raise RuntimeError(f"{model}: materially negative D/s or H/s")


def validate_fit_family(df: pd.DataFrame, model: str) -> None:
    required = {"seed", "test_r2", "output_scale_s"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{model}: missing fit columns {sorted(missing)}")
    seeds = sorted(df["seed"].astype(int).unique().tolist())
    if seeds != EXPECTED_SEEDS or len(df) != len(EXPECTED_SEEDS):
        raise RuntimeError(f"{model}: expected exactly five fit rows")
    if not np.isfinite(df[["test_r2", "output_scale_s"]].to_numpy(float)).all():
        raise RuntimeError(f"{model}: non-finite fit metric")
    if (df["output_scale_s"] <= 0).any():
        raise RuntimeError(f"{model}: non-positive output scale")
    if "predictive_gate" in df.columns:
        parse_bool_column(df, "predictive_gate", model)
        if not df["predictive_gate"].all():
            raise RuntimeError(f"{model}: a predictive gate failed")


def assert_same_endpoints(audits: pd.DataFrame) -> None:
    maps = {}
    for model, group in audits.groupby("model", sort=False):
        maps[model] = (
            group[["audit_id", "dataset_index", "target"]]
            .drop_duplicates()
            .sort_values("audit_id")
            .reset_index(drop=True)
        )
    reference = maps[MODEL_ORDER[0]]
    for model in MODEL_ORDER[1:]:
        other = maps[model]
        if not np.array_equal(reference["audit_id"], other["audit_id"]):
            raise RuntimeError(f"{model}: audit_id panel differs from controls")
        if not np.array_equal(reference["dataset_index"], other["dataset_index"]):
            raise RuntimeError(f"{model}: dataset indices differ from controls")
        if not np.allclose(reference["target"], other["target"], rtol=0, atol=1e-12):
            raise RuntimeError(f"{model}: targets differ from controls")


def load_all(args: argparse.Namespace):
    controls_dir = Path(args.controls_dir)
    mlp_dir = Path(args.mlp_dir)
    crossnet_dir = Path(args.crossnet_dir)

    paths = {
        "control_audits": controls_dir / "control_audits.csv",
        "control_fits": controls_dir / "fit_metrics.csv",
        "mlp_audits": mlp_dir / "mlp_audits.csv",
        "mlp_fits": mlp_dir / "fit_metrics.csv",
        "crossnet_audits": crossnet_dir / "crossnet_audits.csv",
        "crossnet_fits": crossnet_dir / "fit_metrics.csv",
    }

    control_audits = read_csv(paths["control_audits"], "control audits")
    control_fits = read_csv(paths["control_fits"], "control fit metrics")
    mlp_audits = read_csv(paths["mlp_audits"], "MLP audits")
    mlp_fits = read_csv(paths["mlp_fits"], "MLP fit metrics")
    cross_audits = read_csv(paths["crossnet_audits"], "CrossNet audits")
    cross_fits = read_csv(paths["crossnet_fits"], "CrossNet fit metrics")

    audit_parts = []
    fit_parts = []
    for family, model in FAMILY_TO_MODEL.items():
        if family in {"additive", "quadratic"}:
            a = control_audits.loc[control_audits["family"] == family].copy()
            f = control_fits.loc[control_fits["family"] == family].copy()
        elif family == "softplus_mlp":
            a, f = mlp_audits.copy(), mlp_fits.copy()
        else:
            a, f = cross_audits.copy(), cross_fits.copy()

        if a.empty or f.empty:
            raise RuntimeError(f"No rows found for expected family {family}")
        a["model"] = model
        f["model"] = model
        validate_audit_family(a, model)
        validate_fit_family(f, model)
        audit_parts.append(a)
        fit_parts.append(f)

    audits = pd.concat(audit_parts, ignore_index=True)
    fits = pd.concat(fit_parts, ignore_index=True)
    assert_same_endpoints(audits)

    # Add output scale to every audit row. Controls do not store it in their
    # audit CSV, so the fit table is the authoritative common source.
    scale_lookup = fits[["model", "seed", "output_scale_s"]].copy()
    if "output_scale_s" in audits.columns:
        audits = audits.drop(columns=["output_scale_s"])
    audits = audits.merge(
        scale_lookup,
        on=["model", "seed"],
        how="left",
        validate="many_to_one",
    )
    if audits["output_scale_s"].isna().any():
        raise RuntimeError("Failed to attach an output scale to every audit")

    return audits, fits, paths


def endpoint_summary(audits: pd.DataFrame) -> pd.DataFrame:
    d = audits.copy()
    d["visible"] = d["D_over_s"] <= FC_KAPPA
    d["material"] = d["H_over_s"] >= FC_TAU
    d["strict_fc"] = d["visible"] & d["material"]

    rows = []
    for (model, audit_id), g in d.groupby(["model", "audit_id"], sort=True):
        rows.append({
            "model": model,
            "audit_id": int(audit_id),
            "dataset_index": int(g["dataset_index"].iloc[0]),
            "target": float(g["target"].iloc[0]),
            "median_D_over_s": float(g["D_over_s"].median()),
            "median_H_over_s": float(g["H_over_s"].median()),
            "median_chi": (
                float(g["chi"].dropna().median())
                if not g["chi"].dropna().empty else np.nan
            ),
            "visible_seed_count": int(g["visible"].sum()),
            "material_seed_count": int(g["material"].sum()),
            "strict_fc_seed_count": int(g["strict_fc"].sum()),
        })
    result = pd.DataFrame(rows)
    expected = len(MODEL_ORDER) * EXPECTED_ENDPOINTS
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} endpoint summaries, got {len(result)}")
    return result


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = pd.Series(list(x), dtype=float)
    y = pd.Series(list(y), dtype=float)
    keep = x.notna() & y.notna()
    return pearson(
        x.loc[keep].rank(method="average"),
        y.loc[keep].rank(method="average"),
    )


def paired_fitted_endpoints(endpoints: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "audit_id", "dataset_index", "target", "median_D_over_s",
        "median_H_over_s", "median_chi", "visible_seed_count",
        "material_seed_count", "strict_fc_seed_count",
    ]
    mlp = endpoints.loc[endpoints["model"] == "Softplus MLP", columns].copy()
    cross = endpoints.loc[endpoints["model"] == "CrossNet", columns].copy()
    comp = mlp.merge(
        cross,
        on=["audit_id", "dataset_index", "target"],
        how="inner",
        validate="one_to_one",
        suffixes=("_mlp", "_crossnet"),
    )
    if len(comp) != EXPECTED_ENDPOINTS:
        raise RuntimeError("Could not pair the same 100 fitted-model endpoints")
    return comp


def threshold_surface(audits: pd.DataFrame, model: str) -> np.ndarray:
    group = audits.loc[audits["model"] == model]
    matrix = np.zeros((len(TAUS), len(KAPPAS)), dtype=int)
    for ti, tau in enumerate(TAUS):
        for ki, kappa in enumerate(KAPPAS):
            event = (
                (group["D_over_s"] <= kappa)
                & (group["H_over_s"] >= tau)
            )
            recurrence = event.groupby(group["audit_id"]).sum()
            matrix[ti, ki] = int((recurrence >= MAJORITY_SEEDS).sum())
    return matrix


def positive_floor(values: np.ndarray, reference: float | None = None) -> float:
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        floor = 1e-16
    else:
        floor = float(10 ** np.floor(np.log10(positive.min()) - 0.6))
    if reference is not None:
        floor = min(floor, reference / 5.0)
    return max(floor, 1e-18)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "e3_tabular_paper_figures.py", "CreationDate": None},
    )
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "e3_tabular_paper_figures.py"},
    )
    plt.close(fig)
    return [pdf_path, png_path]


def panel_main(
    audits: pd.DataFrame,
    endpoints: pd.DataFrame,
    comp: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    rng = np.random.default_rng(20260860)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 2.90))

    # (a) Null controls versus flexible fitted models.
    ax = axes[0]
    values_by_model = [
        endpoints.loc[endpoints["model"] == model, "median_H_over_s"].to_numpy(float)
        for model in MODEL_ORDER
    ]
    all_h = np.concatenate(values_by_model)
    floor = positive_floor(all_h, reference=FC_TAU)
    plotted = [np.maximum(v, floor) for v in values_by_model]
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    bp = ax.boxplot(
        plotted,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        tick_labels=["Additive", "Quadratic", "MLP", "CrossNet"],
        medianprops={"color": "#111111", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.62)
        patch.set_edgecolor("#333333")
    for position, model, values in zip(positions, MODEL_ORDER, plotted):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=7,
            alpha=0.25,
            color=MODEL_COLORS[model],
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    ax.axhline(FC_TAU, color="#222222", linestyle="--", linewidth=0.9)
    ax.text(
        4.42,
        FC_TAU,
        r"$\tau=.05$",
        fontsize=6.8,
        ha="right",
        va="bottom",
    )
    control_max = {
        model: float(audits.loc[audits["model"] == model, "H_over_s"].max())
        for model in ["Additive", "Quadratic"]
    }
    ax.text(
        0.02,
        0.98,
        "Control audit maxima\n"
        + rf"Additive: ${control_max['Additive']:.1e}$" + "\n"
        + rf"Quadratic: ${control_max['Quadratic']:.1e}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": .88,
              "edgecolor": "#CCCCCC", "linewidth": .5},
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"endpoint-median hidden mass $H/s$")
    ax.set_title("(a) Structural nulls and fitted models", pad=7)
    ax.grid(True, which="major", axis="y")
    ax.tick_params(axis="x", rotation=18)

    # (b) Endpoint-median D-H plane for the two fitted architectures.
    ax = axes[1]
    fitted = endpoints.loc[endpoints["model"].isin(["Softplus MLP", "CrossNet"])]
    x_all = fitted["median_D_over_s"].to_numpy(float)
    y_all = fitted["median_H_over_s"].to_numpy(float)
    x_floor = positive_floor(x_all, reference=FC_KAPPA)
    y_floor = positive_floor(y_all, reference=FC_TAU)
    x_plot = np.maximum(x_all, x_floor)
    y_plot = np.maximum(y_all, y_floor)
    x_min = min(x_plot.min() * .72, FC_KAPPA / 4)
    x_max = max(x_plot.max() * 1.25, FC_KAPPA * 4)
    y_min = min(y_plot.min() * .72, FC_TAU / 4)
    y_max = max(y_plot.max() * 1.30, FC_TAU * 5)
    ax.add_patch(Rectangle(
        (x_min, FC_TAU),
        FC_KAPPA - x_min,
        y_max - FC_TAU,
        facecolor="#E4F1EA",
        edgecolor="none",
        alpha=.8,
        zorder=0,
    ))
    for model in ["Softplus MLP", "CrossNet"]:
        g = fitted.loc[fitted["model"] == model]
        ax.scatter(
            np.maximum(g["median_D_over_s"], x_floor),
            np.maximum(g["median_H_over_s"], y_floor),
            s=16,
            alpha=.62,
            color=MODEL_COLORS[model],
            edgecolors="white",
            linewidths=.25,
            label=model,
            rasterized=True,
        )
    ax.axvline(FC_KAPPA, color="#222222", linestyle="--", linewidth=.85)
    ax.axhline(FC_TAU, color="#222222", linestyle="--", linewidth=.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"endpoint-median visible discrepancy $D/s$")
    ax.set_ylabel(r"endpoint-median hidden mass $H/s$")
    ax.set_title("(b) Fitted-model audit plane", pad=7)
    ax.grid(True, which="major")
    ax.legend(loc="lower right", frameon=False)
    robust_counts = {}
    for model in ["Softplus MLP", "CrossNet"]:
        robust_counts[model] = int((
            endpoints.loc[endpoints["model"] == model, "strict_fc_seed_count"]
            >= MAJORITY_SEEDS
        ).sum())
    ax.text(
        .03,
        .97,
        r"Strict event in $\geq3/5$ fits" + "\n"
        f"MLP: {robust_counts['Softplus MLP']}/100; "
        f"CrossNet: {robust_counts['CrossNet']}/100",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": .88,
              "edgecolor": "#CCCCCC", "linewidth": .5},
    )

    # (c) Cross-architecture replication of endpoint hidden mass.
    ax = axes[2]
    x = comp["median_H_over_s_mlp"].to_numpy(float)
    y = comp["median_H_over_s_crossnet"].to_numpy(float)
    floor_xy = positive_floor(np.concatenate([x, y]), reference=FC_TAU)
    x_plot = np.maximum(x, floor_xy)
    y_plot = np.maximum(y, floor_xy)
    low = min(x_plot.min(), y_plot.min()) * .72
    high = max(x_plot.max(), y_plot.max()) * 1.28

    mlp_top20 = set(comp.nlargest(20, "median_H_over_s_mlp")["audit_id"])
    cross_top20 = set(comp.nlargest(20, "median_H_over_s_crossnet")["audit_id"])
    shared_top20 = mlp_top20 & cross_top20
    shared_mask = comp["audit_id"].isin(shared_top20).to_numpy()

    ax.scatter(
        x_plot[~shared_mask],
        y_plot[~shared_mask],
        s=17,
        alpha=.52,
        color="#6688A3",
        edgecolors="white",
        linewidths=.25,
        label="Other endpoints",
        rasterized=True,
    )
    ax.scatter(
        x_plot[shared_mask],
        y_plot[shared_mask],
        s=26,
        alpha=.9,
        color="#E9B949",
        edgecolors="#5C4714",
        linewidths=.35,
        label="Shared top 20",
        rasterized=True,
    )
    ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    rho = spearman(x, y)
    ax.text(
        .04,
        .96,
        rf"Spearman $\rho={rho:.3f}$" + "\n"
        + f"Shared top 20: {len(shared_top20)}/20",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": .88,
              "edgecolor": "#CCCCCC", "linewidth": .5},
    )
    ax.set_xlabel(r"MLP endpoint-median $H/s$")
    ax.set_ylabel(r"CrossNet endpoint-median $H/s$")
    ax.set_title("(c) Cross-architecture replication", pad=7)
    ax.grid(True, which="major")
    ax.legend(loc="lower right", frameon=False, handletextpad=.35)

    fig.subplots_adjust(left=.06, right=.99, bottom=.23, top=.84, wspace=.34)
    return save_figure(fig, out_dir, "e3_main_fitted_audit", dpi)


def panel_threshold_surfaces(
    audits: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, np.ndarray]]:
    surfaces = {
        "Softplus MLP": threshold_surface(audits, "Softplus MLP"),
        "CrossNet": threshold_surface(audits, "CrossNet"),
    }
    vmax = max(int(matrix.max()) for matrix in surfaces.values())
    norm = Normalize(vmin=0, vmax=max(vmax, 1))
    cmap = mpl.colormaps["Blues"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.90), sharex=True, sharey=True)
    images = []
    for panel, (ax, model) in enumerate(zip(axes, ["Softplus MLP", "CrossNet"])):
        matrix = surfaces[model]
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        images.append(im)
        ax.set_xticks(np.arange(len(KAPPAS)))
        ax.set_xticklabels([f"{x:g}" for x in KAPPAS])
        ax.set_yticks(np.arange(len(TAUS)))
        ax.set_yticklabels([f"{x:g}" for x in TAUS])
        ax.set_xlabel(r"agreement tolerance $\kappa$: $D/s\leq\kappa$")
        if panel == 0:
            ax.set_ylabel(r"hidden threshold $\tau$: $H/s\geq\tau$")
        ax.set_title(f"({'ab'[panel]}) {model}", pad=7)
        for i in range(len(TAUS)):
            for j in range(len(KAPPAS)):
                value = int(matrix[i, j])
                color = "white" if norm(value) > .58 else "#1A1A1A"
                ax.text(j, i, str(value), ha="center", va="center", color=color,
                        fontsize=8, fontweight="bold" if value else "normal")
        primary_i = TAUS.index(FC_TAU)
        primary_j = KAPPAS.index(FC_KAPPA)
        ax.add_patch(Rectangle(
            (primary_j - .44, primary_i - .44), .88, .88,
            fill=False, edgecolor="#111111", linewidth=1.35,
        ))

    cax = fig.add_axes([.915, .21, .018, .58])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(r"endpoints satisfying the criterion in $\geq3/5$ fits")
    fig.suptitle(
        "Complete prespecified endpoint-level threshold surfaces",
        fontsize=9.4,
        y=.995,
    )
    fig.subplots_adjust(left=.085, right=.885, bottom=.22, top=.82, wspace=.12)
    return save_figure(fig, out_dir, "e3_app_threshold_surfaces", dpi), surfaces


def certification_ratio(audits: pd.DataFrame) -> np.ndarray:
    scale = audits["output_scale_s"].to_numpy(float)
    tolerance = np.maximum(1e-10, 1e-8 * np.maximum(scale, 1.0))
    nested = np.abs(audits["nested_quadrature_error"].to_numpy(float)) / tolerance
    cert_values = np.abs(audits[CERTIFICATE_ERROR_COLUMNS].to_numpy(float))
    cert = np.max(cert_values, axis=1) / (10.0 * tolerance)
    return np.maximum(nested, cert)


def panel_predictive_numerical(
    audits: pd.DataFrame,
    fits: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    rng = np.random.default_rng(20260861)
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 2.75))
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    short_labels = ["Additive", "Quadratic", "MLP", "CrossNet"]

    # (a) Predictive validity across the five fits.
    ax = axes[0]
    for pos, model in zip(positions, MODEL_ORDER):
        values = fits.loc[fits["model"] == model, "test_r2"].to_numpy(float)
        jitter = rng.uniform(-.07, .07, len(values))
        ax.scatter(
            pos + jitter, values, s=23, color=MODEL_COLORS[model],
            alpha=.85, edgecolors="white", linewidths=.4, zorder=3,
        )
        ax.hlines(np.median(values), pos-.20, pos+.20, color="#111111", linewidth=1.25)
        ax.vlines(pos, values.min(), values.max(), color="#555555", linewidth=.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=18)
    ax.set_ylabel(r"held-out test $R^2$")
    ax.set_title("(a) Predictive validity", pad=7)
    ax.grid(True, axis="y")

    # (b) Certification residual relative to its declared tolerance.
    ax = axes[1]
    ratio_by_model = []
    for model in MODEL_ORDER:
        g = audits.loc[audits["model"] == model]
        ratio_by_model.append(np.maximum(certification_ratio(g), 1e-18))
    bp = ax.boxplot(
        ratio_by_model,
        positions=positions,
        widths=.52,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        tick_labels=short_labels,
        medianprops={"color": "#111111", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": .75},
        capprops={"color": "#555555", "linewidth": .75},
    )
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(.62)
        patch.set_edgecolor("#333333")
    for pos, model, values in zip(positions, MODEL_ORDER, ratio_by_model):
        jitter = rng.uniform(-.14, .14, len(values))
        ax.scatter(pos+jitter, values, s=5, alpha=.18, color=MODEL_COLORS[model],
                   edgecolors="none", rasterized=True)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=.9, label="pass boundary")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=18)
    ax.set_ylabel("normalized certification residual")
    ax.set_title("(b) Numerical certification", pad=7)
    ax.grid(True, which="major", axis="y")
    ax.legend(loc="upper left", frameon=False)

    # (c) Adaptive quadrature resolution.
    ax = axes[2]
    observed_orders = sorted(int(x) for x in audits["quadrature_order"].unique())
    canonical_orders = [16, 32, 64, 128, 256]
    orders = [order for order in canonical_orders if order in observed_orders]
    orders.extend(order for order in observed_orders if order not in orders)
    quad_colors = mpl.colormaps["viridis"](np.linspace(.20, .82, max(len(orders), 2)))
    bottoms = np.zeros(len(MODEL_ORDER), dtype=float)
    for oi, order in enumerate(orders):
        counts = np.array([
            int((audits.loc[audits["model"] == model, "quadrature_order"] == order).sum())
            for model in MODEL_ORDER
        ])
        ax.bar(
            positions,
            counts,
            bottom=bottoms,
            width=.62,
            color=quad_colors[oi],
            edgecolor="white",
            linewidth=.4,
            label=str(order),
        )
        for pos, bottom, count in zip(positions, bottoms, counts):
            if count >= 35:
                ax.text(pos, bottom + count/2, str(int(count)), ha="center", va="center",
                        fontsize=6.6, color="white" if oi >= len(orders)/2 else "#111111")
        bottoms += counts
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=18)
    ax.set_ylim(0, 585)
    ax.set_ylabel("certified audits (out of 500)")
    ax.set_title("(c) Adaptive quadrature order", pad=7)
    ax.grid(True, axis="y")
    ax.legend(title="order", frameon=False, ncol=min(3, len(orders)),
              loc="upper center", bbox_to_anchor=(.5, .99),
              fontsize=6.5, title_fontsize=6.5, handlelength=.9, columnspacing=.8)

    fig.subplots_adjust(left=.065, right=.99, bottom=.25, top=.82, wspace=.34)
    return save_figure(fig, out_dir, "e3_app_predictive_numerical", dpi)


def panel_cross_architecture_details(
    comp: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 2.75))

    # (a) Visible discrepancy association.
    ax = axes[0]
    x = comp["median_D_over_s_mlp"].to_numpy(float)
    y = comp["median_D_over_s_crossnet"].to_numpy(float)
    floor = positive_floor(np.concatenate([x, y]), reference=FC_KAPPA)
    xp, yp = np.maximum(x, floor), np.maximum(y, floor)
    low, high = min(xp.min(), yp.min())*.72, max(xp.max(), yp.max())*1.28
    ax.scatter(xp, yp, s=18, alpha=.62, color="#6688A3", edgecolors="white",
               linewidths=.25, rasterized=True)
    ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel(r"MLP endpoint-median $D/s$")
    ax.set_ylabel(r"CrossNet endpoint-median $D/s$")
    ax.set_title(
        "(a) Visible discrepancy\n"
        + rf"Spearman $\rho={spearman(x, y):.3f}$",
        pad=6,
    )
    ax.grid(True, which="major")

    # (b) Concealed fraction association.
    ax = axes[1]
    x = comp["median_chi_mlp"].to_numpy(float)
    y = comp["median_chi_crossnet"].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    low = min(0.0, float(min(x.min(), y.min())) - .03)
    high = max(1.0, float(max(x.max(), y.max())) + .03)
    ax.scatter(x, y, s=18, alpha=.62, color="#8B6BB1", edgecolors="white",
               linewidths=.25, rasterized=True)
    ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=.8)
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel(r"MLP endpoint-median $\chi$")
    ax.set_ylabel(r"CrossNet endpoint-median $\chi$")
    ax.set_title(
        "(b) Concealed fraction\n"
        + rf"Spearman $\rho={spearman(x, y):.3f}$",
        pad=6,
    )
    ax.grid(True)

    # (c) Exact joint recurrence counts; a count heatmap avoids overplotting.
    ax = axes[2]
    recurrence = np.zeros((6, 6), dtype=int)
    for _, row in comp.iterrows():
        mi = int(row["strict_fc_seed_count_mlp"])
        ci = int(row["strict_fc_seed_count_crossnet"])
        recurrence[ci, mi] += 1
    im = ax.imshow(recurrence, origin="lower", cmap="Blues", aspect="equal")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel("MLP strict-event seed count")
    ax.set_ylabel("CrossNet strict-event seed count")
    ax.set_title("(c) Strict-event recurrence", pad=7)
    max_count = max(int(recurrence.max()), 1)
    for i in range(6):
        for j in range(6):
            value = int(recurrence[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value/max_count > .50 else "#111111",
                        fontsize=7.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cbar.set_label("endpoints")

    fig.subplots_adjust(left=.065, right=.985, bottom=.24, top=.82, wspace=.34)
    return save_figure(fig, out_dir, "e3_app_cross_architecture", dpi)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_outputs(
    audits: pd.DataFrame,
    fits: pd.DataFrame,
    endpoints: pd.DataFrame,
    comp: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    input_paths: dict[str, Path],
    figure_paths: list[Path],
    out_dir: Path,
) -> tuple[Path, Path]:
    mlp_top10 = set(comp.nlargest(10, "median_H_over_s_mlp")["audit_id"])
    cross_top10 = set(comp.nlargest(10, "median_H_over_s_crossnet")["audit_id"])
    mlp_top20 = set(comp.nlargest(20, "median_H_over_s_mlp")["audit_id"])
    cross_top20 = set(comp.nlargest(20, "median_H_over_s_crossnet")["audit_id"])

    values = {
        "contract": {
            "n_endpoints": EXPECTED_ENDPOINTS,
            "model_seeds": EXPECTED_SEEDS,
            "strict_operating_point": {"kappa_D_over_s_max": FC_KAPPA,
                                       "tau_H_over_s_min": FC_TAU},
            "majority_definition": f"event holds in at least {MAJORITY_SEEDS}/5 fits",
        },
        "model_summary": {},
        "cross_architecture": {
            "median_H_over_s": {
                "pearson": pearson(comp["median_H_over_s_mlp"],
                                   comp["median_H_over_s_crossnet"]),
                "spearman": spearman(comp["median_H_over_s_mlp"],
                                     comp["median_H_over_s_crossnet"]),
            },
            "median_D_over_s": {
                "pearson": pearson(comp["median_D_over_s_mlp"],
                                   comp["median_D_over_s_crossnet"]),
                "spearman": spearman(comp["median_D_over_s_mlp"],
                                     comp["median_D_over_s_crossnet"]),
            },
            "median_chi": {
                "pearson": pearson(comp["median_chi_mlp"], comp["median_chi_crossnet"]),
                "spearman": spearman(comp["median_chi_mlp"], comp["median_chi_crossnet"]),
            },
            "top10_shared": len(mlp_top10 & cross_top10),
            "top20_shared": len(mlp_top20 & cross_top20),
            "material_majority_shared": int((
                (comp["material_seed_count_mlp"] >= MAJORITY_SEEDS)
                & (comp["material_seed_count_crossnet"] >= MAJORITY_SEEDS)
            ).sum()),
        },
        "threshold_surfaces_rows_tau_columns_kappa": {
            model: matrix.tolist() for model, matrix in surfaces.items()
        },
    }

    for model in MODEL_ORDER:
        a = audits.loc[audits["model"] == model]
        f = fits.loc[fits["model"] == model]
        e = endpoints.loc[endpoints["model"] == model]
        values["model_summary"][model] = {
            "n_audits": int(len(a)),
            "test_R2_min": float(f["test_r2"].min()),
            "test_R2_max": float(f["test_r2"].max()),
            "audit_max_H_over_s": float(a["H_over_s"].max()),
            "audit_row_median_D_over_s": float(a["D_over_s"].median()),
            "audit_row_median_H_over_s": float(a["H_over_s"].median()),
            "audit_row_median_chi": (
                float(a["chi"].median()) if a["chi"].notna().any() else None
            ),
            "endpoint_median_D_over_s": float(e["median_D_over_s"].median()),
            "endpoint_median_H_over_s": float(e["median_H_over_s"].median()),
            "endpoint_median_chi": (
                float(e["median_chi"].median())
                if e["median_chi"].notna().any() else None
            ),
            "material_endpoints_in_majority": int((
                e["material_seed_count"] >= MAJORITY_SEEDS
            ).sum()),
            "strict_endpoints_in_majority": int((
                e["strict_fc_seed_count"] >= MAJORITY_SEEDS
            ).sum()),
            "max_normalized_certification_residual": float(
                certification_ratio(a).max()
            ),
        }

    values_path = out_dir / "e3_figure_values.json"
    with open(values_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(values, handle, indent=2, ensure_ascii=False, default=json_number)
        handle.write("\n")

    manifest = {
        "script": Path(__file__).name,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": {},
    }
    for path in [*figure_paths, values_path]:
        manifest["outputs"][path.name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    manifest_path = out_dir / "e3_figure_manifest_sha256.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return values_path, manifest_path


def run(args: argparse.Namespace) -> None:
    configure_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audits, fits, input_paths = load_all(args)
    endpoints = endpoint_summary(audits)
    comp = paired_fitted_endpoints(endpoints)

    figure_paths: list[Path] = []
    figure_paths += panel_main(audits, endpoints, comp, out_dir, args.dpi)
    paths, surfaces = panel_threshold_surfaces(audits, out_dir, args.dpi)
    figure_paths += paths
    figure_paths += panel_predictive_numerical(audits, fits, out_dir, args.dpi)
    figure_paths += panel_cross_architecture_details(comp, out_dir, args.dpi)

    values_path, manifest_path = write_outputs(
        audits=audits,
        fits=fits,
        endpoints=endpoints,
        comp=comp,
        surfaces=surfaces,
        input_paths=input_paths,
        figure_paths=figure_paths,
        out_dir=out_dir,
    )

    print()
    print("=" * 88)
    print("E3a — PAPER FIGURES GENERATED")
    print("=" * 88)
    print(f"Validated audits                  : {len(audits)}")
    print(f"Paired endpoints                  : {EXPECTED_ENDPOINTS}")
    print("All audits resolved/certified     : True")
    print()
    print("Main paper:")
    print(f"  {out_dir / 'e3_main_fitted_audit.pdf'}")
    print()
    print("Appendix:")
    print(f"  {out_dir / 'e3_app_threshold_surfaces.pdf'}")
    print(f"  {out_dir / 'e3_app_predictive_numerical.pdf'}")
    print(f"  {out_dir / 'e3_app_cross_architecture.pdf'}")
    print()
    print(f"Exact plotted values              : {values_path}")
    print(f"SHA-256 manifest                  : {manifest_path}")
    print("PNG copies were also written at the requested DPI.")
    print("=" * 88)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all main-paper and appendix E3a figures from frozen outputs."
    )
    parser.add_argument("--controls-dir", default="./e3_tabular_controls")
    parser.add_argument("--mlp-dir", default="./e3_tabular_mlp")
    parser.add_argument("--crossnet-dir", default="./e3_tabular_crossnet")
    parser.add_argument("--out-dir", default="./ICLR/paper_figures")
    parser.add_argument("--dpi", type=int, default=400)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
