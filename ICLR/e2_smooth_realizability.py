#!/usr/bin/env python3
"""
E2 — Smooth realizability and observational indistinguishability
================================================================

Theory-controlled experiment for:
  (a) unrestricted smooth realizability of prescribed pot allocations;
  (b) positive-complementarity cyclic families;
  (c) two smooth models with identical endpoint game, identical Baseline
      Shapley vector, identical straight-line IG vector, but different
      interaction-transfer ledgers.

This script is intentionally self-contained: NumPy, pandas, matplotlib only.

Locked defaults
---------------
E2a:
  pot sizes k in {2,3,4,5}
  100 targets per k
  phi = 1
  target redistribution R sampled uniformly from [0.05, 1.0]
  seed = 20260829

E2b:
  cycle lengths ell in {3,4,6,8}
  exponents p in {1,2,3,5,9,20}

E2c:
  ell = 6
  F0 = sum_i t_i t_{i+1}
  F1 = sum_i t_i^9 t_{i+1}

Pre-registered numerical requirements
--------------------------------------
E2a planted allocation error       <= 1e-10
E2a planted transfer error         <= 1e-10
pot conservation residual          <= 1e-10
corner / endpoint error            <= 1e-12
E2b R,H,D prediction error         <= 1e-10
E2b visible discrepancy D          <= 1e-10
E2c complete corner-game diff      <= 1e-12
E2c Baseline-Shapley diff          <= 1e-12
E2c IG diff                        <= 1e-10
E2c recovered R(F1)                = 2.4 +/- 1e-10

Outputs
-------
e2_outputs/
  e2a_results.csv
  e2b_results.csv
  e2c_summary.csv
  e2_summary.json
  e2a_planted_vs_recovered.pdf/png
  e2b_positive_complementarity.pdf/png
  e2c_observational_indistinguishability.pdf/png

Run
---
python e2_smooth_realizability.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Generic polynomial helper
# =====================================================================

@dataclass(frozen=True)
class Term:
    coeff: float
    exponents: tuple[int, ...]


class Polynomial:
    def __init__(self, dim: int, terms: list[Term]):
        self.dim = int(dim)
        self.terms = terms
        for term in terms:
            if len(term.exponents) != self.dim:
                raise ValueError("Term exponent length does not match polynomial dimension.")

    def eval_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")
        out = np.zeros(X.shape[0], dtype=float)
        for term in self.terms:
            exps = np.asarray(term.exponents, dtype=int)
            out += term.coeff * np.prod(np.power(X, exps), axis=1)
        return out

    def eval(self, x: np.ndarray) -> float:
        return float(self.eval_batch(np.asarray(x, dtype=float))[0])

    def grad_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")

        G = np.zeros((X.shape[0], self.dim), dtype=float)
        for term in self.terms:
            exps = np.asarray(term.exponents, dtype=int)
            for j in range(self.dim):
                if exps[j] == 0:
                    continue
                dexps = exps.copy()
                dexps[j] -= 1
                G[:, j] += (
                    term.coeff
                    * exps[j]
                    * np.prod(np.power(X, dexps), axis=1)
                )
        return G


# =====================================================================
# Numerical attribution utilities
# =====================================================================

def straight_line_ig(poly: Polynomial, order: int = 64) -> np.ndarray:
    """
    Straight-line IG / path allocation from 0 to 1.
    Since x(s)=s*1 and dx_i/ds=1, allocation_i = int_0^1 partial_i F(s1) ds.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    X = np.repeat(s[:, None], poly.dim, axis=1)
    G = poly.grad_batch(X)
    return w @ G


def all_corner_values(poly: Polynomial) -> np.ndarray:
    n = 1 << poly.dim
    X = np.zeros((n, poly.dim), dtype=float)
    for mask in range(n):
        for i in range(poly.dim):
            X[mask, i] = 1.0 if (mask >> i) & 1 else 0.0
    return poly.eval_batch(X)


def exact_shapley_from_game(game: np.ndarray, d: int) -> np.ndarray:
    if len(game) != (1 << d):
        raise ValueError("Game length must be 2^d.")

    phi = np.zeros(d, dtype=float)
    for i in range(d):
        for mask in range(1 << d):
            if (mask >> i) & 1:
                continue
            s = int(mask.bit_count())
            weight = 1.0 / (d * math.comb(d - 1, s))
            phi[i] += weight * (game[mask | (1 << i)] - game[mask])
    return phi


def ledger_metrics(T: np.ndarray) -> tuple[float, float, float]:
    """
    T shape: (d, n_pots)
    """
    T = np.asarray(T, dtype=float)
    R = 0.5 * float(np.abs(T).sum())
    margins = T.sum(axis=1)
    D = 0.5 * float(np.abs(margins).sum())
    H = R - D
    return R, D, H

# =====================================================================
# Paper-style figure helpers
# =====================================================================

def _paper_axis_style(ax):
    ax.tick_params(labelsize=9)
    ax.grid(
        True,
        color="#E8ECEF",
        linewidth=0.6,
        alpha=0.75,
        zorder=0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#59636B")


def _make_panel_a(ax, planted: np.ndarray, recovered: np.ndarray):
    x = np.asarray(planted, dtype=float)
    y = np.asarray(recovered, dtype=float)

    ax.scatter(
        x, y,
        s=18,
        facecolor="#6F8FAF",   # muted paper blue
        edgecolor="none",
        alpha=0.72,
        zorder=2,
    )

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.04 * max(1e-12, hi - lo)

    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="#98A2AA",     # soft neutral gray
        linestyle="--",
        linewidth=1.0,
        zorder=1
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Planted transfer", fontsize=10)
    ax.set_ylabel("Recovered transfer", fontsize=10)
    ax.set_title("(a) Smooth realizability of prescribed transfers", fontsize=11, pad=8)

    max_err = float(np.max(np.abs(x - y)))
    ax.text(
        0.05, 0.97,
        f"400 problems; max error = {max_err:.1e}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )

    _paper_axis_style(ax)


def _make_panel_b(ax, e2b: pd.DataFrame):
    style_map = {
        3: dict(color="#5F7FA3", marker="o"),  # muted blue
        4: dict(color="#7E9B87", marker="s"),  # muted sage
        6: dict(color="#B08B73", marker="^"),  # muted terracotta
        8: dict(color="#8D84A8", marker="D"),  # muted lavender
    }

    for ell in sorted(e2b["ell"].unique()):
        g = e2b[e2b["ell"] == ell].sort_values("p")
        style = style_map.get(int(ell), dict(color="#6F7780", marker="o"))
        ax.plot(
            g["p"].to_numpy(),
            g["H"].to_numpy(),
            linewidth=1.5,
            markersize=5.5,
            label=rf"$\ell={int(ell)}$",
            **style
        )

    ax.set_xlabel(r"Asymmetry exponent $p$", fontsize=10)
    ax.set_ylabel(r"Hidden redistribution $H$", fontsize=10)
    ax.set_title("(b) Positive-complementarity cycle family", fontsize=11, pad=8)

    max_D = float(e2b["D"].max())
    ax.text(
        0.05, 0.97,
        rf"max visible $D = {max_D:.1e}$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )
    ax.text(
        0.05, 0.86,
        r"$H=\ell\frac{(p-1)}{2(p+1)}$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=12
    )

    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _paper_axis_style(ax)


def _draw_round_box(ax, xy, w, h, text, fontsize=10, weight="normal", fc="1.0", ec="0.35"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight
    )


def _make_panel_c(ax, e2c_vals: dict):
    ax.set_title("(c) Same observables, different ledger", fontsize=11, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header boxes
    _draw_round_box(
        ax, (0.50, 0.86), 0.14, 0.08, "F0",
        fontsize=11, weight="bold",
        fc="#EEF4F8", ec="#7F9DB5",
    )
    _draw_round_box(
        ax, (0.74, 0.86), 0.14, 0.08, "F1",
        fontsize=11, weight="bold",
        fc="#EFF5F0", ec="#819B88",
    )

    # Column guides
    ax.plot([0.47, 0.47], [0.23, 0.83], color="0.80", linewidth=0.8)
    ax.plot([0.71, 0.71], [0.23, 0.83], color="0.80", linewidth=0.8)

    # Horizontal separators
    for y in [0.66, 0.49, 0.32]:
        ax.plot([0.03, 0.93], [y, y], color="0.75", linestyle=":", linewidth=1.0)

    rows = [
        ("Complete\nendpoint game", f"{e2c_vals['corner_diff']:.1e}", f"{e2c_vals['corner_diff']:.1e}"),
        ("Baseline\nShapley", f"{e2c_vals['b_diff']:.1e}", f"{e2c_vals['b_diff']:.1e}"),
        ("Integrated\nGradients", f"{e2c_vals['ig_diff']:.1e}", f"{e2c_vals['ig_diff']:.1e}"),
    ]
    ys = [0.75, 0.58, 0.41]

    # Compact numerical label for the two matching observable columns.
    ax.text(
        0.69, 0.825, r"max $|\Delta|$",
        ha="center", va="center",
        fontsize=8.5, color="#68727A",
        transform=ax.transAxes,
    )

    for (label, left_txt, right_txt), y in zip(rows, ys):
        ax.text(0.08, y, label, ha="left", va="center", fontsize=9.6, transform=ax.transAxes)

        ax.text(
            0.57, y + 0.018, "✓",
            ha="center", va="center",
            fontsize=20, color="#6F9277",
            transform=ax.transAxes,
        )
        ax.text(
            0.57, y - 0.058, left_txt,
            ha="center", va="center",
            fontsize=8.1, color="#4F5961",
            transform=ax.transAxes,
        )

        ax.text(
            0.81, y + 0.018, "✓",
            ha="center", va="center",
            fontsize=20, color="#6F9277",
            transform=ax.transAxes,
        )
        ax.text(
            0.81, y - 0.058, right_txt,
            ha="center", va="center",
            fontsize=8.1, color="#4F5961",
            transform=ax.transAxes,
        )

    # Ledger contrast box
    ledger_box = FancyBboxPatch(
        (0.04, 0.05),
        0.87,
        0.18,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.1,
        edgecolor="#B7A08D",
        facecolor="#FBF6F1",
        transform=ax.transAxes
    )
    ax.add_patch(ledger_box)

    ax.plot([0.38, 0.38], [0.07, 0.21], color="0.82", linewidth=0.8, transform=ax.transAxes)
    ax.plot([0.70, 0.70], [0.07, 0.21], color="0.82", linewidth=0.8, transform=ax.transAxes)

    ax.text(0.13, 0.14, "Ledger\nredistribution", ha="left", va="center",
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.53, 0.14, rf"$R = {e2c_vals['R0']:.1f}$", ha="center", va="center",
            fontsize=12.5, transform=ax.transAxes)
    ax.text(0.82, 0.14, rf"$R = {e2c_vals['R1']:.1f}$", ha="center", va="center",
            fontsize=12.5, transform=ax.transAxes)
    ax.text(0.675, 0.14, "↔", ha="center", va="center",
            fontsize=18, color="#8A7768", transform=ax.transAxes)


def make_combined_e2_figure(
    planted: np.ndarray,
    recovered: np.ndarray,
    e2b: pd.DataFrame,
    e2c_vals: dict,
    out_dir: Path
):
    # Compact paper layout with wide panels for (a) and (b).
    fig = plt.figure(figsize=(12.8, 3.25))
    gs = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1.08, 1.25, 1.22],
        wspace=0.25,
        left=0.055,
        right=0.985,
        bottom=0.20,
        top=0.84,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _make_panel_a(ax1, planted, recovered)
    _make_panel_b(ax2, e2b)
    _make_panel_c(ax3, e2c_vals)

    # Height / width ratios for the two actual plots.
    ax1.set_box_aspect(0.72)
    ax2.set_box_aspect(0.72)

    fig.savefig(out_dir / "e2_three_panel_paper.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2_three_panel_paper.pdf", bbox_inches="tight")
    plt.close(fig)
# =====================================================================
# E2a — unrestricted smooth realizability
# =====================================================================

def realization_polynomial(k: int, z: np.ndarray, phi: float = 1.0) -> Polynomial:
    """
    Explicit diagonal-path construction from the proof:

      r(t) = phi prod_j t_j
             + (k+1) sum_{i=1}^{k-1}
               z_i t_i t_k (t_i - t_k)
               prod_{l notin {i,k}} t_l.

    Coordinates are 0-indexed; coordinate k-1 is the reference coordinate.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (k,):
        raise ValueError("z must have shape (k,).")
    if abs(float(z.sum())) > 1e-10:
        raise ValueError("z must be zero-sum.")

    terms: list[Term] = []

    # Base unanimity monomial.
    terms.append(Term(float(phi), tuple([1] * k)))

    ref = k - 1
    for i in range(k - 1):
        c = float((k + 1) * z[i])

        # + c * t_i^2 * t_ref * prod(other coordinates)
        e_plus = np.ones(k, dtype=int)
        e_plus[i] = 2
        terms.append(Term(c, tuple(int(x) for x in e_plus)))

        # - c * t_i * t_ref^2 * prod(other coordinates)
        e_minus = np.ones(k, dtype=int)
        e_minus[ref] = 2
        terms.append(Term(-c, tuple(int(x) for x in e_minus)))

    return Polynomial(k, terms)


def run_e2a(args, out_dir: Path):
    rng = np.random.default_rng(args.seed)
    rows = []
    scatter_planted = []
    scatter_recovered = []

    for k in args.ks:
        equal = np.full(k, 1.0 / k)

        for rep in range(args.reps):
            raw = rng.normal(size=k)
            raw -= raw.mean()

            raw_R = 0.5 * float(np.abs(raw).sum())
            if raw_R < 1e-14:
                raw[0] = 1.0
                raw[1] = -1.0
                raw_R = 1.0

            target_R = float(rng.uniform(args.R_min, args.R_max))
            z = raw * (target_R / raw_R)
            a = equal + z

            poly = realization_polynomial(k, z, phi=1.0)

            ig = straight_line_ig(poly, order=args.quad_order)
            recovered_z = ig - equal

            game = all_corner_values(poly)
            expected_game = np.zeros_like(game)
            expected_game[-1] = 1.0

            allocation_err = float(np.max(np.abs(ig - a)))
            transfer_err = float(np.max(np.abs(recovered_z - z)))
            corner_err = float(np.max(np.abs(game - expected_game)))
            conservation_err = abs(float(ig.sum()) - 1.0)
            planted_conservation = abs(float(z.sum()))

            rows.append({
                "k": k,
                "rep": rep,
                "target_R": target_R,
                "min_target_allocation": float(a.min()),
                "max_target_allocation": float(a.max()),
                "has_negative_target_allocation": bool(np.any(a < 0)),
                "max_allocation_error": allocation_err,
                "max_transfer_error": transfer_err,
                "corner_error": corner_err,
                "pot_conservation_error": conservation_err,
                "planted_zero_sum_error": planted_conservation,
            })

            scatter_planted.extend(z.tolist())
            scatter_recovered.extend(recovered_z.tolist())

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e2a_results.csv", index=False)

    planted = np.asarray(scatter_planted, dtype=float)
    recovered = np.asarray(scatter_recovered, dtype=float)

    # Optional standalone panel
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    _make_panel_a(ax, planted, recovered)
    fig.tight_layout()
    fig.savefig(out_dir / "e2a_planted_vs_recovered.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2a_planted_vs_recovered.pdf", bbox_inches="tight")
    plt.close(fig)

    return df, planted, recovered


# =====================================================================
# E2b — positive-complementarity cyclic family
# =====================================================================

def cycle_polynomial(ell: int, p: int) -> Polynomial:
    terms: list[Term] = []
    for i in range(ell):
        j = (i + 1) % ell
        exps = np.zeros(ell, dtype=int)
        exps[i] = p
        exps[j] += 1
        terms.append(Term(1.0, tuple(int(x) for x in exps)))
    return Polynomial(ell, terms)


def cycle_ledger(ell: int, p: int, quad_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (T, bshap_total, ig_total) for the pure pair pots of the cycle.
    """
    T_cols = []
    b_total = np.zeros(ell)
    ig_total = np.zeros(ell)

    for i in range(ell):
        j = (i + 1) % ell

        exps = np.zeros(ell, dtype=int)
        exps[i] = p
        exps[j] += 1
        pot_poly = Polynomial(
            ell,
            [Term(1.0, tuple(int(x) for x in exps))]
        )

        ig_pot = straight_line_ig(pot_poly, order=quad_order)

        b_pot = np.zeros(ell)
        b_pot[i] = 0.5
        b_pot[j] = 0.5

        T_cols.append(ig_pot - b_pot)
        b_total += b_pot
        ig_total += ig_pot

    T = np.stack(T_cols, axis=1)
    return T, b_total, ig_total


def run_e2b(args, out_dir: Path) -> pd.DataFrame:
    rows = []

    for ell in args.ells:
        for p in args.ps:
            poly = cycle_polynomial(ell, p)

            game = all_corner_values(poly)
            bshap_exact = exact_shapley_from_game(game, ell)
            ig_direct = straight_line_ig(poly, order=args.quad_order)

            T, b_pot_sum, ig_pot_sum = cycle_ledger(
                ell, p, quad_order=args.quad_order
            )
            R, D, H = ledger_metrics(T)

            cp = (p - 1.0) / (2.0 * (p + 1.0))
            R_expected = ell * cp
            H_expected = ell * cp
            D_expected = 0.0

            rows.append({
                "ell": ell,
                "p": p,
                "R": R,
                "D": D,
                "H": H,
                "R_expected": R_expected,
                "D_expected": D_expected,
                "H_expected": H_expected,
                "R_error": abs(R - R_expected),
                "D_error": abs(D - D_expected),
                "H_error": abs(H - H_expected),
                "max_full_bshap_vs_pot_sum": float(np.max(np.abs(bshap_exact - b_pot_sum))),
                "max_full_ig_vs_pot_sum": float(np.max(np.abs(ig_direct - ig_pot_sum))),
                "max_aggregate_ig_minus_bshap": float(np.max(np.abs(ig_direct - bshap_exact))),
                "corner_game_baseline_value": float(game[0]),
                "corner_game_full_value": float(game[-1]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e2b_results.csv", index=False)

    # Optional standalone panel
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    _make_panel_b(ax, df)
    fig.tight_layout()
    fig.savefig(out_dir / "e2b_positive_complementarity.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2b_positive_complementarity.pdf", bbox_inches="tight")
    plt.close(fig)

    return df


# =====================================================================
# E2c — same observables, different ledger
# =====================================================================

def run_e2c(args, out_dir: Path):
    ell = 6
    p0 = 1
    p1 = 9

    F0 = cycle_polynomial(ell, p0)
    F1 = cycle_polynomial(ell, p1)

    game0 = all_corner_values(F0)
    game1 = all_corner_values(F1)
    b0 = exact_shapley_from_game(game0, ell)
    b1 = exact_shapley_from_game(game1, ell)
    ig0 = straight_line_ig(F0, order=args.quad_order)
    ig1 = straight_line_ig(F1, order=args.quad_order)

    T0, _, _ = cycle_ledger(ell, p0, args.quad_order)
    T1, _, _ = cycle_ledger(ell, p1, args.quad_order)
    R0, D0, H0 = ledger_metrics(T0)
    R1, D1, H1 = ledger_metrics(T1)

    corner_diff = float(np.max(np.abs(game0 - game1)))
    b_diff = float(np.max(np.abs(b0 - b1)))
    ig_diff = float(np.max(np.abs(ig0 - ig1)))
    ledger_max_diff = float(np.max(np.abs(T0 - T1)))

    rows = [
        {
            "quantity": "complete endpoint game (max abs diff)",
            "F0_or_difference": corner_diff,
            "F1": np.nan,
        },
        {
            "quantity": "Baseline Shapley vector (max abs diff)",
            "F0_or_difference": b_diff,
            "F1": np.nan,
        },
        {
            "quantity": "Integrated Gradients vector (max abs diff)",
            "F0_or_difference": ig_diff,
            "F1": np.nan,
        },
        {
            "quantity": "ledger redistribution R",
            "F0_or_difference": R0,
            "F1": R1,
        },
        {
            "quantity": "hidden redistribution H",
            "F0_or_difference": H0,
            "F1": H1,
        },
        {
            "quantity": "visible discrepancy D",
            "F0_or_difference": D0,
            "F1": D1,
        },
        {
            "quantity": "max abs ledger-entry difference",
            "F0_or_difference": ledger_max_diff,
            "F1": np.nan,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e2c_summary.csv", index=False)

    e2c_vals = {
        "corner_diff": corner_diff,
        "b_diff": b_diff,
        "ig_diff": ig_diff,
        "R0": R0,
        "R1": R1,
        "H1": H1,
        "D1": D1,
        "ledger_max_diff": ledger_max_diff,
    }

    # Optional standalone panel
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    _make_panel_c(ax, e2c_vals)
    fig.tight_layout()
    fig.savefig(out_dir / "e2c_observational_indistinguishability.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e2c_observational_indistinguishability.pdf", bbox_inches="tight")
    plt.close(fig)

    return df, e2c_vals


# =====================================================================
# Main + locked checks
# =====================================================================

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e2a, planted, recovered = run_e2a(args, out_dir)
    e2b = run_e2b(args, out_dir)
    e2c, e2c_vals = run_e2c(args, out_dir)

    make_combined_e2_figure(
        planted=planted,
        recovered=recovered,
        e2b=e2b,
        e2c_vals=e2c_vals,
        out_dir=out_dir,
    )

    c_corner = float(e2c_vals["corner_diff"])
    c_bshap = float(e2c_vals["b_diff"])
    c_ig = float(e2c_vals["ig_diff"])
    c_R0 = float(e2c_vals["R0"])
    c_R1 = float(e2c_vals["R1"])
    c_H1 = float(e2c_vals["H1"])
    c_D1 = float(e2c_vals["D1"])
    checks = {
        "E2a_max_allocation_error": float(e2a["max_allocation_error"].max()),
        "E2a_max_transfer_error": float(e2a["max_transfer_error"].max()),
        "E2a_max_corner_error": float(e2a["corner_error"].max()),
        "E2a_max_pot_conservation_error": float(e2a["pot_conservation_error"].max()),
        "E2a_negative_target_fraction": float(e2a["has_negative_target_allocation"].mean()),

        "E2b_max_R_error": float(e2b["R_error"].max()),
        "E2b_max_D_error": float(e2b["D_error"].max()),
        "E2b_max_H_error": float(e2b["H_error"].max()),
        "E2b_max_visible_D": float(e2b["D"].max()),
        "E2b_max_full_bshap_vs_pot_sum": float(e2b["max_full_bshap_vs_pot_sum"].max()),
        "E2b_max_full_ig_vs_pot_sum": float(e2b["max_full_ig_vs_pot_sum"].max()),

        "E2c_corner_game_max_diff": c_corner,
        "E2c_bshap_max_diff": c_bshap,
        "E2c_ig_max_diff": c_ig,
        "E2c_R_F0": c_R0,
        "E2c_R_F1": c_R1,
        "E2c_H_F1": c_H1,
        "E2c_D_F1": c_D1,
    }

    pass_flags = {
        "E2a_allocation": checks["E2a_max_allocation_error"] <= args.tol_main,
        "E2a_transfer": checks["E2a_max_transfer_error"] <= args.tol_main,
        "E2a_conservation": checks["E2a_max_pot_conservation_error"] <= args.tol_main,
        "E2a_corners": checks["E2a_max_corner_error"] <= args.tol_corner,

        "E2b_R": checks["E2b_max_R_error"] <= args.tol_main,
        "E2b_D_prediction": checks["E2b_max_D_error"] <= args.tol_main,
        "E2b_H": checks["E2b_max_H_error"] <= args.tol_main,
        "E2b_visible_D": checks["E2b_max_visible_D"] <= args.tol_main,

        "E2c_corners": checks["E2c_corner_game_max_diff"] <= args.tol_corner,
        "E2c_bshap": checks["E2c_bshap_max_diff"] <= args.tol_corner,
        "E2c_ig": checks["E2c_ig_max_diff"] <= args.tol_main,
        "E2c_R_F1": abs(checks["E2c_R_F1"] - 2.4) <= args.tol_main,
    }
    all_pass = bool(all(pass_flags.values()))

    summary = {
        "experiment": "E2 smooth realizability and observational indistinguishability",
        "seed": args.seed,
        "quadrature_order": args.quad_order,
        "E2a": {
            "ks": args.ks,
            "reps_per_k": args.reps,
            "n_problems": int(len(e2a)),
            "target_R_range": [args.R_min, args.R_max],
        },
        "E2b": {
            "cycle_lengths": args.ells,
            "exponents": args.ps,
            "n_models": int(len(e2b)),
        },
        "E2c": {
            "cycle_length": 6,
            "F0_exponent": 1,
            "F1_exponent": 9,
            "expected_R_F1": 2.4,
        },
        "tolerances": {
            "main": args.tol_main,
            "corner": args.tol_corner,
        },
        "checks": checks,
        "pass_flags": pass_flags,
        "all_pass": all_pass,
        "outputs": {
            "e2a_csv": str(out_dir / "e2a_results.csv"),
            "e2b_csv": str(out_dir / "e2b_results.csv"),
            "e2c_csv": str(out_dir / "e2c_summary.csv"),
            "e2a_figure_pdf": str(out_dir / "e2a_planted_vs_recovered.pdf"),
            "e2b_figure_pdf": str(out_dir / "e2b_positive_complementarity.pdf"),
            "e2c_figure_pdf": str(out_dir / "e2c_observational_indistinguishability.pdf"),
            "e2_combined_figure_pdf": str(out_dir / "e2_three_panel_paper.pdf"),
            "e2_combined_figure_png": str(out_dir / "e2_three_panel_paper.png"),
        },
    }

    with open(out_dir / "e2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 68)
    print("E2 — SMOOTH REALIZABILITY AND OBSERVATIONAL INDISTINGUISHABILITY")
    print("=" * 68)
    print(f"E2a problems                    : {len(e2a)}")
    print(f"E2a max allocation error        : {checks['E2a_max_allocation_error']:.3e}")
    print(f"E2a max transfer error          : {checks['E2a_max_transfer_error']:.3e}")
    print(f"E2a max corner error            : {checks['E2a_max_corner_error']:.3e}")
    print(f"E2a max conservation error      : {checks['E2a_max_pot_conservation_error']:.3e}")
    print(f"E2a negative-target fraction    : {checks['E2a_negative_target_fraction']:.3f}")
    print()
    print(f"E2b models                      : {len(e2b)}")
    print(f"E2b max R prediction error      : {checks['E2b_max_R_error']:.3e}")
    print(f"E2b max H prediction error      : {checks['E2b_max_H_error']:.3e}")
    print(f"E2b max visible D               : {checks['E2b_max_visible_D']:.3e}")
    print()
    print(f"E2c corner-game max diff        : {checks['E2c_corner_game_max_diff']:.3e}")
    print(f"E2c BShap max diff              : {checks['E2c_bshap_max_diff']:.3e}")
    print(f"E2c IG max diff                 : {checks['E2c_ig_max_diff']:.3e}")
    print(f"E2c R(F0)                       : {checks['E2c_R_F0']:.12g}")
    print(f"E2c R(F1)                       : {checks['E2c_R_F1']:.12g}")
    print(f"E2c H(F1)                       : {checks['E2c_H_F1']:.12g}")
    print(f"E2c D(F1)                       : {checks['E2c_D_F1']:.3e}")
    print()
    print("Pre-registered checks:")
    for k, v in pass_flags.items():
        print(f"  {k}: {v}")
    print(f"  all_pass: {all_pass}")
    print()
    print(f"Outputs: {out_dir}")
    print("=" * 68)

    if not all_pass:
        raise RuntimeError(
            "E2 FAILED one or more pre-registered checks. "
            "Do not relax thresholds or reinterpret; inspect and debug."
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="E2 smooth realizability and observational indistinguishability"
    )
    p.add_argument("--out-dir", default="./e2_outputs")
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--R-min", type=float, default=0.05)
    p.add_argument("--R-max", type=float, default=1.0)
    p.add_argument("--ells", type=int, nargs="+", default=[3, 4, 6, 8])
    p.add_argument("--ps", type=int, nargs="+", default=[1, 2, 3, 5, 9, 20])
    p.add_argument("--quad-order", type=int, default=64)
    p.add_argument("--tol-main", type=float, default=1e-10)
    p.add_argument("--tol-corner", type=float, default=1e-12)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
