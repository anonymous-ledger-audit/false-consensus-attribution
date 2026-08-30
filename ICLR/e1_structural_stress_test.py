#!/usr/bin/env python3
"""
E1 — Controlled structural stress test for false consensus
===========================================================

This is the theory-only experiment.

For pairwise pots u={i,j}, column conservation means each pot has only one
free scalar transfer x_e. After fixing an orientation i -> j, the full
feature-level aggregation map is the oriented incidence matrix B:

    aggregate discrepancy = B x.

Thus ker(B) is exactly the cycle space. For a connected graph,

    dim ker(B) = |E| - |V| + 1 = beta,

where beta is the ordinary cycle rank.

The experiment:
  1. Fix d=12 features.
  2. Generate connected graphs with cycle ranks beta in {0,1,2,4,8}.
  3. Numerically compute nullity(B) for 100 seeded graphs per beta.
  4. For cyclic graphs, sample a random h in ker(B), normalize R(h)=1,
     and verify D(h)=0 and H(h)=R(h)=1.
  5. Visualize one aggregation fiber alpha h: the ledger displacement grows
     with alpha while the aggregate displacement stays zero.

Outputs
-------
e1_results.csv
e1_summary.json
e1_structural_stress_test.pdf
e1_structural_stress_test.png

Run
---
python e1_structural_stress_test.py

Optional
--------
python e1_structural_stress_test.py --d 12 --reps 100 --seed 20260828
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------

def random_tree_edges(d: int, rng: np.random.Generator):
    """
    Uniform random labeled tree via a Prüfer sequence.
    Returns undirected edges as sorted tuples (i,j), i<j.
    """
    if d == 2:
        return [(0, 1)]

    prufer = rng.integers(0, d, size=d - 2)
    degree = np.ones(d, dtype=int)
    for v in prufer:
        degree[v] += 1

    edges = []
    for v in prufer:
        leaf = int(np.where(degree == 1)[0][0])
        a, b = sorted((leaf, int(v)))
        edges.append((a, b))
        degree[leaf] -= 1
        degree[v] -= 1

    remaining = np.where(degree == 1)[0]
    a, b = sorted((int(remaining[0]), int(remaining[1])))
    edges.append((a, b))
    return edges


def graph_with_cycle_rank(d: int, beta: int, rng: np.random.Generator):
    """
    Start from a random spanning tree and add beta distinct non-tree edges.
    The graph remains connected and has cycle rank exactly beta.
    """
    tree = random_tree_edges(d, rng)
    edge_set = set(tree)

    all_edges = [(i, j) for i in range(d) for j in range(i + 1, d)]
    candidates = [e for e in all_edges if e not in edge_set]

    if beta > len(candidates):
        raise ValueError(
            f"beta={beta} is impossible for d={d}; max extra edges is {len(candidates)}."
        )

    if beta:
        chosen_idx = rng.choice(len(candidates), size=beta, replace=False)
        for k in np.atleast_1d(chosen_idx):
            edge_set.add(candidates[int(k)])

    return sorted(edge_set)


# ---------------------------------------------------------------------
# Aggregation operator
# ---------------------------------------------------------------------

def oriented_incidence_matrix(d: int, edges):
    """
    B has one column per pair pot e=(i,j), i<j.

    A scalar x_e represents the column-conservative pot transfer
        T_{i,e}=+x_e,
        T_{j,e}=-x_e.

    Therefore the aggregate feature discrepancy is B @ x.
    """
    B = np.zeros((d, len(edges)), dtype=float)
    for e, (i, j) in enumerate(edges):
        B[i, e] = 1.0
        B[j, e] = -1.0
    return B


def svd_nullspace(B: np.ndarray):
    """
    Numerically stable nullspace basis and rank.
    """
    U, s, Vh = np.linalg.svd(B, full_matrices=True)

    if s.size == 0:
        rank = 0
        tol = 0.0
    else:
        tol = max(B.shape) * np.finfo(float).eps * s[0]
        rank = int(np.sum(s > tol))

    null_basis = Vh[rank:].T
    return null_basis, rank, tol, s


# ---------------------------------------------------------------------
# Ledger metrics
# ---------------------------------------------------------------------

def R_of_x(x: np.ndarray) -> float:
    """
    For pair pots, each scalar x_e corresponds to a ledger column (+x_e,-x_e).
    Hence 0.5 * ||T_col||_1 = |x_e| and R = sum_e |x_e|.
    """
    return float(np.abs(x).sum())


def D_of_x(B: np.ndarray, x: np.ndarray) -> float:
    """
    Visible aggregate discrepancy: 0.5 ||B x||_1.
    """
    return 0.5 * float(np.abs(B @ x).sum())


def H_of_x(B: np.ndarray, x: np.ndarray) -> float:
    return R_of_x(x) - D_of_x(B, x)


def random_unit_circulation(
    null_basis: np.ndarray,
    rng: np.random.Generator,
):
    """
    Draw a generic random circulation from the numerical kernel and normalize R=1.
    """
    beta = null_basis.shape[1]
    if beta == 0:
        raise ValueError("No nonzero circulation exists in a zero-dimensional kernel.")

    coeff = rng.normal(size=beta)
    h = null_basis @ coeff

    r = R_of_x(h)
    if r < 1e-14:
        # astronomically unlikely, but deterministic fallback
        h = null_basis[:, 0].copy()
        r = R_of_x(h)

    return h / r


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def circle_positions(d: int):
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False) + np.pi / 2
    return np.column_stack([np.cos(angles), np.sin(angles)])


def draw_graph(ax, d, edges, title):
    pos = circle_positions(d)

    for i, j in edges:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            linewidth=1.0,
            alpha=0.75,
        )

    ax.scatter(pos[:, 0], pos[:, 1], s=45, zorder=3)

    # Label only a subset if d is large enough to clutter.
    if d <= 12:
        for i, (x, y) in enumerate(pos):
            ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=6)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = args.betas
    d = args.d

    max_beta = math.comb(d, 2) - (d - 1)
    for beta in betas:
        if beta < 0 or beta > max_beta:
            raise ValueError(
                f"beta={beta} invalid for d={d}; valid range is [0,{max_beta}]."
            )

    rng = np.random.default_rng(args.seed)

    rows = []
    representative = {}

    # --------------------------------------------------------------
    # Repeated structural test
    # --------------------------------------------------------------
    for beta in betas:
        for rep in range(args.reps):
            edges = graph_with_cycle_rank(d, beta, rng)
            B = oriented_incidence_matrix(d, edges)
            null_basis, rank, rank_tol, singular_values = svd_nullspace(B)

            m = len(edges)
            theoretical_beta = m - d + 1  # graph is connected by construction
            numerical_nullity = null_basis.shape[1]

            row = {
                "beta_requested": beta,
                "rep": rep,
                "d": d,
                "m_edges": m,
                "beta_formula": theoretical_beta,
                "rank_B": rank,
                "numerical_nullity": numerical_nullity,
                "rank_tolerance": rank_tol,
            }

            if numerical_nullity > 0:
                h = random_unit_circulation(null_basis, rng)
                agg = B @ h

                R = R_of_x(h)
                D = D_of_x(B, h)
                H = H_of_x(B, h)

                row.update({
                    "R_hidden": R,
                    "D_hidden": D,
                    "H_hidden": H,
                    "max_abs_Ah": float(np.max(np.abs(agg))),
                    "l2_Ah": float(np.linalg.norm(agg)),
                })
            else:
                row.update({
                    "R_hidden": 0.0,
                    "D_hidden": 0.0,
                    "H_hidden": 0.0,
                    "max_abs_Ah": 0.0,
                    "l2_Ah": 0.0,
                })

            rows.append(row)

            if rep == 0:
                representative[beta] = {
                    "edges": edges,
                    "B": B,
                    "null_basis": null_basis,
                }

    df = pd.DataFrame(rows)
    csv_path = out_dir / "e1_results.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------------
    # Pre-registered computational checks
    # --------------------------------------------------------------
    dim_fail = df["numerical_nullity"].to_numpy() != df["beta_formula"].to_numpy()

    cyclic = df["beta_formula"] > 0
    max_kernel_resid = (
        float(df.loc[cyclic, "max_abs_Ah"].max()) if cyclic.any() else 0.0
    )
    max_D = float(df.loc[cyclic, "D_hidden"].max()) if cyclic.any() else 0.0
    max_R_error = (
        float(np.max(np.abs(df.loc[cyclic, "R_hidden"].to_numpy() - 1.0)))
        if cyclic.any()
        else 0.0
    )
    max_H_error = (
        float(np.max(np.abs(df.loc[cyclic, "H_hidden"].to_numpy() - 1.0)))
        if cyclic.any()
        else 0.0
    )

    checks = {
        "dimension_law_all_pass": bool(not dim_fail.any()),
        "n_dimension_failures": int(dim_fail.sum()),
        "max_abs_Ah_over_cyclic_graphs": max_kernel_resid,
        "max_D_over_normalized_circulations": max_D,
        "max_abs_R_minus_1": max_R_error,
        "max_abs_H_minus_1": max_H_error,
        "numerical_tolerance": args.check_tol,
    }

    checks["all_pass"] = bool(
        checks["dimension_law_all_pass"]
        and max_kernel_resid <= args.check_tol
        and max_D <= args.check_tol
        and max_R_error <= args.check_tol
        and max_H_error <= args.check_tol
    )

    # --------------------------------------------------------------
    # Fiber example
    # --------------------------------------------------------------
    cyclic_betas = [b for b in betas if b > 0]
    fiber_beta = args.fiber_beta
    if fiber_beta not in representative or fiber_beta == 0:
        if not cyclic_betas:
            raise ValueError("Need at least one cyclic beta for the fiber panel.")
        fiber_beta = cyclic_betas[-1]

    Bf = representative[fiber_beta]["B"]
    Nf = representative[fiber_beta]["null_basis"]
    hf = random_unit_circulation(Nf, rng)  # R(h)=1

    alphas = np.array(args.alphas, dtype=float)
    fiber_rows = []
    for alpha in alphas:
        delta = alpha * hf
        fiber_rows.append({
            "alpha": alpha,
            "ledger_displacement_R": R_of_x(delta),
            "aggregate_displacement_D": D_of_x(Bf, delta),
            "max_abs_aggregate_margin": float(np.max(np.abs(Bf @ delta))),
        })

    fiber_df = pd.DataFrame(fiber_rows)
    fiber_csv = out_dir / "e1_fiber.csv"
    fiber_df.to_csv(fiber_csv, index=False)

    # --------------------------------------------------------------
    # Figure: 3 panels
    # --------------------------------------------------------------
    fig = plt.figure(figsize=(10.5, 3.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])

    # (a) Representative graph structures.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.set_title("(a) Structure", fontsize=10, pad=2)

    # Draw three mini-graphs manually inside the left panel.
    mini_betas = [0]
    if 1 in representative:
        mini_betas.append(1)
    elif cyclic_betas:
        mini_betas.append(cyclic_betas[0])
    multi = max(cyclic_betas) if cyclic_betas else 0
    if multi not in mini_betas:
        mini_betas.append(multi)

    # Display at most three representative edges.
    mini_betas = mini_betas[:3]

    boxes = []
    x0s = np.linspace(0.02, 0.68, len(mini_betas))
    for x0, beta in zip(x0s, mini_betas):
        iax = ax_a.inset_axes([x0, 0.18, 0.30, 0.68])
        draw_graph(
            iax,
            d,
            representative[beta]["edges"],
            rf"$\beta={beta}$",
        )
        boxes.append(iax)

    ax_a.text(
        0.5,
        0.03,
        "forest → cycles → more invisible degrees of freedom",
        ha="center",
        va="bottom",
        fontsize=7.5,
        transform=ax_a.transAxes,
    )

    # (b) Dimension law.
    ax_b = fig.add_subplot(gs[0, 1])
    # jitter only horizontally for visibility; y values remain exact integers
    jitter_rng = np.random.default_rng(args.seed + 1)
    xj = (
        df["beta_formula"].to_numpy(dtype=float)
        + jitter_rng.normal(0.0, 0.025, size=len(df))
    )
    ax_b.scatter(
        xj,
        df["numerical_nullity"].to_numpy(dtype=float),
        s=10,
        alpha=0.35,
    )
    lo = min(betas)
    hi = max(betas)
    ax_b.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax_b.set_xlabel(r"cycle rank $\beta$")
    ax_b.set_ylabel(r"computed $\dim\ker\mathsf{A}$")
    ax_b.set_title("(b) Exact dimension law", fontsize=10)
    ax_b.set_xticks(betas)
    ax_b.set_yticks(betas)

    # (c) One aggregation fiber.
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(
        fiber_df["alpha"],
        fiber_df["ledger_displacement_R"],
        marker="o",
        label=r"ledger displacement $R(\alpha h)$",
    )
    ax_c.plot(
        fiber_df["alpha"],
        fiber_df["aggregate_displacement_D"],
        marker="s",
        label=r"aggregate displacement $D(\alpha h)$",
    )
    ax_c.set_xlabel(r"$\alpha$")
    ax_c.set_ylabel("displacement")
    ax_c.set_title(
        rf"(c) Invisible fiber ($\beta={fiber_beta}$)",
        fontsize=10,
    )
    ax_c.legend(fontsize=6.8, frameon=False)

    fig.tight_layout()

    # --------------------------------------------------------------
    # Reduce the vertical extent of panels (b) and (c).
    # --------------------------------------------------------------
    def shrink_axis_height(ax, scale=0.72):
        pos = ax.get_position()
        new_h = pos.height * scale
        new_y = pos.y0 + (pos.height - new_h) / 2.0
        ax.set_position([pos.x0, new_y, pos.width, new_h])

    shrink_axis_height(ax_b, scale=0.72)
    shrink_axis_height(ax_c, scale=0.72)

    pdf_path = out_dir / "e1_structural_stress_test.pdf"
    png_path = out_dir / "e1_structural_stress_test.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    by_beta = []
    for beta, g in df.groupby("beta_formula", sort=True):
        by_beta.append({
            "beta": int(beta),
            "n_graphs": int(len(g)),
            "nullity_unique": sorted(
                int(x) for x in g["numerical_nullity"].unique().tolist()
            ),
            "max_D_hidden": float(g["D_hidden"].max()),
            "max_abs_Ah": float(g["max_abs_Ah"].max()),
        })

    summary = {
        "experiment": "E1 controlled structural stress test",
        "seed": args.seed,
        "d": d,
        "betas": betas,
        "reps_per_beta": args.reps,
        "n_graphs_total": int(len(df)),
        "fiber_beta": int(fiber_beta),
        "checks": checks,
        "by_beta": by_beta,
        "outputs": {
            "results_csv": str(csv_path),
            "fiber_csv": str(fiber_csv),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
    }

    summary_path = out_dir / "e1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n============================================================")
    print("E1 — CONTROLLED STRUCTURAL STRESS TEST")
    print("============================================================")
    print(f"d                  : {d}")
    print(f"betas              : {betas}")
    print(f"reps per beta      : {args.reps}")
    print(f"total graphs       : {len(df)}")
    print()
    print("Dimension law:")
    for row in by_beta:
        print(
            f"  beta={row['beta']:>2} | "
            f"computed nullity={row['nullity_unique']} | "
            f"max D(hidden)={row['max_D_hidden']:.3e} | "
            f"max |A h|={row['max_abs_Ah']:.3e}"
        )
    print()
    print("Pre-registered checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print()
    print(f"results : {csv_path}")
    print(f"fiber   : {fiber_csv}")
    print(f"figure  : {pdf_path}")
    print(f"summary : {summary_path}")
    print("============================================================")

    if not checks["all_pass"]:
        raise RuntimeError(
            "E1 FAILED one or more pre-registered computational checks. "
            "Do not reinterpret; inspect the outputs and debug."
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="E1 structural stress test: forests vs cycles"
    )
    p.add_argument("--d", type=int, default=12)
    p.add_argument(
        "--betas",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8],
    )
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="./e1_structural_outputs")
    p.add_argument("--check-tol", type=float, default=1e-10)
    p.add_argument("--fiber-beta", type=int, default=4)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
