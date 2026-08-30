# Reproducibility guide

All paths below are relative to the repository root. The default arguments are
the frozen experimental contract; changing them defines a different run.

## Expected environment

- Python 3.13
- CPU execution is supported throughout
- Internet access is required only when scikit-learn or torchvision first
  downloads a dataset
- Sufficient disk space is required for Fashion-MNIST, model checkpoints, and
  figure outputs

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Stage map

| Order | Command | Principal output |
|---:|---|---|
| 1 | `python ICLR/e1_structural_stress_test.py` | `e1_structural_outputs/` |
| 2 | `python ICLR/e2_smooth_realizability.py` | `e2_outputs/` |
| 3 | `python ICLR/e3_tabular_protocol.py` | `e3_tabular_protocol/` |
| 4 | `python ICLR/e3_tabular_controls.py` | `e3_tabular_controls/` |
| 5 | `python ICLR/e3_tabular_mlp.py` | `e3_tabular_mlp/` |
| 6 | `python ICLR/e3_tabular_mlp_endpoint_stability.py` | files added to `e3_tabular_mlp/` |
| 7 | `python ICLR/e3_tabular_mlp_threshold_surface.py` | files added to `e3_tabular_mlp/` |
| 8 | `python ICLR/e3_tabular_crossnet.py` | `e3_tabular_crossnet/` |
| 9 | `python ICLR/e3_tabular_cross_architecture.py` | `e3_tabular_cross_architecture/` |
| 10 | `python ICLR/e3_tabular_paper_figures.py` | `ICLR/paper_figures/` |
| 11 | `python ICLR/e3_vision_protocol.py` | `e3_vision_protocol/` |
| 12 | `python ICLR/e3_vision_controls.py` | `e3_vision_controls/` |
| 13 | `python ICLR/e3_vision_cnn.py` | `e3_vision_cnn/` |
| 14 | `python ICLR/e3_vision_paper_artifacts.py` | `ICLR/paper_figures/` |

The two theory schematics are independent of the fitted experiments:

```bash
python ICLR/fig_ledger_aggregation.py
python ICLR/fig_forest_vs_cycle.py
```

## E1: structural stress test

The default run fixes 12 features, cycle ranks
`{0, 1, 2, 4, 8}`, 100 repetitions per rank, and seed `20260828`. It verifies
the incidence-kernel dimension law and constructs a normalized hidden
circulation on cyclic graphs.

## E2: smooth constructions

The default run uses seed `20260829`, pot sizes `{2, 3, 4, 5}`, 100 targets per
pot size, cycle lengths `{3, 4, 6, 8}`, and exponents
`{1, 2, 3, 5, 9, 20}`. Every numerical requirement is checked before the stage
passes.

## E3a: California Housing

The protocol freezes:

- all eight native numerical features;
- split seed `20260830` and audit seed `20260831`;
- an observed training-row baseline;
- 100 test endpoints, ten per target decile;
- model seeds `20260840` through `20260844`;
- complete 256-corner games and the straight baseline-to-endpoint path.

The additive and quadratic controls must pass before either fitted nonlinear
family runs. The MLP and CrossNet scripts import the same exhaustive audit
implementation and numerical certificates. The endpoint-stability, threshold,
and cross-architecture scripts consume frozen CSV outputs and perform no model
training.

## E3b: Fashion-MNIST

The protocol freezes:

- a stratified 51,000/9,000 split of the canonical training set with seed
  `20260910`;
- an observed training-image baseline;
- eight fixed 2-by-4 regions;
- 100 official-test endpoints, ten per class, with seed `20260911`;
- model seeds `20260920` through `20260924`;
- the true-class centered logit and all 256 Boolean region mosaics.

The vision controls and CNN train in float32. Validation-selected weights are
cast to float64 before scale computation and every attribution audit. The paper
artifact script is read-only: it verifies frozen manifests and creates figures
and LaTeX tables without importing PyTorch or recomputing attributions.

## Output integrity

Protocol, control, and fitted-model stages write SHA-256 manifests where
applicable. Keep those manifests with their corresponding artifacts. A failed
hard check leaves diagnostic outputs for inspection and must not be treated as
a completed run.

The repository-level source verifier is independent of the experiment checks:

```bash
python tools/verify_release.py
python tools/verify_artifacts.py
```

Frozen outputs are stored below `artifacts/`, leaving the default output paths
free for an independent rerun. See `ARTIFACTS.md` for commands that regenerate
paper figures directly from the included audit tables.
