# False Consensus in Feature Attribution

Anonymous reproducibility package for the experiments accompanying **“False
Consensus in Feature Attribution: When Baseline Shapley and Integrated
Gradients Agree for the Wrong Reason.”**

The repository contains the code used by the reported experiments together
with compact frozen, non-checkpoint artifacts under `artifacts/`. The tabular
and vision protocols freeze every split, baseline, audit endpoint, model seed,
threshold grid, and numerical certification rule before the fitted-model
audits are run.

## Contents

| Study | Purpose | Entry points |
|---|---|---|
| E1 | Structural cycle-space stress test | `ICLR/e1_structural_stress_test.py` |
| E2 | Smooth realizability and observational indistinguishability | `ICLR/e2_smooth_realizability.py` |
| E3a | California Housing protocol, null controls, MLP, CrossNet, and paired analyses | `ICLR/e3_tabular_*.py` |
| E3b | Fashion-MNIST protocol, null controls, CNN, and appendix artifacts | `ICLR/e3_vision_*.py` |
| Theory figures | Ledger aggregation and forest/cycle schematics | `ICLR/fig_*.py` |

## Environment

The experiments were executed with Python 3.13 on CPU. Create an isolated
environment and install the declared dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`; on POSIX shells, use
`source .venv/bin/activate`.

## Reproduction order

Run every command from the repository root. E1 and E2 are lightweight. The
five-seed fitted-model stages, especially the complete Fashion-MNIST CNN
audits, are CPU-intensive.

```bash
# E1: structural stress test
python ICLR/e1_structural_stress_test.py

# E2: smooth realizability
python ICLR/e2_smooth_realizability.py

# E3a: frozen tabular protocol and controls
python ICLR/e3_tabular_protocol.py
python ICLR/e3_tabular_controls.py

# E3a: fitted tabular models and paired endpoint analyses
python ICLR/e3_tabular_mlp.py
python ICLR/e3_tabular_mlp_endpoint_stability.py
python ICLR/e3_tabular_mlp_threshold_surface.py
python ICLR/e3_tabular_crossnet.py
python ICLR/e3_tabular_cross_architecture.py

# E3a: paper figures from frozen outputs
python ICLR/e3_tabular_paper_figures.py

# E3b: frozen vision protocol and controls
python ICLR/e3_vision_protocol.py
python ICLR/e3_vision_controls.py

# E3b: fitted CNN and appendix artifacts
python ICLR/e3_vision_cnn.py
python ICLR/e3_vision_paper_artifacts.py

# Theory schematics
python ICLR/fig_ledger_aggregation.py
python ICLR/fig_forest_vs_cycle.py
```

The scripts create their output directories relative to the repository root.
Protocol and vision runners deliberately refuse to overwrite a non-empty
freeze directory.

## Experimental safeguards

- California Housing uses one fixed 70/15/15 split, an observed training-row
  baseline, 100 fixed test endpoints, five model seeds, and all 256 Boolean
  corners of the eight-feature game.
- Fashion-MNIST uses one fixed 51,000/9,000 train/validation split, the
  canonical test set, an observed training-image baseline, 100 class-balanced
  endpoints, eight fixed regions, five model seeds, and all 256 region mosaics.
- Additive and quadratic control families are mathematically required to have
  a null transfer ledger; all fitted-model audits reuse their certified audit
  machinery.
- Model and architecture selection uses validation predictive loss only. No
  attribution quantity affects fitting, endpoint selection, or acceptance.
- Adaptive quadrature, reconstruction, completeness, and conservation checks
  are enforced before an audit is marked certified.

Further command and output details are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The included frozen outputs, their release policy, and read-only regeneration
commands are documented in [`ARTIFACTS.md`](ARTIFACTS.md).

## Release verification

Before publishing or after modifying the package, run:

```bash
python tools/verify_release.py
python tools/verify_artifacts.py
```

The two verifiers check the experiment-script inventory, source hashes, Python
syntax, internal artifact manifests, hard-pass flags, audit row counts, and the
absence of identity or absolute local-path markers.

## Data

California Housing is obtained through scikit-learn. Fashion-MNIST is obtained
through torchvision and cached under `./data/`. Dataset files and trained
checkpoints are intentionally excluded from version control.

## License

Code is released under the MIT License. Dataset terms remain governed by their
respective providers.
