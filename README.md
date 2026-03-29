# Identifying Abusive Company Divisions: Replication Package

**Paper**: Identifying Abusive Company Divisions: A Multi-Dimensional Screening Model with Optimal Remedy Selection

**Authors**: [Author names — to be inserted in final version]

**Journal**: [Target journal — to be inserted]

---

## Overview

This repository contains the complete replication package for all simulation results reported in the paper. The package implements four Python modules corresponding to Sections 5.3–5.5 of the paper:

| Module | File | Section | Description |
|---|---|---|---|
| 1 | `montecarlo_calibration.py` | 5.3(1) | MC threshold calibration; verifies Proposition 6 |
| 2 | `signaling_equilibrium.py` | 5.3(2) | Signaling game; regulatory effect of disclosure standards |
| 3 | `bottleneck_sensitivity.py` | 5.3(3) | Bottleneck sensitivity by grey zone (G1–G4) |
| 4 | `asset_dissipation.py` | 5.3(4) | Asset dissipation dynamics; optimal filing timing |

All parameters are stored in a single YAML file (`params/type_space.yaml`) which serves as the single source of truth for replication. A complete parameter specification with three-way distinction between baseline calibration values, Module 1 optimised thresholds, and Module 3 abandonment thresholds is provided in the Online Appendix (OA.2–OA.8).

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py                    ← master script (runs all four modules)
├── montecarlo_calibration.py     ← Module 1
├── signaling_equilibrium.py      ← Module 2 (supplementary; see OA.5.2)
├── bottleneck_sensitivity.py     ← Module 3
├── asset_dissipation.py          ← Module 4
├── params/
│   └── type_space.yaml           ← all shared parameters
├── data/
│   └── case_coding_sheet.csv     ← anonymised judicial case codes
└── outputs/                      ← generated figures and CSV tables (git-ignored)
```

---

## Requirements

Python 3.10 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24`, `scipy>=1.11`, `pandas>=2.0`, `matplotlib>=3.7`, `PyYAML>=6.0`.

No proprietary software is required.

---

## Reproducing All Results

To reproduce all tables and figures exactly as reported in the paper:

```bash
python run_all.py --n_firms 5000 --grid_points 15
```

Results are saved to the `outputs/` directory as CSV files and PNG figures. **All random seeds are fixed** — Module 1 uses `numpy.random.default_rng(seed=42)`; Module 4 uses `numpy.random.seed(2025)` for the extended run. Expected runtime on a standard laptop: 3–8 minutes.

### Extended robustness check (Section 6.8)

The extended robustness analysis (3 scenarios × 50,000 Monte Carlo draws per grey zone) is embedded in `bottleneck_sensitivity.py` and runs automatically as part of Module 3.

### Quick test run

```bash
python run_all.py --n_firms 500 --grid_points 5
```

Completes in ~15 seconds and produces all output files at reduced precision.

---

## Parameter Configuration

All parameters are stored in `params/type_space.yaml`. **Do not edit source code to change parameters** — modify the YAML file only.

The parameter file covers:
- Prior probabilities (λ_G, λ_M, λ_B)
- Type-conditional signal distributions (δ, ρ, φ, σ)
- Baseline remedy activation thresholds (κ_P, κ_A, κ_F)
- Procedural costs and error weights
- Asset dissipation process parameters

> **Important distinction** (see OA.4 of the Appendix):
> - **Baseline calibration thresholds** (κ_P=0.25, κ_A=0.50, κ_F=0.75): stored in the YAML file; used as optimizer starting points.
> - **Module 1 optimised thresholds** (κ_P\*=0.05, κ_A\*=0.11, κ_F\*=0.18): outputs of the Module 1 grid search; not stored in the YAML.
> - **Module 3 abandonment threshold** (κ_A^abn ≈ 0.45): computed from the cost-benefit formula (k_A + Pr(G|s)·v)/w; not stored in the YAML.

---

## Module Notes

### Module 2 — Signaling Equilibrium (supplementary)

Module 2 is a supplementary illustration of the regulatory implications of the minimum disclosure standard ρ̄ for the sustainability of the separating equilibrium. The analytical result — c\* = w_B/(2·ρ̄²) ≈ 102 m yen at baseline — is confirmed numerically with the search range c_B ∈ [0.1, 200] m yen (200 grid points). Because the signaling game's model closure is less complete than Modules 1, 3, and 4, results from this module should be treated as directional illustrations. Full specification is in OA.5.2 of the Online Appendix.

### Module 4 — Seed Note

The base run uses `n_paths=1500, seed=42`. The extended run reported in Sections 6.7–6.8 uses `n_paths=10000, seed=2025`. Both are reproduced by `run_all.py`. Minor floating-point differences may arise with alternative RNG implementations; see OA.5.4 for details.

---

## Data

`data/case_coding_sheet.csv` contains the anonymised signal values assigned to each judicial case in the calibration dataset. Case identifiers are anonymised in accordance with the TKC Legal Information Database terms of use. The coding rubric for ρ (disclosure adequacy score) is described in Section 6.3 of the paper and OA.3 of the Online Appendix.

---

## Output Files

| File | Module | Contents |
|---|---|---|
| `firms_simulated.csv` | 1 | Simulated firm characteristics and posteriors |
| `threshold_calibration.csv` | 1 | Welfare loss across threshold grid |
| `proposition6_verification.csv` | 1 | Intervention rate by ρ bin (φ<0, δ>ε subset) |
| `montecarlo_calibration.png` | 1 | Three-panel figure: Λ(s) distribution, heatmap, Proposition 6 |
| `signaling_sweep.csv` | 2 | Deviation gain by c_B |
| `regulation_effect.csv` | 2 | c\* by ρ̄ (regulatory effect table) |
| `signaling_equilibrium.png` | 2 | Three-panel figure: cost functions, equilibrium, regulation effect |
| `bottleneck_sensitivity_table.csv` | 3 | Table 6.1: Π and ∂Π/∂pᵢ by grey zone |
| `marginal_return_G3.csv` | 3 | G3 marginal return comparison (p2 vs p1) |
| `bottleneck_sensitivity.png` | 3 | Two-panel figure: sensitivities, marginal returns |
| `sigma_dual_effect.csv` | 4 | Dual-channel summary (Bayesian + dissipation) |
| `delay_cost_sigma1.csv` | 4 | Filing delay cost analysis |
| `asset_dissipation.png` | 4 | Three-panel figure: asset paths, threshold, recovery |

---

## Citation

If you use this code, please cite:

```bibtex
@article{arai2025abusive,
  title   = {Identifying Abusive Company Divisions: A Multi-Dimensional Screening Model with Optimal Remedy Selection},
  author  = {[Author(s)]},
  journal = {[Journal]},
  year    = {2025},
  note    = {Replication code: https://github.com/[repository]}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

[Contact information — to be inserted in final version]
