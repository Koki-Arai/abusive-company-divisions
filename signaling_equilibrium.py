"""
signaling_equilibrium.py
========================
Module 2 — Signaling Equilibrium Simulation  (Section 5.3(2) / Supplementary)

Theoretical correspondence:
  - Section 3: Proposition 1 (existence of separating equilibrium)
  - Spence (1973)-type signaling model
  - Single crossing property: numerical verification
  - Effect of disclosure regulation stringency on equilibrium sustainability

Analytical result: c* = w_B / (2 * rho_bar^2) ~ 102 m yen at baseline (rho_bar=0.70)
  Raising rho_bar to 0.80 reduces c* to ~78 m yen; to 0.90 reduces c* to ~62 m yen.

Note: This module is supplementary to Modules 1, 3, 4.
  Full specification in Online Appendix OA.5.2.

Usage:
    python signaling_equilibrium.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional

PARAM_PATH = os.path.join(os.path.dirname(__file__), "params", "type_space.yaml")


#
@dataclass
class SignalingParams:
    """Parameters for the signaling game."""
    # fraudulent transfer gain
    w_B: float = 100.0       # Abusive type (B)fraudulent transfer amount (m yen)
    # remedy procedural costs
    k_A: float = 12.0        # avoidance rights exercise cost (m yen)
    # legal authority threshold rho_bar for separating equilibrium
    rho_bar: float = 0.70    # minimum disclosure level for legitimate type (G) classification
    rho_tilde: float = 0.40  # threshold for value-compensation remedy
    # disclosure cost function parameters
    c_G_base: float = 0.5    # true disclosure cost for legitimate type G (m yen)
    # simulation settings
    n_grid: int = 400        #
    c_B_grid_max: float = 200.0  # Abusive type (B) w_B/(2_bar)102


@dataclass
class EquilibriumResult:
    """Equilibrium computation results."""
    separating: bool           # Separating equilibrium
    c_star: float              # Critical costSeparating equilibrium
    rho_star_G: float          # Legitimate type (G)
    rho_star_B: float          # Abusive type (B)
    single_crossing: bool      #
    welfare_separating: float  # Separating equilibrium
    welfare_pooling: float     #


#
def cost_G(rho: np.ndarray, c_G_base: float) -> np.ndarray:
    """
    Disclosure cost function for legitimate type G.
    Truth-telling is cheap: C_G(rho) = c_G_base * rho^0.5
    """
    return c_G_base * np.sqrt(np.clip(rho, 0, 1))


def cost_B(rho: np.ndarray, c_B: float) -> np.ndarray:
    """
    Disclosure cost function for abusive type B.
    Steep because concealing fraud is costly: C_B(rho) = c_B * rho^2
    Single crossing condition requires dC_B/drho > dC_G/drho (B has steeper marginal cost)
    """
    return c_B * np.power(np.clip(rho, 0, 1), 2)


def marginal_cost_G(rho: np.ndarray, c_G_base: float) -> np.ndarray:
    eps = 1e-9
    return c_G_base / (2 * np.sqrt(np.clip(rho, eps, 1)))


def marginal_cost_B(rho: np.ndarray, c_B: float) -> np.ndarray:
    return 2 * c_B * np.clip(rho, 0, 1)


#
def payoff_G(rho: float, rho_bar: float, k_A: float, c_G_base: float) -> float:
    """
    Payoff for legitimate type G: intervention-avoidance value minus disclosure cost.
    rho >= rho_bar -> no intervention (empty set), so intervention cost = 0
    rho < rho_bar -> misclassification risk incurs expected cost k_A
    """
    if rho >= rho_bar:
        return -cost_G(np.array([rho]), c_G_base)[0]
    else:
        misclass_prob = 1 - rho / rho_bar
        return -cost_G(np.array([rho]), c_G_base)[0] - misclass_prob * k_A


def payoff_B(rho: float, rho_bar: float, w_B: float,
             k_A: float, c_B: float) -> float:
    """
    Abusive type (B) - Disclosure cost - 
    ρ >= ρ_bar →  w_B 
    ρ < rho_bar → 
    """
    if rho >= rho_bar:
        return w_B - cost_B(np.array([rho]), c_B)[0]
    else:
        detect_prob = 1 - rho / rho_bar
        expected_gain = w_B * (1 - detect_prob)
        return expected_gain - cost_B(np.array([rho]), c_B)[0]


#
def check_single_crossing(rho_grid: np.ndarray,
                           c_G_base: float, c_B: float) -> bool:
    """
    SCP: ∂C_B/∂ρ > ∂C_G/∂ρ for all ρ ∈ (0,1)
    →  ρ  B  > G 
    """
    mc_G = marginal_cost_G(rho_grid, c_G_base)
    mc_B = marginal_cost_B(rho_grid, c_B)
    return bool(np.all(mc_B > mc_G))


#
def compute_equilibrium(sp: SignalingParams, c_B: float) -> EquilibriumResult:
    """
     c_B Separating equilibrium

    Separating equilibrium
    (i)  G  ρ_bar = ρ*_G = ρ_bar
    (ii) B  ρ_bar Deviation gain ≤ 0

    B Deviation gainρ_bar  - 
    Separating equilibrium
    """
    rho_grid = np.linspace(0.001, 0.999, sp.n_grid)
    scp = check_single_crossing(rho_grid[1:], sp.c_G_base, c_B)

    # G :  = _barDisclosure cost
    rho_star_G = sp.rho_bar

    # B : Disclosure level
    payoffs_B = np.array([
        payoff_B(r, sp.rho_bar, sp.w_B, sp.k_A, c_B) for r in rho_grid
    ])
    idx_B = np.argmax(payoffs_B)
    rho_star_B = rho_grid[idx_B]
    max_payoff_B_no_mimic = payoffs_B[idx_B]

    # B  _bar
    mimic_payoff_B = payoff_B(sp.rho_bar, sp.rho_bar, sp.w_B, sp.k_A, c_B)
    deviation_gain = mimic_payoff_B - max_payoff_B_no_mimic

    separating = deviation_gain <= 0

    #
    if separating:
        # Separating equilibriumG  minimalB
        welf_sep = -cost_G(np.array([rho_star_G]), sp.c_G_base)[0]
    else:
        # B  w_B
        welf_sep = -sp.w_B  #

    welf_pool = -sp.w_B * 0.5  #

    return EquilibriumResult(
        separating=separating,
        c_star=0.0,  #  c*
        rho_star_G=rho_star_G,
        rho_star_B=rho_star_B,
        single_crossing=scp,
        welfare_separating=welf_sep,
        welfare_pooling=welf_pool,
    )


#  Critical cost c*
def find_critical_cost(sp: SignalingParams,
                        n_search: int = 100) -> Tuple[float, pd.DataFrame]:
    """
    Separating equilibriumCritical cost c* 
    c_B > c* → Separating equilibrium
    c_B < c* → Separating equilibriumB 
    """
    c_B_grid = np.linspace(0.1, sp.c_B_grid_max, n_search)
    rows = []
    c_star = None

    for c_B in c_B_grid:
        res = compute_equilibrium(sp, c_B)
        rho_grid = np.linspace(0.001, 0.999, sp.n_grid)
        payoffs_B = np.array([
            payoff_B(r, sp.rho_bar, sp.w_B, sp.k_A, c_B) for r in rho_grid
        ])
        idx_B = np.argmax(payoffs_B)
        mimic_payoff = payoff_B(sp.rho_bar, sp.rho_bar, sp.w_B, sp.k_A, c_B)
        dev_gain = mimic_payoff - payoffs_B[idx_B]

        rows.append({
            "c_B": round(c_B, 3),
            "separating": res.separating,
            "deviation_gain": round(dev_gain, 4),
            "rho_star_B": round(res.rho_star_B, 3),
            "scp": res.single_crossing,
        })
        if c_star is None and res.separating:
            c_star = c_B

    df_res = pd.DataFrame(rows)
    return c_star if c_star else sp.c_B_grid_max, df_res


#  Regulation stringency
def analyze_regulation_effect(base_sp: SignalingParams,
                               rho_min_values: np.ndarray,
                               n_search: int = 50) -> pd.DataFrame:
    """
    Disclosure level ρ_bar Critical cost c* 
    ρ_bar →  c_B 
    → Separating equilibrium
    """
    rows = []
    for rho_min in rho_min_values:
        sp = SignalingParams(
            w_B=base_sp.w_B,
            k_A=base_sp.k_A,
            rho_bar=rho_min,
            rho_tilde=base_sp.rho_tilde,
            c_G_base=base_sp.c_G_base,
            c_B_grid_max=base_sp.c_B_grid_max,
            n_grid=base_sp.n_grid,
        )
        c_star, _ = find_critical_cost(sp, n_search)
        rows.append({"rho_bar": round(rho_min, 3), "c_star": round(c_star, 3)})
    return pd.DataFrame(rows)


#
def plot_signaling_results(sp: SignalingParams,
                           sweep_df: pd.DataFrame,
                           reg_df: pd.DataFrame,
                           c_star: float,
                           save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Signaling Equilibrium Analysis", fontsize=13, y=1.01)

    rho_grid = np.linspace(0.001, 0.999, 200)

    # (1)
    ax = axes[0]
    c_B_lo = sp.c_B_grid_max * 0.2   # Separating equilibrium
    c_B_hi = c_star * 1.5 if c_star else sp.c_B_grid_max * 0.8  # Separating equilibrium
    ax.plot(rho_grid, cost_G(rho_grid, sp.c_G_base),
            lw=2, color="#1D9E75", label="C_G(ρ)  Legitimate type (G)")
    ax.plot(rho_grid, cost_B(rho_grid, c_B_lo),
            lw=2, color="#D85A30", ls="--", label=f"C_B() ")
    ax.plot(rho_grid, cost_B(rho_grid, c_B_hi),
            lw=2, color="#3B6D11", ls="-.", label=f"C_B() Separating equilibrium")
    ax.axvline(sp.rho_bar, color="gray", lw=1, ls=":",
               label=f"ρ_bar={sp.rho_bar}")
    ax.set_xlabel("ρ  ")
    ax.set_ylabel("Disclosure costm yen")
    ax.set_title("(1) ")
    ax.legend(fontsize=7)

    # (2) c_B Deviation gaindeviation gain
    ax = axes[1]
    ax.plot(sweep_df["c_B"], sweep_df["deviation_gain"],
            lw=2, color="#378ADD")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    if c_star:
        ax.axvline(c_star, color="red", lw=1.5, ls="--",
                   label=f"c* = {c_star:.2f}Critical cost")
    ax.fill_between(sweep_df["c_B"],
                    sweep_df["deviation_gain"], 0,
                    where=sweep_df["deviation_gain"] > 0,
                    alpha=0.2, color="red", label="")
    ax.fill_between(sweep_df["c_B"],
                    sweep_df["deviation_gain"], 0,
                    where=sweep_df["deviation_gain"] <= 0,
                    alpha=0.2, color="green", label="Separating equilibrium")
    ax.set_xlabel("c_B  Abusive type (B)Disclosure cost")
    ax.set_ylabel("B Deviation gain")
    ax.set_title(f"(2) Separating equilibrium\nDeviation gain ≤ 0 → ")
    ax.legend(fontsize=8)

    # (3) Regulation stringency_barCritical cost c*
    ax = axes[2]
    ax.plot(reg_df["rho_bar"], reg_df["c_star"],
            lw=2, color="#533AB7", marker="o", ms=5)
    ax.set_xlabel("ρ_bar  Minimum disclosure levelRegulation stringency")
    ax.set_ylabel("c*  Critical costm yen")
    ax.set_title("(3) Separating equilibrium\nρ_bar↑ → c* ↓")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#
def main(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    params = load_params()
    costs = params["costs"]
    value = params["value"]

    sp = SignalingParams(
        w_B=value["w_B"],
        k_A=costs["k_A"],
        rho_bar=0.70,
        rho_tilde=0.40,
        c_G_base=0.5,
        c_B_grid_max=200.0,  # Extended to encompass analytical c* = w_B/(2·ρ_bar²) ≈ 102 m yen
        n_grid=400,
    )

    print("[1/3] Critical cost c* ...")
    c_star, sweep_df = find_critical_cost(sp, n_search=200)  # Finer grid for accurate c* location
    sweep_df.to_csv(f"{output_dir}/signaling_sweep.csv", index=False)
    print(f"  Critical cost c* ≈ {c_star:.2f} m yen")
    print(f"  c_B > {c_star:.2f} → Separating equilibrium")
    print(f"  c_B < {c_star:.2f} → Abusive type (B)")

    print("[2/3] Regulation stringencyρ_bar...")
    rho_min_values = np.linspace(0.30, 0.90, 13)
    reg_df = analyze_regulation_effect(sp, rho_min_values, n_search=50)
    reg_df.to_csv(f"{output_dir}/regulation_effect.csv", index=False)
    print(reg_df.to_string(index=False))

    print("[3/3] generating figures...")
    plot_signaling_results(sp, sweep_df, reg_df, c_star,
                           save_path=f"{output_dir}/signaling_equilibrium.png")

    print(f"\nDone. Output directory: {output_dir}/")
    return c_star, sweep_df, reg_df


def load_params(path: str = PARAM_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
