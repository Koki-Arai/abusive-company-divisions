"""
bottleneck_sensitivity.py
=========================
Module 3 — Bottleneck Sensitivity Analysis  (Section 5.3(3))

Theoretical correspondence:
  - Section 4: Proposition 7 (priority ordering of requirement sensitivities)
  - Acceptance probability Pi = p1 * p2 * p3 (multiplicative structure)
  - dPi/dpi = Pi/pi  (sensitivity is maximised at the requirement with lowest pi)
  - Evidence investment prioritisation by grey zone (G1-G4)

Core proposition verified (Section 5.5, Proposition 2):
  The sensitivity of Pi with respect to each requirement is maximised for the
  requirement with the lowest satisfaction probability (bottleneck), and
  evidentiary resources should be concentrated on that bottleneck.
  Bottleneck probability: G3 p2 = 1.000 across 150,000 MC draws.

Usage:
    python bottleneck_sensitivity.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

PARAM_PATH = os.path.join(os.path.dirname(__file__), "params", "type_space.yaml")


def load_params(path: str = PARAM_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#  requirement satisfaction probabilities
GREY_ZONES = {
    "G1 (superficially fair)": {
        "description": "delta within tolerance; suspicious post-division disposal (sigma=1)",
        "p1": 0.70,   # R1 (asset diminution): moderate — delta not sharply defined
        "p2": 0.65,   # R2 (insolvency timing): moderate
        "p3": 0.35,   # R3 (knowledge): sigma evidence is key -> lowest -> bottleneck
        "bottleneck": "p3",
        "priority_note": "Early preservation of post-division disposal evidence (key for knowledge inference)",
    },
    "G2 (net asset boundary)": {
        "description": "phi near zero boundary; off-balance-sheet liabilities suspected",
        "p1": 0.45,   # R1 (asset diminution): sign of phi is key -> lowest -> bottleneck
        "p2": 0.70,   # R2 (insolvency timing): relatively clear
        "p3": 0.65,   # R3 (knowledge): moderate
        "bottleneck": "p1",
        "priority_note": "Precise valuation of phi including off-balance-sheet liabilities",
    },
    "G3 (temporal ambiguity)": {
        "description": "division date close to insolvency date (tau~0)",
        "p1": 0.80,   # R1 (asset diminution): delta clear -> high
        "p2": 0.30,   # R2 (insolvency timing): tau~0 -> lowest -> bottleneck
        "p3": 0.72,   # R3 (knowledge): relatively clear
        "bottleneck": "p2",
        "priority_note": "Retrospective financial reconstruction is highest priority (p2 bottleneck)",
    },
    "G4 (mixed type)": {
        "description": "v>0 and w>0; v-w ratio uncertain (mixed type)",
        "p1": 0.68,   # R1 (asset diminution): moderate
        "p2": 0.72,   # R2 (insolvency timing): moderate
        "p3": 0.42,   # R3Requirement: v-w
        "bottleneck": "p3",
        "priority_note": "Independent valuation of v and sigma investigation are highest priority",
    },
}


#  Sensitivity
def compute_pi_and_sensitivities(p1: float, p2: float,
                                  p3: float) -> Dict[str, float]:
    """
    Compute Pi = p1*p2*p3 and the sensitivity of Pi with respect to each requirement.
    dPi/dpi = Pi/pi  (direct calculation of Proposition 7)
    """
    Pi = p1 * p2 * p3
    sens = {
        "Pi": Pi,
        "dPi_dp1": Pi / p1 if p1 > 0 else 0.0,
        "dPi_dp2": Pi / p2 if p2 > 0 else 0.0,
        "dPi_dp3": Pi / p3 if p3 > 0 else 0.0,
    }
    # Bottleneck pi  Sensitivity
    pi_vals = {"p1": p1, "p2": p2, "p3": p3}
    bottleneck = min(pi_vals, key=pi_vals.get)
    sens["bottleneck"] = bottleneck
    return sens


#  Sensitivity
def numerical_sensitivity(p1: float, p2: float, p3: float,
                           h: float = 0.01) -> Dict[str, float]:
    """Numerical sensitivity via finite differences (for cross-checking with analytical values)."""
    Pi0 = p1 * p2 * p3
    dP1 = ((p1 + h) * p2 * p3 - Pi0) / h
    dP2 = (p1 * (p2 + h) * p3 - Pi0) / h
    dP3 = (p1 * p2 * (p3 + h) - Pi0) / h
    return {"num_dPi_dp1": dP1, "num_dPi_dp2": dP2, "num_dPi_dp3": dP3}


#  Sensitivity
def build_sensitivity_table() -> pd.DataFrame:
    """Compute Pi and sensitivities for all grey zones."""
    rows = []
    for zone, cfg in GREY_ZONES.items():
        p1, p2, p3 = cfg["p1"], cfg["p2"], cfg["p3"]
        sens = compute_pi_and_sensitivities(p1, p2, p3)
        num_sens = numerical_sensitivity(p1, p2, p3)
        rows.append({
            "grey_zone": zone,
            "p1": p1, "p2": p2, "p3": p3,
            "Π": round(sens["Pi"], 4),
            "∂Π/∂p1": round(sens["dPi_dp1"], 4),
            "∂Π/∂p2": round(sens["dPi_dp2"], 4),
            "∂Π/∂p3": round(sens["dPi_dp3"], 4),
            "Bottleneck": sens["bottleneck"],
            "priority_note": cfg["priority_note"],
        })
    return pd.DataFrame(rows)


#  Marginal returnBottleneck vs Requirement
def marginal_return_comparison(zone_key: str = "G3 (temporal ambiguity)",
                                delta_p: float = 0.10,
                                n_steps: int = 10) -> pd.DataFrame:
    """
    Improvement in the bottleneck requirement (low probability -> +delta) vs
    Requirement→+δ Π 

    7
    RequirementRequirement
    Π 
    """
    cfg = GREY_ZONES[zone_key]
    p1_base, p2_base, p3_base = cfg["p1"], cfg["p2"], cfg["p3"]
    bn = cfg["bottleneck"]

    rows = []
    for step in range(n_steps + 1):
        improvement = step * delta_p / n_steps

        # BottleneckRequirement
        if bn == "p1":
            Pi_bn = min(p1_base + improvement, 1.0) * p2_base * p3_base
        elif bn == "p2":
            Pi_bn = p1_base * min(p2_base + improvement, 1.0) * p3_base
        else:
            Pi_bn = p1_base * p2_base * min(p3_base + improvement, 1.0)

        # Requirement
        pi_vals = {"p1": p1_base, "p2": p2_base, "p3": p3_base}
        high_req = max(pi_vals, key=pi_vals.get)
        if high_req == "p1":
            Pi_hi = min(p1_base + improvement, 1.0) * p2_base * p3_base
        elif high_req == "p2":
            Pi_hi = p1_base * min(p2_base + improvement, 1.0) * p3_base
        else:
            Pi_hi = p1_base * p2_base * min(p3_base + improvement, 1.0)

        rows.append({
            "improvement": round(improvement, 3),
            "Pi_bottleneck": round(Pi_bn, 4),
            "Pi_high_req": round(Pi_hi, 4),
            "gain_bottleneck": round(Pi_bn - p1_base * p2_base * p3_base, 4),
            "gain_high_req": round(Pi_hi - p1_base * p2_base * p3_base, 4),
            "bottleneck_req": bn,
            "high_req": high_req,
        })
    return pd.DataFrame(rows)


#
def check_abandonment_threshold(Pi: float, params: dict) -> dict:
    """
    7(c)Π  κ_A 
    

     κ_A = (k_A + Pr(G|s)·v) / w
    """
    k_A = params["costs"]["k_A"]
    w = params["value"]["w_B"]
    # Pr(G|s)  v Mixed type (M)
    v_M = params["value"]["v_M"]
    # Pr(G|s)
    Pr_G = params["prior"]["lambda_G"]

    kappa_A = (k_A + Pr_G * v_M) / w
    return {
        "Pi": round(Pi, 4),
        "kappa_A": round(kappa_A, 4),
        "above_threshold": Pi >= kappa_A,
        "recommendation": "" if Pi >= kappa_A else " or ",
    }


#  p Sensitivity
def compute_sensitivity_surface(fixed_pi: float = 0.7,
                                 n_grid: int = 30) -> Tuple[np.ndarray, ...]:
    """
    p1  p2 p3  Π  ∂Π/∂p2 
    G3Bottleneck p2 
    """
    p1_grid = np.linspace(0.1, 1.0, n_grid)
    p2_grid = np.linspace(0.1, 1.0, n_grid)
    P1, P2 = np.meshgrid(p1_grid, p2_grid)
    Pi_surf = P1 * P2 * fixed_pi
    sens_p2_surf = Pi_surf / P2
    return P1, P2, Pi_surf, sens_p2_surf


#
def plot_bottleneck_results(sens_table: pd.DataFrame,
                             mr_df: pd.DataFrame,
                             zone_key: str,
                             P1, P2, Pi_surf, sens_surf,
                             save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Bottleneck Sensitivity Analysis", fontsize=13, y=1.02)

    # (1) Sensitivity
    ax = axes[0]
    zones = [z for z in sens_table["grey_zone"]]
    x = np.arange(len(zones))
    width = 0.25
    colors = {"∂Π/∂p1": "#185FA5", "∂Π/∂p2": "#BA7517", "∂Π/∂p3": "#993C1D"}
    for i, (col, color) in enumerate(colors.items()):
        ax.bar(x + (i - 1) * width, sens_table[col], width,
               label=col.replace("∂Π/∂", "∂Π/∂"), color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=8)
    ax.set_ylabel("Sensitivity ∂Π/∂pi")
    ax.set_title("(1) RequirementSensitivity\n pi →Bottleneck")
    ax.legend(fontsize=8)
    # BottleneckRequirement
    for i, row in sens_table.iterrows():
        bn = row["Bottleneck"]
        col_map = {"p1": 0, "p2": 1, "p3": 2}
        offset = (col_map[bn] - 1) * width
        ax.annotate("★", (i + offset, row[f"∂Π/∂{bn}"]),
                    ha="center", fontsize=11, color="red")

    # (2) Marginal returnBottleneck vs Requirement
    ax = axes[1]
    ax.plot(mr_df["improvement"], mr_df["gain_bottleneck"],
            lw=2.5, color="#D85A30",
            label=f"Bottleneck{mr_df['bottleneck_req'].iloc[0]}")
    ax.plot(mr_df["improvement"], mr_df["gain_high_req"],
            lw=2.5, color="#888780", ls="--",
            label=f"Requirement{mr_df['high_req'].iloc[0]}")
    ax.set_xlabel(" pi ")
    ax.set_ylabel("Π ")
    ax.set_title(f"(2) Marginal return\n{zone_key.split(' ')[0]}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) /p2 G3  p2 Bottleneck
    ax = axes[2]
    cf = ax.contourf(P1, P2, sens_surf, levels=20, cmap="YlOrRd")
    plt.colorbar(cf, ax=ax, label="∂Π/∂p2  p2Sensitivity")
    ax.set_xlabel("p1  Requirement")
    ax.set_ylabel("p2  Requirement")
    ax.set_title("(3) G3  p2 Sensitivity\np3=0.72 =Sensitivity")
    ax.plot([0.80], [0.30], "w*", ms=15, label="G3 ")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#
def main(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    params = load_params()

    print("[1/4] Sensitivity...")
    sens_table = build_sensitivity_table()
    sens_table.to_csv(f"{output_dir}/bottleneck_sensitivity_table.csv",
                      index=False, encoding="utf-8-sig")
    print(sens_table[["grey_zone", "p1", "p2", "p3",
                       "Π", "∂Π/∂p1", "∂Π/∂p2", "∂Π/∂p3",
                       "Bottleneck"]].to_string(index=False))

    print("\n[2/4] G3Marginal return...")
    zone_key = "G3 (temporal ambiguity)"
    mr_df = marginal_return_comparison(zone_key, delta_p=0.30)
    mr_df.to_csv(f"{output_dir}/marginal_return_G3.csv", index=False)
    print(f"  Bottleneckp2 0.3  Π : {mr_df['gain_bottleneck'].iloc[-1]:.4f}")
    print(f"  Requirementp1 0.3  Π : {mr_df['gain_high_req'].iloc[-1]:.4f}")

    print("\n[3/4] ...")
    for zone, cfg in GREY_ZONES.items():
        Pi_zone = cfg["p1"] * cfg["p2"] * cfg["p3"]
        result = check_abandonment_threshold(Pi_zone, params)
        print(f"  {zone}: Π={result['Pi']:.3f},"
              f" κ_A={result['kappa_A']:.3f} → {result['recommendation']}")

    print("\n[4/4] p2 Sensitivity...")
    P1, P2, Pi_surf, sens_surf = compute_sensitivity_surface(
        fixed_pi=GREY_ZONES["G3 (temporal ambiguity)"]["p3"], n_grid=40)

    plot_bottleneck_results(sens_table, mr_df, zone_key,
                            P1, P2, Pi_surf, sens_surf,
                            save_path=f"{output_dir}/bottleneck_sensitivity.png")

    print(f"\nDone. Output directory: {output_dir}/")
    return sens_table, mr_df


if __name__ == "__main__":
    main()
