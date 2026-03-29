"""
asset_dissipation.py
====================
Module 4 — Asset Dissipation Dynamics Simulation  (Section 5.3(4))

Theoretical correspondence:
  - Section 4: Proposition 6 (optimal conditions for value-compensation claim)
  - U3 (recovery uncertainty): dynamic modelling
  - A_N(t): stochastic process for recoverable assets of NewCo
  - A_N*(t) = (k + OC) / Pi(t): cost threshold (time-varying)
  - Companies Act Art. 764(7): direct claim barred after bankruptcy filing

Core proposition verified (Section 5.5, Proposition 3):
  Observing sigma=1 operates through two independent causal channels:
  (i)  Bayesian updating: Pr(B|s) rises from 0.200 to 0.876 (4.4x)
  (ii) Asset dissipation: mean feasible filing window shrinks from 24.98 to 8.74 months;
       expected recovery falls by 55.1%; F-remedy rate rises to 11.9%

Usage:
    python asset_dissipation.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

PARAM_PATH = os.path.join(os.path.dirname(__file__), "params", "type_space.yaml")


def load_params(path: str = PARAM_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#
@dataclass
class DissipationConfig:
    """Simulation configuration for asset dissipation process."""
    A_N_init: float = 80.0      # initial recoverable assets (m yen)
    drift_sigma0: float = 0.0   # monthly drift for sigma=0 (~random walk)
    drift_sigma1: float = -8.0  # monthly drift for sigma=1 (negative: dissipation)
    diffusion: float = 5.0      # diffusion coefficient (m yen / month^0.5)
    T_months: int = 24          # simulation horizon (months)
    k_cost: float = 10.0        # filing cost k (m yen)
    OC: float = 3.0             # opportunity cost (m yen)
    Pi_base: float = 0.35       # baseline acceptance probability Pi (pre-evidence)
    Pi_improved: float = 0.60   # improved acceptance probability Pi (post-evidence)
    Pi_improvement_month: int = 6  # evidence completion month
    n_paths: int = 1000         #
    bankruptcy_threshold: float = 0.0  # m yen
    # 7647
    #  A_N  bankruptcy_threshold month


#
def simulate_asset_paths(cfg: DissipationConfig, sigma_val: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Generate n_paths realisations of the A_N(t) stochastic process.
    sigma=0: random walk (drift=0)
    sigma=1: arithmetic Brownian motion with negative drift
    Returns: shape (n_paths, T_months+1)
    """
    drift = cfg.drift_sigma0 if sigma_val == 0 else cfg.drift_sigma1
    n, T = cfg.n_paths, cfg.T_months

    paths = np.zeros((n, T + 1))
    paths[:, 0] = cfg.A_N_init

    shocks = rng.normal(0, cfg.diffusion, size=(n, T))
    for t in range(T):
        paths[:, t + 1] = paths[:, t] + drift + shocks[:, t]

    #  0
    for t in range(1, T + 1):
        paths[:, t] = np.where(paths[:, t] < cfg.bankruptcy_threshold,
                                cfg.bankruptcy_threshold, paths[:, t])
    return paths


#   A_N*(t)
def compute_cost_threshold(cfg: DissipationConfig) -> np.ndarray:
    """
    A_N*(t) = (k + OC) / Π(t)
    Pi(t) improves from Pi_base to Pi_improved as evidence collection progresses.
    """
    T = cfg.T_months
    Pi_t = np.where(np.arange(T + 1) < cfg.Pi_improvement_month,
                    cfg.Pi_base, cfg.Pi_improved)
    return (cfg.k_cost + cfg.OC) / Pi_t


#
def compute_optimal_timing(paths: np.ndarray,
                            threshold: np.ndarray,
                            cfg: DissipationConfig) -> pd.DataFrame:
    """
    For each path, record the duration for which A_N(t) >= A_N*(t), and
    the first time the threshold is breached (opportunity-loss month).
    """
    n, T1 = paths.shape
    T = T1 - 1
    feasible = paths >= threshold[np.newaxis, :]  # (n, T+1)

    #
    feasible_duration = feasible.sum(axis=1)  # month

    # monthmonth
    first_miss = np.argmax(~feasible, axis=1)
    first_miss = np.where(feasible.all(axis=1), T + 1, first_miss)

    # monthA_N month
    optimal_t = np.argmax(paths, axis=1)

    # monthA_N  0 month
    bankrupt = (paths <= cfg.bankruptcy_threshold)
    bankrupt_t = np.argmax(bankrupt, axis=1)
    bankrupt_t = np.where(bankrupt.any(axis=1), bankrupt_t, T + 1)

    return pd.DataFrame({
        "feasible_months": feasible_duration,
        "first_miss_month": first_miss,
        "optimal_filing_month": optimal_t,
        "bankruptcy_month": bankrupt_t,
    })


#  Expected recovery
def expected_recovery(paths: np.ndarray,
                       threshold: np.ndarray,
                       Pi_t: np.ndarray,
                       cfg: DissipationConfig,
                       filing_month: Optional[int] = None) -> pd.Series:
    """
    For each path, compute the expected net recovery at filing month t.
    E[recovery] = Pi(t) * min(A_N(t), w) - k_cost - OC

    filing_month=None A_N month
    """
    w = 100.0   # m yen
    recoveries = []

    for i in range(cfg.n_paths):
        if filing_month is None:
            t = int(np.argmax(paths[i]))
        else:
            t = min(filing_month, cfg.T_months)

        A_t = paths[i, t]
        Pi = Pi_t[t]
        gross = Pi * min(A_t, w)
        net = gross - cfg.k_cost - cfg.OC
        recoveries.append(max(net, 0.0))

    return pd.Series(recoveries)


#
def quantify_sigma_dual_effect(cfg: DissipationConfig,
                                params: dict,
                                rng: np.random.Generator) -> pd.DataFrame:
    """
    45
    σ=1 
    (i)  Pr(B|s) 
    (ii) A Dissipation
    
    """
    results = []
    for sigma_val in [0, 1]:
        paths = simulate_asset_paths(cfg, sigma_val, rng)
        Pi_t = np.where(np.arange(cfg.T_months + 1) < cfg.Pi_improvement_month,
                         cfg.Pi_base, cfg.Pi_improved)
        threshold = compute_cost_threshold(cfg)

        timing_df = compute_optimal_timing(paths, threshold, cfg)
        rec = expected_recovery(paths, threshold, Pi_t, cfg)

        # Pr(B|s)
        p_B_sigma = params["signal_sigma"]["B"]["p"]
        p_G_sigma = params["signal_sigma"]["G"]["p"]
        if sigma_val == 1:
            # log(_) = log(p_B / p_G)
            Pr_B_prior = params["prior"]["lambda_B"]
            log_odds_prior = np.log(Pr_B_prior / (1 - Pr_B_prior))
            log_odds_updated = log_odds_prior + np.log(p_B_sigma / p_G_sigma)
            Pr_B_updated = 1 / (1 + np.exp(-log_odds_updated))
        else:
            Pr_B_updated = params["prior"]["lambda_B"]

        # A A_N
        miss_rate = (timing_df["first_miss_month"] < cfg.Pi_improvement_month).mean()

        results.append({
            "sigma": sigma_val,
            "Pr_B_updated": round(Pr_B_updated, 4),
            "mean_feasible_months": round(timing_df["feasible_months"].mean(), 2),
            "mean_optimal_filing": round(timing_df["optimal_filing_month"].mean(), 2),
            "mean_expected_recovery_MM": round(rec.mean(), 2),
            "early_miss_rate": round(miss_rate, 4),
            "F_remedy_needed_rate": round(miss_rate, 4),  # F
        })

    return pd.DataFrame(results)


#
def delay_cost_analysis(cfg: DissipationConfig,
                         rng: np.random.Generator,
                         sigma_val: int = 1) -> pd.DataFrame:
    """
    month t  0T_months 
    
    """
    paths = simulate_asset_paths(cfg, sigma_val, rng)
    Pi_t = np.where(np.arange(cfg.T_months + 1) < cfg.Pi_improvement_month,
                     cfg.Pi_base, cfg.Pi_improved)

    rows = []
    for t in range(cfg.T_months + 1):
        rec = expected_recovery(paths, None, Pi_t, cfg, filing_month=t)
        rows.append({
            "filing_month": t,
            "mean_recovery": round(rec.mean(), 3),
            "recovery_10pct": round(np.percentile(rec, 10), 3),
            "recovery_90pct": round(np.percentile(rec, 90), 3),
            "Pi_at_t": round(Pi_t[t], 4),
        })
    return pd.DataFrame(rows)


#
def plot_dissipation_results(paths_s0: np.ndarray,
                              paths_s1: np.ndarray,
                              threshold: np.ndarray,
                              sigma_df: pd.DataFrame,
                              delay_df: pd.DataFrame,
                              cfg: DissipationConfig,
                              save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Asset Dissipation Dynamics  σ ",
                 fontsize=13, y=1.02)

    months = np.arange(cfg.T_months + 1)

    # (1) =0 vs =1
    ax = axes[0]
    #
    for i in range(min(60, cfg.n_paths)):
        ax.plot(months, paths_s0[i], color="#1D9E75", alpha=0.05, lw=0.8)
        ax.plot(months, paths_s1[i], color="#D85A30", alpha=0.05, lw=0.8)
    ax.plot(months, np.percentile(paths_s0, 50, axis=0),
            color="#1D9E75", lw=2.5, label="=0 ")
    ax.plot(months, np.percentile(paths_s1, 50, axis=0),
            color="#D85A30", lw=2.5, label="=1 ")
    ax.plot(months, threshold, color="black", lw=2, ls="--",
            label="A_N* ")
    ax.axvline(cfg.Pi_improvement_month, color="gray", lw=1, ls=":",
               label=f"evidence completion montht={cfg.Pi_improvement_month}")
    ax.set_xlabel("month")
    ax.set_ylabel("A_N(t)  m yen")
    ax.set_title("(1) Dissipation\nσ=0: σ=1: : ")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(bottom=0)

    # (2)
    ax = axes[1]
    metrics = ["Pr_B_updated", "mean_feasible_months", "mean_expected_recovery_MM",
               "F_remedy_needed_rate"]
    labels = ["Pr(B|s)\n", "\nmonth",
              "Expected recovery\nm yen", "F\n"]
    x = np.arange(len(metrics))
    width = 0.35
    v0 = sigma_df[sigma_df["sigma"] == 0][metrics].values[0]
    v1 = sigma_df[sigma_df["sigma"] == 1][metrics].values[0]

    #  0-1
    max_vals = np.maximum(np.abs(v0), np.abs(v1)) + 1e-9
    v0_norm = v0 / max_vals
    v1_norm = v1 / max_vals

    bars0 = ax.bar(x - width / 2, v0_norm, width, label="σ=0",
                   color="#1D9E75", alpha=0.8)
    bars1 = ax.bar(x + width / 2, v1_norm, width, label="σ=1",
                   color="#D85A30", alpha=0.8)

    #
    for bar, val in zip(bars0, v0):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{val:.2f}",
                ha="center", fontsize=7, color="#1D9E75")
    for bar, val in zip(bars1, v1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{val:.2f}",
                ha="center", fontsize=7, color="#D85A30")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("=1")
    ax.set_title("(2) σ \n i: Pr(B)  ii: ")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.35)

    # (3) =1
    ax = axes[2]
    ax.plot(delay_df["filing_month"], delay_df["mean_recovery"],
            lw=2.5, color="#378ADD", label="Expected recovery")
    ax.fill_between(delay_df["filing_month"],
                    delay_df["recovery_10pct"], delay_df["recovery_90pct"],
                    alpha=0.2, color="#378ADD", label="10-90 ")
    opt_t = delay_df.loc[delay_df["mean_recovery"].idxmax(), "filing_month"]
    ax.axvline(opt_t, color="red", lw=1.5, ls="--",
               label=f"month t*={int(opt_t)}")
    ax.axvline(cfg.Pi_improvement_month, color="gray", lw=1, ls=":",
               label=f"month t={cfg.Pi_improvement_month}")
    ax.set_xlabel("month t")
    ax.set_ylabel("m yen")
    ax.set_title("(3) σ=1\n")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#
def main(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    params = load_params()
    ad = params["asset_dissipation"]
    rng = np.random.default_rng(params["seed"])

    cfg = DissipationConfig(
        A_N_init=ad["A_N_init"],
        drift_sigma0=ad["drift_sigma0"],
        drift_sigma1=ad["drift_sigma1"],
        diffusion=ad["diffusion"],
        T_months=ad["T_months"],
        k_cost=params["costs"]["k_P"],
        OC=ad["OC"],
        Pi_base=0.30,
        Pi_improved=0.55,
        Pi_improvement_month=6,
        n_paths=1500,
    )

    print("[1/4] Dissipationσ=0  σ=1...")
    paths_s0 = simulate_asset_paths(cfg, sigma_val=0, rng=rng)
    paths_s1 = simulate_asset_paths(cfg, sigma_val=1, rng=rng)
    threshold = compute_cost_threshold(cfg)

    print("[2/4] σ ...")
    sigma_df = quantify_sigma_dual_effect(cfg, params, rng)
    sigma_df.to_csv(f"{output_dir}/sigma_dual_effect.csv", index=False)
    print(sigma_df.to_string(index=False))
    print()
    print("  ")
    pr_b0 = sigma_df.loc[sigma_df["sigma"]==0, "Pr_B_updated"].values[0]
    pr_b1 = sigma_df.loc[sigma_df["sigma"]==1, "Pr_B_updated"].values[0]
    rec0 = sigma_df.loc[sigma_df["sigma"]==0, "mean_expected_recovery_MM"].values[0]
    rec1 = sigma_df.loc[sigma_df["sigma"]==1, "mean_expected_recovery_MM"].values[0]
    print(f"  (i) : Pr(B|σ=0)={pr_b0:.3f} → Pr(B|σ=1)={pr_b1:.3f}")
    print(f"  (ii) Dissipation: Expected recovery σ=0  {rec0:.1f} MMσ=1  {rec1:.1f} MM")
    print(f"  → σ=1 F ")

    print("\n[3/4] σ=1...")
    delay_df = delay_cost_analysis(cfg, rng, sigma_val=1)
    delay_df.to_csv(f"{output_dir}/delay_cost_sigma1.csv", index=False)
    opt_t = delay_df.loc[delay_df["mean_recovery"].idxmax(), "filing_month"]
    print(f"  month: t* = {int(opt_t)} month")
    print(f"  t* Expected recovery: {delay_df.loc[delay_df['filing_month']==int(opt_t), 'mean_recovery'].values[0]:.2f} m yen")

    print("\n[4/4] generating figures...")
    plot_dissipation_results(paths_s0, paths_s1, threshold,
                             sigma_df, delay_df, cfg,
                             save_path=f"{output_dir}/asset_dissipation.png")

    print(f"\nDone. Output directory: {output_dir}/")
    return sigma_df, delay_df


if __name__ == "__main__":
    main()
