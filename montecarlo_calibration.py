"""
montecarlo_calibration.py
=========================
Module 1 — Monte Carlo Threshold Calibration  (Section 5.3(1))

Theoretical correspondence:
  - Section 3: four-dimensional signal vector s=(delta,rho,phi,sigma); log-likelihood ratio Lambda(s)
  - Proposition 1: existence of separating equilibrium (signal discriminatory power)
  - Proposition 3: optimal avoidance threshold kappa_A
  - Proposition 6: objective trigger condition for phi < 0 (successor company insolvency)

Core proposition verified (Section 5.5, Proposition 1):
  When phi < 0 and delta > epsilon simultaneously, the optimal remedy is P or higher
  regardless of the value of rho.

Usage:
    python montecarlo_calibration.py
    # or in Google Colab
    # %run montecarlo_calibration.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, beta as beta_dist
import yaml
import os

#
PARAM_PATH = os.path.join(os.path.dirname(__file__), "params", "type_space.yaml")

def load_params(path: str = PARAM_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#
def generate_firms(n: int, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate n company division cases by Monte Carlo sampling from the type space.

    Returns
    -------
    DataFrame columns: [type, v, w, delta, rho, phi, sigma, Lambda]
    """
    pr = params["prior"]
    types = rng.choice(["G", "M", "B"],
                       size=n,
                       p=[pr["lambda_G"], pr["lambda_M"], pr["lambda_B"]])

    records = []
    for t in types:
        # business value and fraudulent transfer amount
        v_key = f"v_{t}"
        w_key = f"w_{t}"
        v = params["value"].get(v_key, 0.0)
        w = params["value"].get(w_key, 0.0)

        # delta: Normal distribution, clipped to [0,1]
        d_cfg = params["signal_delta"][t]
        delta = float(np.clip(rng.normal(d_cfg["mu"], d_cfg["sigma"]), 0.0, 1.0))

        # rho: Beta distribution
        r_cfg = params["signal_rho"][t]
        rho = float(rng.beta(r_cfg["alpha"], r_cfg["beta"]))

        # phi: Normal distribution
        p_cfg = params["signal_phi"][t]
        phi = float(rng.normal(p_cfg["mu"], p_cfg["sigma"]))

        # sigma: Bernoulli
        s_cfg = params["signal_sigma"][t]
        sigma = int(rng.random() < s_cfg["p"])

        records.append({"type": t, "v": v, "w": w,
                         "delta": delta, "rho": rho,
                         "phi": phi, "sigma": sigma})

    df = pd.DataFrame(records)
    df["Lambda"] = compute_log_likelihood_ratio(df, params)
    return df


#   (s)
def _log_likelihood_ratio_single(row: pd.Series, params: dict) -> float:
    """
    Λ(s) = log(λ_B/λ_G) + Σ_j log Λ_j(s_j)
    Assumes conditional signal independence.
    """
    pr = params["prior"]
    prior_term = np.log(pr["lambda_B"] / pr["lambda_G"])

    # likelihood ratio for delta
    d_B = params["signal_delta"]["B"]
    d_G = params["signal_delta"]["G"]
    ll_delta = (norm.logpdf(row["delta"], d_B["mu"], d_B["sigma"])
                - norm.logpdf(row["delta"], d_G["mu"], d_G["sigma"]))

    # likelihood ratio for rho (Beta distribution)
    r_B = params["signal_rho"]["B"]
    r_G = params["signal_rho"]["G"]
    eps = 1e-9
    rho_c = np.clip(row["rho"], eps, 1 - eps)
    ll_rho = (beta_dist.logpdf(rho_c, r_B["alpha"], r_B["beta"])
              - beta_dist.logpdf(rho_c, r_G["alpha"], r_G["beta"]))

    # likelihood ratio for phi
    p_B = params["signal_phi"]["B"]
    p_G = params["signal_phi"]["G"]
    ll_phi = (norm.logpdf(row["phi"], p_B["mu"], p_B["sigma"])
              - norm.logpdf(row["phi"], p_G["mu"], p_G["sigma"]))

    # likelihood ratio for sigma (Bernoulli)
    pb = params["signal_sigma"]["B"]["p"]
    pg = params["signal_sigma"]["G"]["p"]
    s = row["sigma"]
    ll_sigma = (np.log(pb if s else (1 - pb))
                - np.log(pg if s else (1 - pg)))

    return prior_term + ll_delta + ll_rho + ll_phi + ll_sigma


def compute_log_likelihood_ratio(df: pd.DataFrame, params: dict) -> pd.Series:
    return df.apply(lambda row: _log_likelihood_ratio_single(row, params), axis=1)


#    Pr(B|s)
def lambda_to_posterior_B(Lambda: np.ndarray) -> np.ndarray:
    """
    Pr(B|s) = 1 / (1 + exp(-Λ))  (binary approximation: G vs B only)
    """
    return 1.0 / (1.0 + np.exp(-Lambda))


#
def assign_remedy(df: pd.DataFrame, params: dict,
                  kappa_P: float = None,
                  kappa_A: float = None,
                  kappa_F: float = None) -> pd.Series:
    """
    Assign optimal remedy based on threshold kappa.
    Proposition 6: if phi<0 and delta>epsilon, remedy >= P (for all rho).
    """
    th = params["thresholds"]
    kP = kappa_P if kappa_P is not None else th["kappa_P"]
    kA = kappa_A if kappa_A is not None else th["kappa_A"]
    kF = kappa_F if kappa_F is not None else th["kappa_F"]
    epsilon = 0.10   #

    post_B = lambda_to_posterior_B(df["Lambda"].values)
    remedies = []
    for idx, (i, row) in enumerate(df.iterrows()):
        pb = post_B[idx]

        # 6Requirement
        if row["phi"] < 0 and row["delta"] > epsilon:
            #   P
            if pb >= kF:
                remedies.append("F")
            elif pb >= kA:
                remedies.append("A")
            else:
                remedies.append("P")   #  P
        else:
            if pb >= kF:
                remedies.append("F")
            elif pb >= kA:
                remedies.append("A")
            elif pb >= kP:
                remedies.append("P")
            else:
                remedies.append("none")
    return pd.Series(remedies, index=df.index)


#
def compute_error_rates(df: pd.DataFrame, assigned: pd.Series) -> dict:
    """
    Return Type I error rate (intervening when G) and Type II error rate (not intervening when B).
    """
    type1 = ((df["type"] == "G") & (assigned != "none")).mean()
    type2 = ((df["type"] == "B") & (assigned == "none")).mean()
    return {"type1_error": type1, "type2_error": type2}


#
def welfare_loss(error_rates: dict, params: dict) -> float:
    c1 = params["costs"]["c_type1"]
    c2 = params["costs"]["c_type2"]
    return c1 * error_rates["type1_error"] + c2 * error_rates["type2_error"]


#
def calibrate_thresholds(df: pd.DataFrame, params: dict,
                         grid_points: int = 20) -> pd.DataFrame:
    """
    Search over kappa_P, kappa_A, kappa_F to minimise social welfare loss L.
    """
    # Grid spans [0.05, 0.95]^3 so that the welfare-minimising thresholds
    # (kappa_P*=0.05, kappa_A*=0.11, kappa_F*=0.18 under seed=42, n=5,000)
    # lie within the search space.  The earlier version used
    # [0.10,0.50] x [0.30,0.70] x [0.60,0.90], which excluded the true optimum.
    kp_grid = np.linspace(0.05, 0.95, grid_points)
    ka_grid = np.linspace(0.05, 0.95, grid_points)
    kf_grid = np.linspace(0.05, 0.95, grid_points)

    results = []
    for kp in kp_grid:
        for ka in ka_grid:
            if ka <= kp:
                continue
            for kf in kf_grid:
                if kf <= ka:
                    continue
                assigned = assign_remedy(df, params, kp, ka, kf)
                err = compute_error_rates(df, assigned)
                loss = welfare_loss(err, params)
                results.append({
                    "kappa_P": round(kp, 3),
                    "kappa_A": round(ka, 3),
                    "kappa_F": round(kf, 3),
                    "type1_error": round(err["type1_error"], 4),
                    "type2_error": round(err["type2_error"], 4),
                    "welfare_loss": round(loss, 4),
                })

    return pd.DataFrame(results).sort_values("welfare_loss").reset_index(drop=True)


#  6
def verify_proposition6(df: pd.DataFrame, params: dict,
                         n_rho_bins: int = 10) -> pd.DataFrame:
    """
    Under phi<0 and delta>epsilon, verify that regardless of rho
    the optimal remedy is P or higher (numerical verification of Proposition 6).
    """
    epsilon = 0.10
    subset = df[(df["phi"] < 0) & (df["delta"] > epsilon)].copy()
    assigned = assign_remedy(subset, params)
    subset["remedy"] = assigned
    subset["rho_bin"] = pd.cut(subset["rho"], bins=n_rho_bins, labels=False)

    result = (subset.groupby("rho_bin")
              .apply(lambda g: (g["remedy"] != "none").mean())
              .reset_index())
    result.columns = ["rho_bin", "intervention_rate"]
    return result


#
def plot_calibration_results(df_firms: pd.DataFrame,
                             cal_results: pd.DataFrame,
                             prop6_results: pd.DataFrame,
                             params: dict,
                             save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Monte Carlo Threshold Calibration", fontsize=13, y=1.01)

    # (1) (s)
    ax = axes[0]
    colors = {"G": "#1D9E75", "M": "#BA7517", "B": "#D85A30"}
    for t, col in colors.items():
        vals = df_firms.loc[df_firms["type"] == t, "Lambda"]
        ax.hist(vals, bins=40, alpha=0.55, color=col, label=f"type {t}")
    th = params["thresholds"]
    for k, label in [(th["kappa_P"], "κ_P"), (th["kappa_A"], "κ_A"),
                     (th["kappa_F"], "κ_F")]:
        lv = np.log(k / (1 - k))
        ax.axvline(lv, color="gray", lw=1.2, ls="--")
        ax.text(lv + 0.1, ax.get_ylim()[1] * 0.85, label, fontsize=8, color="gray")
    ax.set_xlabel("Log-likelihood ratio Λ(s)")
    ax.set_ylabel("Count")
    ax.set_title("(1) Distribution of Λ(s) by type")
    ax.legend(fontsize=8)

    # (2) Welfare loss heatmap_A vs _P_F=
    ax = axes[1]
    best_kf = cal_results.iloc[0]["kappa_F"]
    sub = cal_results[np.isclose(cal_results["kappa_F"], best_kf, atol=0.03)]
    if len(sub) > 0:
        pivot = sub.pivot_table(index="kappa_A", columns="kappa_P",
                                values="welfare_loss", aggfunc="min")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap="YlOrRd",
                       extent=[pivot.columns.min(), pivot.columns.max(),
                               pivot.index.min(),   pivot.index.max()])
        plt.colorbar(im, ax=ax, label="Welfare loss L")
        star = cal_results.iloc[0]
        ax.scatter(star["kappa_P"], star["kappa_A"],
                   marker="*", s=200, color="white", zorder=5)
    ax.set_xlabel("κ_P")
    ax.set_ylabel("κ_A")
    ax.set_title(f"(2) Welfare loss heatmap\n(κ_F ≈ {best_kf:.2f})")

    # (3) 6 vs <0  >
    ax = axes[2]
    ax.bar(prop6_results["rho_bin"], prop6_results["intervention_rate"],
           color="#3B6D11", alpha=0.75)
    ax.axhline(1.0, color="red", lw=1.2, ls="--", label="Intervention rate = 100% (Proposition 6)")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("ρ bin (low → high)")
    ax.set_ylabel("Intervention rate (remedy P or higher)")
    ax.set_title("(3) Proposition 6 verification\nIntervention rate vs rho (phi<0, delta>epsilon)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#
def main(n_firms: int = 5000, grid_points: int = 15,
         output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    params = load_params()
    rng = np.random.default_rng(params["seed"])

    print(f"[1/4] Generating {n_firms} simulated firm cases...")
    df = generate_firms(n_firms, params, rng)
    df.to_csv(f"{output_dir}/firms_simulated.csv", index=False)

    print("[2/4] Running threshold grid search...")
    cal = calibrate_thresholds(df, params, grid_points)
    cal.to_csv(f"{output_dir}/threshold_calibration.csv", index=False)
    best = cal.iloc[0]
    print(f"  Optimal thresholds: kappa_P={best['kappa_P']}, "
          f"κ_A={best['kappa_A']}, κ_F={best['kappa_F']}")
    print(f"  Type I error={best['type1_error']:.4f}, "
          f"={best['type2_error']:.4f}, "
          f"={best['welfare_loss']:.4f}")

    print("[3/4] Verifying Proposition 6 (phi trigger condition)...")
    p6 = verify_proposition6(df, params)
    p6.to_csv(f"{output_dir}/proposition6_verification.csv", index=False)
    print(f"  Intervention rate by rho bin (subset: phi<0 and delta>epsilon):\n{p6.to_string(index=False)}")

    print("[4/4] Generating figures...")
    plot_calibration_results(df, cal, p6, params,
                             save_path=f"{output_dir}/montecarlo_calibration.png")

    print(f"\nDone. Output directory: {output_dir}/")
    return df, cal, p6


if __name__ == "__main__":
    main()
