"""
run_all.py
==========
Master script: runs all four simulation modules in sequence and saves
results to the outputs/ directory (Sections 5.3-5.5 of the paper).

Usage:
    python run_all.py                         # paper baseline (n=5000, grid=15)
    python run_all.py --n_firms 10000         # larger sample
    python run_all.py --output myresults      # custom output directory
    python run_all.py --n_firms 500 --grid_points 5  # quick test (~15s)

Google Colab:
    !python run_all.py
"""
import argparse
import os
import time
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Abusive Company Division Simulation: run all modules")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--n_firms", type=int, default=5000,
                        help="Number of simulated firms (default: 5000)")
    parser.add_argument("--grid_points", type=int, default=15,
                        help="Number of threshold grid points (default: 15)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    start_total = time.time()

    print("=" * 60)
    print("  Abusive Company Divisions: Multi-Dimensional Screening Simulation")
    print("  Sections 5.3–5.5 — All modules")
    print("=" * 60)

    # ── Module 1 ───────────────────────────────────────
    print("\n▶ Module 1: montecarlo_calibration.py")
    print("-" * 40)
    t0 = time.time()
    import montecarlo_calibration as mc
    mc.main(n_firms=args.n_firms,
            grid_points=args.grid_points,
            output_dir=args.output)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── Module 2 ───────────────────────────────────────
    print("\n▶ Module 2: signaling_equilibrium.py")
    print("-" * 40)
    t0 = time.time()
    import signaling_equilibrium as se
    se.main(output_dir=args.output)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── Module 3 ───────────────────────────────────────
    print("\n▶ Module 3: bottleneck_sensitivity.py")
    print("-" * 40)
    t0 = time.time()
    import bottleneck_sensitivity as bs
    bs.main(output_dir=args.output)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── Module 4 ───────────────────────────────────────
    print("\n▶ Module 4: asset_dissipation.py")
    print("-" * 40)
    t0 = time.time()
    import asset_dissipation as ad
    ad.main(output_dir=args.output)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── Output files ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Output files")
    print("=" * 60)
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:<45} {size_kb:>6.1f} KB")

    print(f"\nTotal runtime: {time.time()-start_total:.1f}s")
    print(f"Output directory: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
