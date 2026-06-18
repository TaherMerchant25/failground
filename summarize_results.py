#!/usr/bin/env python3
"""
Summarize FAILGROUND experiment results across benchmarks and ablations.
"""
import json, os, glob

results_dir = os.path.join(os.path.dirname(__file__), "results")

files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
files = [f for f in files if not os.path.basename(f).startswith("fakg_")]

print(f"{'File':<35} {'TSR':>6} {'SHR':>6} {'GCS':>6} {'CP':>6} {'FAKG':>8} {'Churn':>7}")
print("-" * 80)

for fp in files:
    name = os.path.basename(fp).replace(".json", "")
    try:
        with open(fp) as f:
            data = json.load(f)
        for bench, m in data.items():
            tsr = m.get("tsr", 0)
            shr = m.get("shr", 0)
            gcs = m.get("gcs", 0)
            cp  = m.get("cp", 0)
            fakg = m.get("fakg_growth", 0)
            churn = m.get("churn", 0)
            n = m.get("n_episodes", "?")
            print(f"{name:<35} {tsr:>6.3f} {shr:>6.3f} {gcs:>6.3f} {cp:>6.3f} {fakg:>8.3f} {churn:>7.3f}  (n={n})")
    except Exception as e:
        print(f"{name:<35} ERROR: {e}")
