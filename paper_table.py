#!/usr/bin/env python3
"""
Print FAILGROUND results as LaTeX table rows for paper.
Usage: python paper_table.py
"""
import json, glob, os

ABLATION_LABELS = {
    "none": "FAILGROUND (full)",
    "A1": "w/o 8-tuple FPT",
    "A2": "w/o graph edges",
    "A3": "BM25 retrieval",
    "A4": "FIFO eviction",
    "A5": "no task-sim AFM",
    "A6": "no gamma filter",
}

results_dir = os.path.join(os.path.dirname(__file__), "results")
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
files = [f for f in files if not os.path.basename(f).startswith("fakg_")]

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{llrrrrr}")
print("\\toprule")
print("Benchmark & Method & TSR & SHR & FAKG & Churn & n \\\\")
print("\\midrule")

for fp in files:
    name = os.path.basename(fp).replace(".json", "")
    parts = name.rsplit("_", 1)
    bench = parts[0] if len(parts) == 2 else name
    ablation = parts[1] if len(parts) == 2 else "none"
    label = ABLATION_LABELS.get(ablation, ablation)
    try:
        with open(fp) as f:
            data = json.load(f)
        for b, m in data.items():
            tsr  = m.get("tsr", 0)
            shr  = m.get("shr", 0)
            fakg = m.get("fakg_growth", 0)
            churn= m.get("churn", 0)
            n    = m.get("n_episodes", "?")
            print(f"{b} & {label} & {tsr:.3f} & {shr:.3f} & {fakg:.2f} & {churn:.3f} & {n} \\\\")
    except Exception as e:
        print(f"% Error reading {fp}: {e}")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{{FAILGROUND ablation results}}")
print("\\end{table}")
