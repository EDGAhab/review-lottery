"""08_build_rho_table.py — Assemble rho_table.csv from per-year fit JSONs."""
import json, os
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
NOTES = os.path.join(WORK, "phase2_notes")
rows = []
for y in range(2017, 2026):
    fp = os.path.join(NOTES, f"fit_{y}.json")
    if not os.path.exists(fp):
        print(f"missing fit_{y}.json, skipping"); continue
    with open(fp) as f:
        s = json.load(f)
    rows.append(dict(
        year=y,
        n_papers=s.get("n_papers"), n_reviews=s.get("n_rev"),
        sigma_paper_mean=round(s["sigma_paper_mean"], 3),
        sigma_paper_sd=round(s["sigma_paper_sd"], 3),
        rho_mean=round(s["rho_mean"], 3),
        rho_lo=round(s["rho_ci95"][0], 3), rho_hi=round(s["rho_ci95"][1], 3),
        divergences=s.get("divergences"),
        split_rhat_sigma=s.get("split_rhat_sigma"),
        minutes=s.get("minutes"),
    ))
df = pd.DataFrame(rows).sort_values("year")
df.to_csv(os.path.join(NOTES, "rho_table.csv"), index=False)
print(df.to_string(index=False))
