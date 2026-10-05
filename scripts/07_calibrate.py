"""07_calibrate.py — Dual calibration.

External (partial, numbers comparison; G1 ruled out refitting on public NeurIPS
data since only accepted papers are public):
  - simulated disagreement (k=2,3; logistic) vs 23.0% (NeurIPS 2021), 25.9% (2014)
  - 1 - P(flip|accepted) vs accept precision 50.6%
  - committee-mean correlation vs 0.575 (2021) / 0.55 (2014)
  - rho vs 2014 ~50% subjective variance share
Internal (model-free):
  - ICLR papers with 4+ reviews: 200 random 2+2 splits through the fitted
    decision model -> disagreement distribution; compare with k=2 simulation.
  - characteristics of 4+ papers vs all papers (v3.0 risk table requirement).
Outputs: calibration tables + 3 figures + checkpoint assessment JSON.
"""
import json, os
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
FIG = os.path.join(WORK, "figures")
NOTES = os.path.join(WORK, "phase2_notes")
os.makedirs(FIG, exist_ok=True)

def load_sim():
    dfs = []
    for y in range(2017, 2026):
        fp = os.path.join(NOTES, f"sim_{y}.csv")
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
    return pd.concat(dfs, ignore_index=True)

def get(df, year, k, metric, rule="logistic"):
    r = df[(df.year == year) & (df.k == k) & (df.metric == metric) & (df.rule == rule)]
    assert len(r) == 1, (year, k, metric, rule)
    return r.iloc[0]

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib

    sim = load_sim()
    rho = pd.read_csv(os.path.join(NOTES, "rho_table.csv"))

    # ---------- external anchor table ----------
    ext_rows = []
    for y in range(2017, 2026):
        for k in (2, 3):
            d = get(sim, y, k, "dis")
            pf = get(sim, y, k, "pfa")
            co = get(sim, y, k, "corr")
            ext_rows.append(dict(year=y, k=k, dis_mean=d["mean"], dis_lo=d["lo"], dis_hi=d["hi"],
                                 accept_precision=1 - pf["mean"],
                                 ap_lo=1 - pf["hi"], ap_hi=1 - pf["lo"],
                                 corr_mean=co["mean"], corr_lo=co["lo"], corr_hi=co["hi"]))
    ext = pd.DataFrame(ext_rows)
    ext.to_csv(os.path.join(NOTES, "calib_external.csv"), index=False)

    # ---------- internal anchor: 200 random 2+2 splits ----------
    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"),
                        columns=["paper_id", "year", "score_raw"])
    p4 = p[(p["k"] >= 4) & (p["decision_binary"] != "unknown")].copy()
    # score matrix per paper (pad to max k)
    grp = r[r["paper_id"].isin(set(p4["paper_id"]))].copy()
    maxk = int(grp.groupby("paper_id").size().max())
    pids = p4["paper_id"].to_numpy()
    pid_to_row = {pid: i for i, pid in enumerate(pids)}
    smat = np.full((len(pids), maxk), np.nan)
    kvec = np.zeros(len(pids), dtype=int)
    for pid, g in grp.groupby("paper_id"):
        i = pid_to_row[pid]
        v = g["score_raw"].to_numpy()
        smat[i, :len(v)] = v
        kvec[i] = len(v)
    dm = joblib.load(os.path.join(WORK, "decision_model.joblib"))
    feats = dm["features"]; coef = dm["model"].coef_[0]; icept = dm["model"].intercept_[0]
    years = p4["year"].to_numpy()
    year_cols = {int(c.split("_")[1]): feats.index(c) for c in feats if c.startswith("y_")}

    def predict_groups(m, sd, yrs):
        X = np.zeros((m.shape[0], len(feats)))
        X[:, feats.index("mean_score")] = m
        X[:, feats.index("sd_score")] = sd
        for yy, col in year_cols.items():
            X[yrs == yy, col] = 1.0
        return (1 / (1 + np.exp(-(X @ coef + icept)))) >= 0.5

    rng = np.random.default_rng(11)
    P4 = len(pids)
    dis_reps = []
    for rep in range(200):
        U = rng.random((P4, maxk))
        U[np.arange(maxk)[None, :] >= kvec[:, None]] = np.inf
        order = np.argsort(U, axis=1)[:, :4]
        s4 = smat[np.arange(P4)[:, None], order]
        g1, g2 = s4[:, :2], s4[:, 2:]
        m1, sd1 = g1.mean(1), g1.std(1, ddof=1)
        m2, sd2 = g2.mean(1), g2.std(1, ddof=1)
        d1 = predict_groups(m1, sd1, years); d2 = predict_groups(m2, sd2, years)
        dis_reps.append(float((d1 != d2).mean()))
    dis_reps = np.array(dis_reps)
    # per-year internal splits (for comparison with per-year k=2 sims)
    per_year = {}
    for y in sorted(p4["year"].unique()):
        mask = years == y
        n_y = mask.sum()
        reps_y = []
        for rep in range(200):
            U = rng.random((n_y, maxk))
            kv = kvec[mask]
            U[np.arange(maxk)[None, :] >= kv[:, None]] = np.inf
            order = np.argsort(U, axis=1)[:, :4]
            s4 = smat[mask][np.arange(n_y)[:, None], order]
            g1, g2 = s4[:, :2], s4[:, 2:]
            d1 = predict_groups(g1.mean(1), g1.std(1, ddof=1), np.full(n_y, y))
            d2 = predict_groups(g2.mean(1), g2.std(1, ddof=1), np.full(n_y, y))
            reps_y.append(float((d1 != d2).mean()))
        per_year[y] = dict(mean=float(np.mean(reps_y)),
                           lo=float(np.quantile(reps_y, 0.025)),
                           hi=float(np.quantile(reps_y, 0.975)))
    with open(os.path.join(NOTES, "internal_anchor.json"), "w") as f:
        json.dump({"pooled": {"mean": float(dis_reps.mean()),
                              "lo": float(np.quantile(dis_reps, 0.025)),
                              "hi": float(np.quantile(dis_reps, 0.975)),
                              "n_papers": int(P4)},
                   "per_year": {str(k): v for k, v in per_year.items()}}, f, indent=2)

    # characteristics: 4+ papers vs all (v3.0 risk requirement)
    pall = p[p["decision_binary"] != "unknown"]
    char = pd.DataFrame([
        {"group": "all", "n": len(pall), "mean_score": pall["mean_score"].mean(),
         "accept_rate": (pall["decision_binary"] == "accept").mean(), "mean_k": pall["k"].mean()},
        {"group": "4+reviews", "n": len(p4), "mean_score": p4["mean_score"].mean(),
         "accept_rate": (p4["decision_binary"] == "accept").mean(), "mean_k": p4["k"].mean()},
    ])
    char.to_csv(os.path.join(NOTES, "anchor_characteristics.csv"), index=False)

    # ---------- figures ----------
    # 1. external: sim disagreement by year vs NeurIPS anchors
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, mk in [(2, "o"), (3, "s")]:
        e = ext[ext.k == k].sort_values("year")
        ax.errorbar(e["year"], e["dis_mean"], yerr=[e["dis_mean"] - e["dis_lo"], e["dis_hi"] - e["dis_mean"]],
                    marker=mk, capsize=3, label=f"simulated k={k} (logistic)")
    ax.axhline(0.230, color="red", ls="--", label="NeurIPS 2021: 23.0%")
    ax.axhline(0.259, color="darkred", ls=":", label="NeurIPS 2014: 25.9%")
    ax.set_xlabel("ICLR year"); ax.set_ylabel("two-committee disagreement rate")
    ax.set_title("External calibration: simulated disagreement vs NeurIPS experiments")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_calib_external.png"), dpi=120); plt.close(fig)

    # 2. internal: per-year k=2 sim posterior mean vs internal split distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    yrs = sorted(per_year)
    simk2 = [get(sim, y, 2, "dis") for y in yrs]
    ax.errorbar(yrs, [s["mean"] for s in simk2],
                yerr=[[s["mean"] - s["lo"] for s in simk2], [s["hi"] - s["mean"] for s in simk2]],
                marker="o", capsize=3, label="k=2 simulation (model)")
    ax.errorbar([y + 0.15 for y in yrs], [per_year[y]["mean"] for y in yrs],
                yerr=[[per_year[y]["mean"] - per_year[y]["lo"] for y in yrs],
                      [per_year[y]["hi"] - per_year[y]["mean"] for y in yrs]],
                marker="s", capsize=3, label="internal 2+2 split (model-free)")
    ax.set_xlabel("ICLR year"); ax.set_ylabel("disagreement rate")
    ax.set_title("Internal calibration: model simulation vs model-free split-half")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_calib_internal.png"), dpi=120); plt.close(fig)

    # 3. rho with 0.5 reference (2014 ~50% subjective)
    fig, ax = plt.subplots(figsize=(9, 5))
    rr = rho.sort_values("year")
    ax.errorbar(rr["year"], rr["rho_mean"],
                yerr=[rr["rho_mean"] - rr["rho_lo"], rr["rho_hi"] - rr["rho_mean"]],
                marker="o", capsize=3)
    ax.axhline(0.5, color="red", ls="--", label="NeurIPS 2014: ~50% subjective variance")
    ax.set_xlabel("ICLR year"); ax.set_ylabel("rho = paper variance share")
    ax.set_title("Signal-to-noise ratio by year")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rho.png"), dpi=120); plt.close(fig)

    print("calibration done:", os.path.join(NOTES, "calib_external.csv"), flush=True)

if __name__ == "__main__":
    main()
