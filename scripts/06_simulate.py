"""06_simulate.py — Two-committee simulation from posterior draws.

For each year, B=1000 posterior draws:
    theta_i^(b) = sigma_paper^(b) * z_i^(b)
    two independent committees of k scores: latent ~ Normal(theta, 1) -> discretize via cutpoints^(b)
    each committee -> (mean, sd) -> decision model (logistic + threshold rule)
Metrics per draw: disagreement rate, P(flip|accepted), P(flip|rejected),
accepted-set Jaccard distance, committee-mean correlation.
k = 2, 3, 4. Batches of 100 draws for memory.

Cohort: papers with known decisions (accept/reject); withdrawn excluded here
(decision undefined), declared. Theta draws subset from the full-paper trace.

Usage: python3 06_simulate.py [--year YYYY] [--all]  (venv python)
"""
import argparse, json, os
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
NOTES = os.path.join(WORK, "phase2_notes")
B = 1000
BATCH = 100
KS = [2, 3, 4]
SEED = 7

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_year_inputs(year):
    import joblib
    d = np.load(os.path.join(POST, f"draws_{year}.npz"))
    z, sp, c0, deltas = d["z"], d["sigma_paper"], d["c0"], d["deltas"]
    ch, dr, P = z.shape
    D = ch * dr
    z = z.reshape(D, P); sp = sp.reshape(D); c0 = c0.reshape(D)
    C = deltas.shape[-1] + 2
    if C > 2:
        dl = deltas.reshape(D, C - 2)
        cuts = np.concatenate([c0[:, None], c0[:, None] + np.cumsum(dl, axis=1)], axis=1)
    else:
        cuts = c0[:, None]
    # subsample B draws
    rng = np.random.default_rng(SEED)
    idx = rng.choice(D, size=min(B, D), replace=False)
    # cohort papers: known decisions, order matching trace paper order
    r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"), columns=["paper_id", "year", "score_raw"])
    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    pids_all = r[r["year"] == year]["paper_id"].unique()
    pos = {pid: i for i, pid in enumerate(pids_all)}
    pmap = pd.DataFrame({"paper_id": pids_all})
    py = p[p["year"] == year].set_index("paper_id")
    keep = [pid for pid in pids_all if pid in py.index and py.loc[pid, "decision_binary"] != "unknown"]
    keep_idx = np.array([pos[pid] for pid in keep])
    scale_vals = np.array(sorted(r[r["year"] == year]["score_raw"].unique()))
    dm = joblib.load(os.path.join(WORK, "decision_model.joblib"))
    with open(os.path.join(NOTES, "threshold_rule.json")) as f:
        thr = json.load(f)[str(year)]["threshold"]
    return dict(theta=(sp[idx][:, None] * z[idx][:, keep_idx]),
                cuts=cuts[idx], scale=scale_vals, dm=dm, thr=thr,
                n_cohort=len(keep), year=year)

def committee_decisions(theta_b, cuts_b, scale, k, dm, thr, year, rng):
    """theta_b: (Bb,P). Returns (dec_log, dec_thr, mean1, mean2) per committee pair."""
    Bb, P = theta_b.shape
    lat1 = theta_b[:, :, None] + rng.standard_normal((Bb, P, k))
    lat2 = theta_b[:, :, None] + rng.standard_normal((Bb, P, k))
    # discretize: level = #(cutpoints < latent)
    lv1 = (lat1[..., None] > cuts_b[:, None, None, :]).sum(-1)
    lv2 = (lat2[..., None] > cuts_b[:, None, None, :]).sum(-1)
    s1 = scale[lv1]; s2 = scale[lv2]
    m1 = s1.mean(-1); m2 = s2.mean(-1)
    sd1 = s1.std(-1, ddof=1); sd2 = s2.std(-1, ddof=1)
    feats = dm["features"]
    def predict(m, sd):
        X = np.zeros((m.size, len(feats)))
        X[:, feats.index("mean_score")] = m.ravel()
        X[:, feats.index("sd_score")] = sd.ravel()
        col = f"y_{year}"
        if col in feats:
            X[:, feats.index(col)] = 1.0
        return sigmoid(X @ dm["model"].coef_[0] + dm["model"].intercept_[0]).reshape(m.shape) >= 0.5
    d1 = predict(m1, sd1); d2 = predict(m2, sd2)
    t1 = m1 >= thr; t2 = m2 >= thr
    return (d1, d2), (t1, t2), m1, m2

def metrics(d1, d2):
    d1 = d1.astype(bool); d2 = d2.astype(bool)
    dis = (d1 != d2).mean(axis=1)
    a1 = d1.sum(axis=1); a2 = d2.sum(axis=1)
    inter = (d1 & d2).sum(axis=1); union = (d1 | d2).sum(axis=1)
    jacc = np.where(union > 0, 1 - inter / union, np.nan)
    with np.errstate(invalid="ignore"):
        p_flip_acc = np.where(a1 > 0, ((d1) & (~d2)).sum(axis=1) / a1, np.nan)
        r1 = (~d1).sum(axis=1)
        p_flip_rej = np.where(r1 > 0, ((~d1) & (d2)).sum(axis=1) / r1, np.nan)
    return dis, p_flip_acc, p_flip_rej, jacc

def run_year(year):
    inp = load_year_inputs(year)
    theta, cuts = inp["theta"], inp["cuts"]
    D = theta.shape[0]
    rng = np.random.default_rng(SEED + year)
    rows = []
    for k in KS:
        acc = {m: [] for m in ["dis", "pfa", "pfr", "jac", "corr",
                               "dis_t", "pfa_t", "pfr_t", "jac_t"]}
        for s in range(0, D, BATCH):
            sl = slice(s, min(s + BATCH, D))
            (d1, d2), (t1, t2), m1, m2 = committee_decisions(
                theta[sl], cuts[sl], inp["scale"], k, inp["dm"], inp["thr"], year, rng)
            dis, pfa, pfr, jac = metrics(d1, d2)
            dist, pfat, pfrt, jact = metrics(t1, t2)
            corr = np.array([np.corrcoef(m1[b], m2[b])[0, 1] for b in range(m1.shape[0])])
            for key, arr in [("dis", dis), ("pfa", pfa), ("pfr", pfr), ("jac", jac),
                             ("corr", corr), ("dis_t", dist), ("pfa_t", pfat),
                             ("pfr_t", pfrt), ("jac_t", jact)]:
                acc[key].append(arr)
        for key, arr in acc.items():
            v = np.concatenate(arr)
            v = v[~np.isnan(v)]
            rows.append(dict(year=year, k=k, metric=key, rule="logistic" if not key.endswith("_t") else "threshold",
                             mean=float(v.mean()), lo=float(np.quantile(v, 0.025)),
                             hi=float(np.quantile(v, 0.975)), n_draws=int(v.size)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(NOTES, f"sim_{year}.csv"), index=False)
    print(f"[year {year}] simulation done, cohort n={inp['n_cohort']}", flush=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int); ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    years = list(range(2017, 2026)) if args.all else [args.year]
    for y in years:
        out = os.path.join(NOTES, f"sim_{y}.csv")
        if os.path.exists(out):
            print(f"[year {y}] sim exists, skipping", flush=True); continue
        run_year(y)

if __name__ == "__main__":
    main()
