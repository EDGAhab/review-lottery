"""04b_conf_sensitivity.py — Sensitivity: ordered probit WITH beta*confidence.

Fits the prereg 3.1 model including beta*conf on the PeerSum ICLR subset
(2018-2022, the only years with confidence), and compares rho against the
main fits (04_model_fit.py, no confidence). If |delta rho| is small, the
main fits are robust to omitting confidence.

Declared analysis (see Phase 2 report).
Usage: python3 04b_conf_sensitivity.py [--year YYYY] (venv python)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
NOTES = os.path.join(WORK, "phase2_notes")

def fit_year_conf(year, tune=500, draws=500, seed=1):
    import pymc as pm
    import pytensor.tensor as pt
    pr = pd.read_parquet(os.path.join(PROC, "peersum_reviews.parquet"))
    g = pr[(pr["venue"] == "iclr") & (pr["year"] == year)].copy()
    scale_vals = sorted(g["rating"].unique())
    lm = {v: i for i, v in enumerate(scale_vals)}
    g["level"] = g["rating"].map(lm).astype(int)
    pids = g["paper_id"].unique()
    pmap = {p: i for i, p in enumerate(pids)}
    g["pidx"] = g["paper_id"].map(pmap).astype(int)
    level = g["level"].to_numpy(); pidx = g["pidx"].to_numpy()
    conf = g["confidence"].to_numpy(dtype=float)
    conf_c = conf - conf.mean()
    n_papers, C = len(pids), len(scale_vals)
    print(f"[conf {year}] papers={n_papers} reviews={len(g)} C={C}", flush=True)

    def Phi(x):
        return 0.5 * (1 + pt.erf(x / pt.sqrt(2.0)))
    t0 = time.time()
    with pm.Model() as model:
        sigma_paper = pm.HalfNormal("sigma_paper", sigma=2.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=n_papers)
        c0 = pm.Normal("c0", 0.0, 5.0)
        deltas = pm.HalfNormal("deltas", sigma=2.0, shape=C - 2) if C > 2 else None
        cutpoints = pt.concatenate([[c0], c0 + pt.cumsum(deltas)]) if C > 2 else pt.stack([c0])
        eta = sigma_paper * z[pidx] + beta * conf_c
        cuts = pt.concatenate([pt.as_tensor_variable(np.array([-np.inf])), cutpoints,
                               pt.as_tensor_variable(np.array([np.inf]))])
        up = Phi(cuts[level + 1] - eta); lo = Phi(cuts[level] - eta)
        pm.Potential("lik", pt.sum(pt.log(pt.maximum(up - lo, 1e-12))))
        trace = pm.sample(draws=draws, tune=tune, chains=2, cores=2,
                          init="advi", target_accept=0.9, random_seed=seed, progressbar=False)
    dt = time.time() - t0
    sp = trace.posterior["sigma_paper"].to_numpy().ravel()
    rho = sp ** 2 / (sp ** 2 + 1.0)
    be = trace.posterior["beta"].to_numpy().ravel()
    return {"year": year, "rho_mean": float(rho.mean()),
            "rho_ci95": [float(np.quantile(rho, 0.025)), float(np.quantile(rho, 0.975))],
            "beta_mean": float(be.mean()),
            "beta_ci95": [float(np.quantile(be, 0.025)), float(np.quantile(be, 0.975))],
            "divergences": int(trace.sample_stats["diverging"].sum()),
            "minutes": dt / 60, "n_papers": n_papers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    args = ap.parse_args()
    years = [args.year] if args.year else [2018, 2019, 2020, 2021, 2022]
    rows = []
    for y in years:
        rows.append(fit_year_conf(y))
        print(f"[conf {y}] rho={rows[-1]['rho_mean']:.3f} beta={rows[-1]['beta_mean']:.3f}", flush=True)
    with open(os.path.join(NOTES, "conf_sensitivity.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("saved conf_sensitivity.json", flush=True)

if __name__ == "__main__":
    main()
