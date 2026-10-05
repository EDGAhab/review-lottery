"""04_model_fit.py — Per-year Bayesian ordered probit variance decomposition.

Model (per year y):
    level_ij in {0..C-1}  (ordinal encoding of score_raw)
    eta_i = sigma_paper * z_i,  z_i ~ Normal(0,1)   [non-centered; mu=0 FIXED for identification]
    P(level=m) = Phi(cut[m] - eta_i) - Phi(cut[m-1] - eta_i)
    cutpoints parameterized as c0, c0+cumsum(deltas), deltas>0
    rho = sigma_paper^2 / (sigma_paper^2 + 1)   [latent residual variance = 1 by probit identification]

Identification note (declared): with free cutpoints, mu is not identified
(location shift); we fix mu_year = 0, cutpoints absorb location. This is within
the preregistered model family (prereg 3.1).

Confidence: berenslab has no confidence field. Per task instruction, the
beta*conf term is fit ONLY on the PeerSum 2018-2022 subset as a sensitivity
check (see scripts/04b_conf_sensitivity.py), and the main per-year fits below
omit it. Declared deviation from prereg 3.1 formula.

Usage:
    python3 04_model_fit.py --year 2023 --pilot     # pilot: tune=500, draws=500
    python3 04_model_fit.py --year 2023 --full      # full:  tune=1000, draws=1000
    python3 04_model_fit.py --all --full            # loop all years with checkpointing
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
NOTES = os.path.join(WORK, "phase2_notes")
os.makedirs(POST, exist_ok=True)
os.makedirs(NOTES, exist_ok=True)

YEARS = list(range(2017, 2026))

def load_year(year):
    r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"), columns=["paper_id", "year", "score_raw"])
    ry = r[r["year"] == year].copy()
    scale_vals = sorted(ry["score_raw"].unique())
    level_map = {v: i for i, v in enumerate(scale_vals)}
    ry["level"] = ry["score_raw"].map(level_map).astype(int)
    pids = ry["paper_id"].unique()
    pid_map = {p: i for i, p in enumerate(pids)}
    ry["pidx"] = ry["paper_id"].map(pid_map).astype(int)
    return (ry["level"].to_numpy(), ry["pidx"].to_numpy(),
            len(pids), len(scale_vals), scale_vals)

def fit_year(year, tune, draws, chains=2, seed=0):
    import pymc as pm
    import pytensor.tensor as pt

    level, pidx, n_papers, C, scale_vals = load_year(year)
    n_rev = len(level)
    print(f"[year {year}] papers={n_papers} reviews={n_rev} categories={C} scale={scale_vals}", flush=True)

    def Phi(x):
        return 0.5 * (1 + pt.erf(x / pt.sqrt(2.0)))

    t0 = time.time()
    with pm.Model() as model:
        sigma_paper = pm.HalfNormal("sigma_paper", sigma=2.0)
        z = pm.Normal("z", 0.0, 1.0, shape=n_papers)
        c0 = pm.Normal("c0", 0.0, 5.0)
        deltas = pm.HalfNormal("deltas", sigma=2.0, shape=C - 2) if C > 2 else pt.as_tensor_variable(np.array([]))
        if C > 2:
            cutpoints = pt.concatenate([[c0], c0 + pt.cumsum(deltas)])
        else:
            cutpoints = pt.stack([c0])
        eta = sigma_paper * z[pidx]
        # ordinal log-lik: P(y=m) = Phi(cut[m]-eta) - Phi(cut[m-1]-eta)
        cuts = pt.concatenate([pt.as_tensor_variable(np.array([-np.inf])),
                               cutpoints,
                               pt.as_tensor_variable(np.array([np.inf]))])
        upper = Phi(cuts[level + 1] - eta)
        lower = Phi(cuts[level] - eta)
        pm.Potential("lik", pt.sum(pt.log(pt.maximum(upper - lower, 1e-12))))
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          init="advi", target_accept=0.9, random_seed=seed,
                          progressbar=False)
    dt = time.time() - t0
    return trace, dt, dict(n_papers=n_papers, n_rev=n_rev, C=C, scale_vals=[int(v) for v in scale_vals])

def summarize(trace):
    sp = trace.posterior["sigma_paper"].to_numpy()  # (chains, draws)
    rho = sp ** 2 / (sp ** 2 + 1.0)
    div = int(trace.sample_stats["diverging"].sum())
    # split-rhat for sigma_paper (2 chains -> 4 half-chains)
    def split_rhat(x):
        c, n = x.shape
        h = x.reshape(c * 2, n // 2)
        m = h.mean(axis=1); v = h.var(axis=1, ddof=1)
        B = (n // 2) * m.var(ddof=1); W = v.mean()
        var_hat = ((n // 2 - 1) / (n // 2)) * W + B / (n // 2)
        return float(np.sqrt(var_hat / W)) if W > 0 else float("nan")
    rhat_sp = split_rhat(sp.reshape(sp.shape[0], -1))
    return {
        "sigma_paper_mean": float(sp.mean()), "sigma_paper_sd": float(sp.std()),
        "rho_mean": float(rho.mean()),
        "rho_ci95": [float(np.quantile(rho, 0.025)), float(np.quantile(rho, 0.975))],
        "divergences": div, "split_rhat_sigma": rhat_sp,
        "n_draws": int(sp.size),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    tune, draws = (500, 500) if args.pilot else (1000, 1000)
    years = YEARS if args.all else [args.year]
    for y in years:
        ckpt = os.path.join(POST, f"draws_{y}.npz")
        if os.path.exists(ckpt):
            print(f"[year {y}] checkpoint exists, skipping", flush=True)
            continue
        trace, dt, meta = fit_year(y, tune, draws)
        post = trace.posterior
        np.savez_compressed(
            os.path.join(POST, f"draws_{y}.npz"),
            z=post["z"].to_numpy(),
            sigma_paper=post["sigma_paper"].to_numpy(),
            c0=post["c0"].to_numpy(),
            deltas=post["deltas"].to_numpy() if "deltas" in post else np.zeros((2, 1, 0)),
        )
        try:
            trace.to_netcdf(os.path.join(POST, f"trace_{y}.nc"))
        except Exception as e:
            print(f"[year {y}] netcdf save failed ({e}); npz saved", flush=True)
        s = summarize(trace)
        s.update(meta); s["sampling_seconds"] = dt
        s["tune"] = tune; s["draws"] = draws
        with open(os.path.join(NOTES, f"fit_{y}.json"), "w") as f:
            json.dump(s, f, indent=2)
        print(f"[year {y}] done in {dt/60:.1f} min. rho={s['rho_mean']:.3f} "
              f"CI=[{s['rho_ci95'][0]:.3f},{s['rho_ci95'][1]:.3f}] div={s['divergences']}", flush=True)

if __name__ == "__main__":
    main()
