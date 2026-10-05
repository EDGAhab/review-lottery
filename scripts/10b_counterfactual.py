"""10b_counterfactual.py - 反事实小实验: 2021 分数量化到 2020 口径后重拟合+重模拟

把 2021 年 10 点量表分数按最近邻映射到 2020 年 4 点量表 {1,3,6,8}
(边界 2 / 4.5 / 7), 重拟合有序 probit, 用量化后均分/标准差重拟合
2021 年决策模型 (logistic, 保持真实 decision 为因变量: "若 AC 只看到
粗分数"的反事实), 再做双委员会模拟 (B=1000, k=2/3/4, 与 06 同种子方案).

若量化后分歧率 / P(翻转|已录用) 向 2020 的值靠拢, 则支持"量表压缩"假说.

输出: posteriors/draws_2021_q4.npz, phase3_notes/fit_2021_q4.json,
      phase3_notes/sim_2021_q4.csv, phase3_notes/counterfactual_2021.json
"""
import json, os, time
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
OUT = os.path.join(WORK, "phase3_notes")
os.makedirs(OUT, exist_ok=True)

QMAP = {1: 1, 2: 1, 3: 3, 4: 3, 5: 6, 6: 6, 7: 6, 8: 8, 9: 8, 10: 8}
YEAR = 2021
B, BATCH, KS, SEED = 1000, 100, [2, 3, 4], 7

def quantized_reviews():
    r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"),
                        columns=["paper_id", "year", "score_raw"])
    ry = r[r["year"] == YEAR].copy()
    ry["q"] = ry["score_raw"].map(QMAP).astype(int)
    assert set(ry["q"].unique()) == {1, 3, 6, 8}
    return ry

def fit_quantized(tune=1000, draws=1000, chains=2, seed=0):
    import pymc as pm
    import pytensor.tensor as pt
    ry = quantized_reviews()
    scale_vals = sorted(ry["q"].unique())
    level_map = {v: i for i, v in enumerate(scale_vals)}
    ry["level"] = ry["q"].map(level_map).astype(int)
    pids = ry["paper_id"].unique()
    pid_map = {p: i for i, p in enumerate(pids)}
    ry["pidx"] = ry["paper_id"].map(pid_map).astype(int)
    level = ry["level"].to_numpy(); pidx = ry["pidx"].to_numpy()
    n_papers, C = len(pids), len(scale_vals)

    def Phi(x):
        return 0.5 * (1 + pt.erf(x / pt.sqrt(2.0)))
    t0 = time.time()
    with pm.Model() as model:
        sigma_paper = pm.HalfNormal("sigma_paper", sigma=2.0)
        z = pm.Normal("z", 0.0, 1.0, shape=n_papers)
        c0 = pm.Normal("c0", 0.0, 5.0)
        deltas = pm.HalfNormal("deltas", sigma=2.0, shape=C - 2)
        cutpoints = pt.concatenate([[c0], c0 + pt.cumsum(deltas)])
        eta = sigma_paper * z[pidx]
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
    post = trace.posterior
    np.savez_compressed(os.path.join(POST, "draws_2021_q4.npz"),
                        z=post["z"].to_numpy(), sigma_paper=post["sigma_paper"].to_numpy(),
                        c0=post["c0"].to_numpy(), deltas=post["deltas"].to_numpy())
    sp = post["sigma_paper"].to_numpy()
    rho = sp ** 2 / (sp ** 2 + 1.0)
    div = int(trace.sample_stats["diverging"].sum())
    s = {"year": "2021_quantized_to_2020_scale", "mapping": "1,2->1; 3,4->3; 5,6,7->6; 8,9,10->8",
         "sigma_paper_mean": float(sp.mean()), "rho_mean": float(rho.mean()),
         "rho_ci95": [float(np.quantile(rho, 0.025)), float(np.quantile(rho, 0.975))],
         "divergences": div, "n_papers": n_papers, "n_rev": len(level),
         "sampling_seconds": dt}
    with open(os.path.join(OUT, "fit_2021_q4.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(f"[2021_q4] fit done in {dt/60:.1f} min. rho={s['rho_mean']:.3f} div={div}", flush=True)
    return pids

def decision_model_quantized():
    from sklearn.linear_model import LogisticRegression
    ry = quantized_reviews()
    pq = ry.groupby("paper_id")["q"].agg(["mean", "std"]).rename(
        columns={"mean": "qmean", "std": "qsd"})
    pq["qsd"] = pq["qsd"].fillna(0.0)
    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    py = p[(p["year"] == YEAR) & (p["decision_binary"] != "unknown") & (p["has_scores"])].copy()
    py["y"] = (py["decision_binary"] == "accept").astype(int)
    d = py.join(pq, on="paper_id")
    assert d[["qmean", "qsd"]].notna().all().all(), "NaN in decision-model features"
    X = d[["qmean", "qsd"]].to_numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, d["y"].to_numpy())
    print(f"[2021_q4] decision model: coef={clf.coef_[0].round(3)}, intercept={clf.intercept_.round(3)}, "
          f"n={len(d)}, acc_rate={d['y'].mean():.3f}", flush=True)
    return clf

def simulate_quantized(pids, clf):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    d = np.load(os.path.join(POST, "draws_2021_q4.npz"))
    z, sp, c0, deltas = d["z"], d["sigma_paper"], d["c0"], d["deltas"]
    D = z.shape[0] * z.shape[1]
    z = z.reshape(D, -1); sp = sp.reshape(D); c0 = c0.reshape(D)
    dl = deltas.reshape(D, -1)
    cuts = np.concatenate([c0[:, None], c0[:, None] + np.cumsum(dl, axis=1)], axis=1)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(D, size=min(B, D), replace=False)
    # cohort: 已知 decision, 顺序与 pids 一致
    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    py = p[p["year"] == YEAR].set_index("paper_id")
    keep = [pid for pid in pids if py.loc[pid, "decision_binary"] != "unknown"]
    pos = {pid: i for i, pid in enumerate(pids)}
    keep_idx = np.array([pos[pid] for pid in keep])
    theta = sp[idx][:, None] * z[idx][:, keep_idx]
    cuts = cuts[idx]
    scale = np.array([1, 3, 6, 8])
    rng2 = np.random.default_rng(SEED + YEAR)
    rows = []
    for k in KS:
        dis_l, pfa_l = [], []
        for s in range(0, theta.shape[0], BATCH):
            sl = slice(s, min(s + BATCH, theta.shape[0]))
            th = theta[sl]
            Bb, P = th.shape
            lat1 = th[:, :, None] + rng2.standard_normal((Bb, P, k))
            lat2 = th[:, :, None] + rng2.standard_normal((Bb, P, k))
            lv1 = (lat1[..., None] > cuts[sl][:, None, None, :]).sum(-1)
            lv2 = (lat2[..., None] > cuts[sl][:, None, None, :]).sum(-1)
            s1, s2 = scale[lv1], scale[lv2]
            m1, m2 = s1.mean(-1), s2.mean(-1)
            sd1 = s1.std(-1, ddof=1); sd2 = s2.std(-1, ddof=1)
            pr1 = sigmoid(clf.coef_[0][0] * m1 + clf.coef_[0][1] * sd1 + clf.intercept_[0])
            pr2 = sigmoid(clf.coef_[0][0] * m2 + clf.coef_[0][1] * sd2 + clf.intercept_[0])
            a1, a2 = pr1 >= 0.5, pr2 >= 0.5
            dis = (a1 != a2).mean(axis=1)
            with np.errstate(invalid="ignore"):
                pfa = np.where(a1.sum(1) > 0, ((a1) & (~a2)).sum(1) / a1.sum(1), np.nan)
            dis_l.append(dis); pfa_l.append(pfa)
        dis = np.concatenate(dis_l)
        pfa = np.concatenate(pfa_l); pfa = pfa[~np.isnan(pfa)]
        rows.append(dict(k=k, disagree=float(dis.mean()),
                         d_lo=float(np.quantile(dis, 0.025)), d_hi=float(np.quantile(dis, 0.975)),
                         pflip_acc=float(pfa.mean()),
                         pfa_lo=float(np.quantile(pfa, 0.025)), pfa_hi=float(np.quantile(pfa, 0.975))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sim_2021_q4.csv"), index=False)
    return df

def main():
    ckpt = os.path.join(POST, "draws_2021_q4.npz")
    if os.path.exists(ckpt):
        print("[2021_q4] posterior checkpoint exists, skipping refit", flush=True)
        ry = quantized_reviews()
        pids = ry["paper_id"].unique()
    else:
        pids = fit_quantized()
    clf = decision_model_quantized()
    df = simulate_quantized(pids, clf)
    # 与原始 2021 对比
    orig = pd.read_csv(os.path.join(WORK, "phase2_notes/sim_summary.csv"))
    o = orig[(orig.year == 2021)][["k", "disagree", "pflip_acc"]].set_index("k")
    cmp = df.set_index("k").join(o, rsuffix="_orig2021")
    cmp["delta_disagree"] = cmp["disagree"] - cmp["disagree_orig2021"]
    cmp["delta_pflip_acc"] = cmp["pflip_acc"] - cmp["pflip_acc_orig2021"]
    cmp.to_csv(os.path.join(OUT, "counterfactual_2021.csv"))
    print(cmp.round(4).to_string())
    print("\nDONE -> phase3_notes/sim_2021_q4.csv, counterfactual_2021.csv")

if __name__ == "__main__":
    main()
