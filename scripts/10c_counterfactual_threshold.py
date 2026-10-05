"""10c_counterfactual_threshold.py - 反事实变体: 阈值规则 (接受率精确校准)

10b 的 logistic 固定 0.5 截断在反事实世界中导致模拟接受率坍缩
(9.9% vs 实际 33.1%), 不可比. 确定性阈值在粗量表下也会因巨量
ties 无法精确命中目标接受率. 本变体用随机化 tie-break:
mean > t* 全录, mean == t* 以概率 q 录用, 使模拟接受率精确等于
经验接受率 (2021: 33.1%), 两世界可比.

输出: phase3_notes/counterfactual_2021_threshold.csv
"""
import os
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
OUT = os.path.join(WORK, "phase3_notes")

B, BATCH, KS, SEED = 1000, 100, [2, 3, 4], 7
YEAR = 2021
ACC_RATE_2021 = 0.331  # 经验接受率

def simulate_means(post_path, scale, keep_idx, n_papers_all):
    d = np.load(post_path)
    z, sp, c0, deltas = d["z"], d["sigma_paper"], d["c0"], d["deltas"]
    D = z.shape[0] * z.shape[1]
    z = z.reshape(D, -1); sp = sp.reshape(D); c0 = c0.reshape(D)
    dl = deltas.reshape(D, -1)
    cuts = np.concatenate([c0[:, None], c0[:, None] + np.cumsum(dl, axis=1)], axis=1)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(D, size=min(B, D), replace=False)
    theta = sp[idx][:, None] * z[idx][:, keep_idx]
    cuts = cuts[idx]
    rng2 = np.random.default_rng(SEED + YEAR)
    m1_all, m2_all = [], []
    for s in range(0, theta.shape[0], BATCH):
        sl = slice(s, min(s + BATCH, theta.shape[0]))
        th = theta[sl]
        Bb, P = th.shape
        out = []
        for rep in range(2):
            lat = th[:, :, None] + rng2.standard_normal((Bb, P, KS[-1]))
            lv = (lat[..., None] > cuts[sl][:, None, None, :]).sum(-1)
            s_all = scale[lv]  # (Bb,P,4)
            out.append(s_all)
        m1_all.append(out[0])
        m2_all.append(out[1])
    # 为每个 k 取前 k 个评审的均分
    res = {}
    for k in KS:
        res[k] = (np.concatenate([a[:, :, :k].mean(-1) for a in m1_all]),
                  np.concatenate([a[:, :, :k].mean(-1) for a in m2_all]))
    return res

def cohort_idx(year):
    r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"), columns=["paper_id", "year"])
    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    pids = r[r["year"] == year]["paper_id"].unique()
    py = p[(p["year"] == year)].set_index("paper_id")
    keep = [pid for pid in pids if py.loc[pid, "decision_binary"] != "unknown"]
    pos = {pid: i for i, pid in enumerate(pids)}
    return np.array([pos[pid] for pid in keep])

def run_variant(name, post_path, scale):
    ki = cohort_idx(YEAR)
    means = simulate_means(post_path, scale, ki, None)
    rows = []
    for k in KS:
        m1, m2 = means[k]
        pooled = np.concatenate([m1.ravel(), m2.ravel()])
        # 随机化 tie-break: t* 使 P(>t*) <= target <= P(>=t*)
        vals = np.unique(pooled)
        t_star = vals[np.searchsorted(-np.array([ (pooled > v).mean() for v in vals ]), -ACC_RATE_2021)]
        p_gt = (pooled > t_star).mean()
        p_eq = (pooled == t_star).mean()
        q = (ACC_RATE_2021 - p_gt) / p_eq if p_eq > 0 else 0.0
        q = float(np.clip(q, 0, 1))
        rngt = np.random.default_rng(SEED + 999 + k)
        def decide(m):
            acc = m > t_star
            tie = (m == t_star) & (rngt.random(m.shape) < q)
            return acc | tie
        a1, a2 = decide(m1), decide(m2)
        dis = (a1 != a2).mean(axis=1)
        with np.errstate(invalid="ignore"):
            pfa = np.where(a1.sum(1) > 0, ((a1) & (~a2)).sum(1) / a1.sum(1), np.nan)
        pfa = pfa[~np.isnan(pfa)]
        rows.append(dict(variant=name, k=k, threshold=round(float(t_star), 4),
                         tiebreak_q=round(q, 4),
                         sim_accept_rate=round(float(np.concatenate([a1, a2]).mean()), 4),
                         disagree=round(float(dis.mean()), 4),
                         d_lo=round(float(np.quantile(dis, 0.025)), 4),
                         d_hi=round(float(np.quantile(dis, 0.975)), 4),
                         pflip_acc=round(float(pfa.mean()), 4),
                         pfa_lo=round(float(np.quantile(pfa, 0.025)), 4),
                         pfa_hi=round(float(np.quantile(pfa, 0.975)), 4)))
    return pd.DataFrame(rows)

ki = cohort_idx(YEAR)
df_q4 = run_variant("2021_quantized_4pt", os.path.join(POST, "draws_2021_q4.npz"), np.array([1, 3, 6, 8]))
r1 = pd.read_parquet(os.path.join(PROC, "reviews.parquet"), columns=["paper_id", "year", "score_raw"])
scale10 = np.array(sorted(r1[r1.year == YEAR]["score_raw"].unique()))
df_orig = run_variant("2021_original_10pt", os.path.join(POST, "draws_2021.npz"), scale10)
out = pd.concat([df_orig, df_q4], ignore_index=True)
out.to_csv(os.path.join(OUT, "counterfactual_2021_threshold.csv"), index=False)
print(out.to_string(index=False))
print("\nDONE -> phase3_notes/counterfactual_2021_threshold.csv")
