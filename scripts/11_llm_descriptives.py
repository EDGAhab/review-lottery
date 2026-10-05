"""11_llm_descriptives.py - RQ-C LLM 描述性 (Phase 3)

预注册口径 (prereg §3.7): 年际间断时间序列 (评审长度、置信度、论文内方差),
以 2023 为断点, 只描述水平/斜率变化, 不做因果表述.
Liang et al. (arXiv:2403.07183) 聚合 LLM 修改比例估计: 方法需要
ChatGPT 后年份的评审**文本**; berenslab 无评论文本 (0/134912),
PeerSum 止于 2022, 故不可行 -> 诚实报告行为代理指标, 局限中写明无法归因.
(已声明的偏离, 见报告 §1.)

ITS: y_t = b0 + b1*t + b2*I(t>=2023) + b3*t*I(t>=2023); 检验 b2 (水平), b3 (斜率).
指标: 论文内分数方差 (归一化到量表极差, 全 9 年) — 唯一可做 ITS 的指标.
评审长度 / 置信度: PeerSum 2018–2022, 仅描述 (无 2023+ 数据, 不做 ITS).

输出: phase3_notes/llm_its.json, llm_descriptives.csv, figures/fig_llm_descriptives.png
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
OUT = os.path.join(WORK, "phase3_notes")
FIG = os.path.join(WORK, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 论文内方差 (归一化): 来自 10_anomaly 的描述表 ----
desc = pd.read_csv(os.path.join(OUT, "anomaly_descriptives.csv"), index_col=0)
desc["within_var_norm"] = desc["within_sd_norm"] ** 2

# ITS (year-level, n=9)
t = (desc.index - 2017).to_numpy(dtype=float)
post = (desc.index >= 2023).astype(float)
X = pd.DataFrame({"const": 1.0, "t": t, "post": post, "t_post": t * post})
y = desc["within_var_norm"].to_numpy()
m = sm.OLS(y, X).fit()
se = m.bse
t_crit = stats.t.ppf(0.975, m.df_resid)
t_pow = stats.t.ppf(0.80, m.df_resid)
its = {"n": 9, "breakpoint": 2023, "metric": "within-paper score variance (normalized by scale range)",
       "df_resid": int(m.df_resid), "rsquared": float(m.rsquared), "coefs": {}}
for v in ["t", "post", "t_post"]:
    ci = m.conf_int().loc[v].tolist()
    its["coefs"][v] = {"coef": float(m.params[v]), "se": float(se[v]),
                       "ci95": [float(ci[0]), float(ci[1])],
                       "p": float(m.pvalues[v]),
                       "mde_80pct": float((t_crit + t_pow) * se[v])}
its["interpretation"] = {
    "post_level_p_lt_05": bool(m.pvalues["post"] < 0.05),
    "post_slope_p_lt_05": bool(m.pvalues["t_post"] < 0.05),
}

# 敏感性: 去掉 2020 (4 点量表伪影)
mask = desc.index != 2020
X2 = X[mask]; y2 = y[mask]
m2 = sm.OLS(y2, X2).fit()
tc2 = stats.t.ppf(0.975, m2.df_resid); tp2 = stats.t.ppf(0.80, m2.df_resid)
its["sensitivity_drop2020"] = {
    var: {"coef": float(m2.params[var]), "p": float(m2.pvalues[var]),
          "mde_80pct": float((tc2 + tp2) * m2.bse[var])}
    for var in ["t", "post", "t_post"]}

# ---- PeerSum 2018–2022: 评审长度 / 置信度 (仅描述) ----
ps = pd.read_parquet(os.path.join(PROC, "peersum_reviews.parquet"))
print("peersum review_len sample:", ps["review_len"].head(3).tolist())
print("peersum confidence sample:", ps["confidence"].head(3).tolist())
pl = ps.groupby("year").agg(n=("review_len", "size"),
                            len_mean=("review_len", "mean"),
                            len_median=("review_len", "median"),
                            len_sd=("review_len", "std"),
                            conf_mean=("confidence", "mean"),
                            conf_median=("confidence", "median"),
                            conf_sd=("confidence", "std")).reset_index()
pl.to_csv(os.path.join(OUT, "llm_descriptives.csv"), index=False)
# 置信度分布形态: 每年各置信度取值的占比
conf_dist = ps.groupby(["year", "confidence"]).size().unstack(fill_value=0)
conf_dist = conf_dist.div(conf_dist.sum(axis=1), axis=0)
conf_dist.to_csv(os.path.join(OUT, "llm_conf_dist.csv"))

with open(os.path.join(OUT, "llm_its.json"), "w") as f:
    json.dump({"its_within_var": its,
               "peersum_years": [2018, 2019, 2020, 2021, 2022],
               "peersum_note": "no 2023+ review text or confidence available; descriptives only, no ITS",
               "liang_method": "infeasible: requires post-ChatGPT review text corpora; "
                               "berenslab has 0/134912 review texts, PeerSum ends 2022. "
                               "No per-review AI classification attempted (also prohibited by prereg)."},
              f, indent=2)

# ---- 图 ----
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
yrs = desc.index.values
ax = axes[0]
ax.plot(yrs, desc["within_var_norm"], "o-", label="observed")
xf = np.linspace(2017, 2025, 100)
tf = xf - 2017; pf = (xf >= 2023).astype(float)
yf = m.params["const"] + m.params["t"] * tf + m.params["post"] * pf + m.params["t_post"] * tf * pf
ax.plot(xf, yf, "--", color="gray", label="ITS fit")
ax.axvline(2022.5, color="red", linestyle=":", label="breakpoint 2023")
ax.set_xticks(yrs); ax.set_title("(a) within-paper variance (normalized)\n"
              f"level change p={m.pvalues['post']:.3f}, slope change p={m.pvalues['t_post']:.3f}")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(pl["year"], pl["len_mean"], "o-", color="teal")
ax.fill_between(pl["year"], pl["len_mean"] - pl["len_sd"] / np.sqrt(pl["n"]),
                pl["len_mean"] + pl["len_sd"] / np.sqrt(pl["n"]), color="teal", alpha=0.2)
ax.set_xticks(pl["year"]); ax.set_title("(b) mean review length (PeerSum 2018-2022)\nno post-2023 text available")
ax.set_ylabel("review_len (dataset units)")

ax = axes[2]
ax.plot(pl["year"], pl["conf_mean"], "o-", color="purple")
ax.fill_between(pl["year"], pl["conf_mean"] - pl["conf_sd"] / np.sqrt(pl["n"]),
                pl["conf_mean"] + pl["conf_sd"] / np.sqrt(pl["n"]), color="purple", alpha=0.2)
ax.set_xticks(pl["year"]); ax.set_title("(c) mean reviewer confidence (PeerSum 2018-2022)")
ax.set_ylabel("confidence")
fig.suptitle("RQ-C: LLM-era descriptives (no causal claims; breakpoint 2023 descriptive only)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_llm_descriptives.png"), dpi=150)
plt.close(fig)

print("ITS (within-paper variance, normalized):")
print(m.summary().tables[1].as_text())
print("\nSensitivity (drop 2020):")
print(m2.summary().tables[1].as_text())
print("\nPeerSum yearly:")
print(pl.round(1).to_string(index=False))
print("\nDONE")
