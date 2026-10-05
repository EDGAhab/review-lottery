"""10_anomaly.py - 2020/2021 P(翻转|已录用) 异常探究 (Phase 3, 描述部分)

检验"2020 量表压缩"假说:
 A. 各年论文内分数标准差 (原始 + 按量表极差归一化)、论文均分标准差、
    分数分布熵、零方差论文占比
 B. 后验 cutpoint 间距 (潜变量单位, 残差方差=1) 与 spacing/sigma_paper
 C. P(flip|acc) 的跨年分解: 与 rho / 录用率 / mean_k 的相关 + OLS
 D 在 10b_counterfactual.py (2021 分数量化到 2020 口径后重拟合+重模拟)

输出: phase3_notes/anomaly_descriptives.csv, anomaly_decomp.json,
      figures/fig_scale_compression.png
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
POST = os.path.join(WORK, "posteriors")
OUT = os.path.join(WORK, "phase3_notes")
FIG = os.path.join(WORK, "figures")
os.makedirs(OUT, exist_ok=True)

SCALE_RANGE = {2017: 9, 2018: 9, 2019: 9, 2020: 7, 2021: 9,
               2022: 9, 2023: 9, 2024: 9, 2025: 9}

r = pd.read_parquet(os.path.join(PROC, "reviews.parquet"),
                    columns=["paper_id", "year", "score_raw"])
sim = pd.read_csv(os.path.join(WORK, "phase2_notes/sim_summary.csv"))
sim2 = sim[sim.k == 2].set_index("year")

# ---- A. 论文内/论文间离散度 ----
g = r.groupby(["year", "paper_id"])["score_raw"]
paper = g.agg(["mean", "std", "count"]).reset_index()
paper["std"] = paper["std"].fillna(0.0)
paper["range"] = paper["year"].map(SCALE_RANGE)
paper["std_norm"] = paper["std"] / paper["range"]

rows = []
for y, gy in paper.groupby("year"):
    rng = SCALE_RANGE[y]
    # 分数分布熵
    vals, counts = np.unique(r[r.year == y]["score_raw"], return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    rows.append(dict(
        year=int(y),
        n_papers=int(gy["paper_id"].nunique()),
        within_sd_raw=float(gy["std"].mean()),
        within_sd_norm=float(gy["std_norm"].mean()),
        between_mean_sd_raw=float(gy["mean"].std()),
        between_mean_sd_norm=float(gy["mean"].std() / rng),
        score_entropy=float(entropy),
        n_distinct_scores=int(len(vals)),
        frac_zero_var=float((gy["std"] == 0).mean()),
        pflip_acc_k2=float(sim2.loc[y, "pflip_acc"]),
        dis_k2=float(sim2.loc[y, "disagree"]),
    ))
desc = pd.DataFrame(rows).set_index("year")
desc.to_csv(os.path.join(OUT, "anomaly_descriptives.csv"))

# ---- B. cutpoint 间距 ----
cut_rows = []
for y in range(2017, 2026):
    d = np.load(os.path.join(POST, f"draws_{y}.npz"))
    c0 = d["c0"].reshape(-1)
    dl = d["deltas"].reshape(-1, d["deltas"].shape[-1]) if d["deltas"].shape[-1] > 0 else np.zeros((c0.size, 0))
    cuts = np.concatenate([c0[:, None], c0[:, None] + np.cumsum(dl, axis=1)], axis=1)
    spacing = np.diff(cuts, axis=1)  # 相邻 cutpoint 间距, 潜变量单位
    sp = d["sigma_paper"].reshape(-1)
    cut_rows.append(dict(
        year=y, C=cuts.shape[1] + 1,
        mean_spacing=float(spacing.mean()),
        median_spacing=float(np.median(spacing)),
        mean_spacing_over_sigma=float((spacing / sp[:, None]).mean()),
        min_spacing=float(spacing.min()), max_spacing=float(spacing.max()),
    ))
cutdf = pd.DataFrame(cut_rows).set_index("year")
cutdf.to_csv(os.path.join(OUT, "anomaly_cutpoints.csv"))

# ---- C. P(flip|acc) 跨年分解 ----
decomp = desc.join(pd.read_csv(os.path.join(WORK, "phase2_notes/rho_table.csv")).set_index("year")[["rho_mean"]])
decomp = decomp.join(pd.read_csv(os.path.join(WORK, "phase1_notes/annual_table.csv")).set_index("year")[["accept_rate", "mean_k"]])
corrs = {}
for v in ["rho_mean", "accept_rate", "mean_k", "within_sd_norm", "score_entropy"]:
    cc = decomp[["pflip_acc_k2", v]].corr().iloc[0, 1]
    corrs[v] = float(cc)
m = sm.OLS(decomp["pflip_acc_k2"], sm.add_constant(decomp[["rho_mean", "accept_rate"]])).fit()
decomp_out = {
    "pearson_corr_with_pflip_acc_k2": corrs,
    "ols_pflip_acc__rho_acceptrate": {
        var: {"coef": float(m.params[var]), "p": float(m.pvalues[var]),
              "ci95": [float(x) for x in m.conf_int().loc[var]]}
        for var in m.params.index},
    "rsquared": float(m.rsquared),
    "note": "n=9, descriptive only",
}
with open(os.path.join(OUT, "anomaly_decomp.json"), "w") as f:
    json.dump(decomp_out, f, indent=2)

# ---- 图 ----
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
yrs = desc.index.values
ax = axes[0, 0]
ax.plot(yrs, desc["within_sd_norm"], "o-", label="within-paper sd / scale range")
ax.plot(yrs, desc["between_mean_sd_norm"], "s--", label="between-paper mean sd / scale range")
ax.axvspan(2019.7, 2020.3, color="orange", alpha=0.12)
ax.set_xticks(yrs); ax.set_title("(a) normalized score dispersion")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.plot(yrs, cutdf["mean_spacing_over_sigma"], "o-", color="purple")
ax.axvspan(2019.7, 2020.3, color="orange", alpha=0.12)
ax.set_xticks(yrs); ax.set_title("(b) mean cutpoint spacing / sigma_paper")

ax = axes[1, 0]
ax.plot(yrs, desc["frac_zero_var"], "o-", color="teal")
ax.axvspan(2019.7, 2020.3, color="orange", alpha=0.12)
ax.set_xticks(yrs); ax.set_title("(c) fraction of papers with zero within-paper variance")

ax = axes[1, 1]
ax.scatter(decomp["rho_mean"], decomp["pflip_acc_k2"], c=yrs, cmap="viridis", s=80)
for y in yrs:
    ax.annotate(str(y), (decomp.loc[y, "rho_mean"], decomp.loc[y, "pflip_acc_k2"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("rho_year"); ax.set_ylabel("P(flip | accepted), k=2")
ax.set_title("(d) flip-probability vs noise level")
fig.suptitle("2020/2021 anomaly: scale-compression descriptives (orange band = 2020, 4-pt scale)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_scale_compression.png"), dpi=150)
plt.close(fig)

print(desc[["within_sd_norm", "between_mean_sd_norm", "score_entropy", "frac_zero_var", "pflip_acc_k2"]].round(4).to_string())
print()
print(cutdf[["C", "mean_spacing", "mean_spacing_over_sigma"]].round(4).to_string())
print()
print(json.dumps(decomp_out, indent=2))
print("\nDONE")
