"""09_longitudinal.py - RQ-B 纵向回归 (Phase 3)

预注册口径 (prereg §3.6): 回归 分歧率 ~ log(投稿量) + 每篇评审数 + 量表版本;
NeurIPS 两点 (2014: 25.9%, 2021: 23.0%) 画参照.
正式趋势检验: rho ~ year; 零结果如实报告并报告检验功效.

量表版本实查 (phase1_notes/describe.md scale_values, 已验证):
  scale10: 2017-2019, 2021  (1-10)
  scale4 : 2020             ({1,3,6,8})
  scale6 : 2022-2025        ({1,3,5,6,8,10})

输出: phase3_notes/longitudinal_table.csv, regression_results.json,
      figures/fig_rho_trend.png, figures/fig_disagreement_trend.png
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO = "phase2_notes/rho_table.csv"
SIM = "phase2_notes/sim_summary.csv"
ANN = "phase1_notes/annual_table.csv"
OUT = "phase3_notes"
FIG = "figures"

import os
os.makedirs(OUT, exist_ok=True)

rho = pd.read_csv(RHO)
sim = pd.read_csv(SIM)
ann = pd.read_csv(ANN)

sim2 = sim[sim.k == 2].set_index("year")
sim3 = sim[sim.k == 3].set_index("year")

df = rho.set_index("year")[["rho_mean", "rho_lo", "rho_hi"]].copy()
df["dis_k2"] = sim2["disagree"]
df["dis_k2_lo"] = sim2["d_lo"]
df["dis_k2_hi"] = sim2["d_hi"]
df["dis_k3"] = sim3["disagree"]
df["dis_k3_lo"] = sim3["d_lo"]
df["dis_k3_hi"] = sim3["d_hi"]
df["pflip_acc_k2"] = sim2["pflip_acc"]
ann_i = ann.set_index("year")
df["n_sub"] = ann_i["n_papers"]
df["log_n_sub"] = np.log(ann_i["n_papers"])
df["mean_k"] = ann_i["mean_k"]
df["accept_rate"] = ann_i["accept_rate"]
# 量表哑变量 (实查): 2020=4点, 2022-2025=6点, 其余=10点(参照)
df["scale4"] = (df.index == 2020).astype(int)
df["scale6"] = (df.index >= 2022).astype(int)
df["year"] = df.index
df.to_csv(f"{OUT}/longitudinal_table.csv")

results = {"n_years": len(df), "scale_coding": "scale10: 2017-2019,2021 (ref); scale4: 2020; scale6: 2022-2025"}

def ols_report(y, X, name):
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc).fit()
    out = {
        "n": int(len(y)), "df_resid": int(m.df_resid), "rsquared": float(m.rsquared),
        "coefs": {},
    }
    for var in Xc.columns:
        ci = m.conf_int().loc[var].tolist()
        out["coefs"][var] = {
            "coef": float(m.params[var]), "se": float(m.bse[var]),
            "ci95": [float(ci[0]), float(ci[1])],
            "t": float(m.tvalues[var]), "p": float(m.pvalues[var]),
        }
    # VIF (不含截距)
    vifs = {}
    for j, col in enumerate(X.columns):
        vifs[col] = float(variance_inflation_factor(X.values, j))
    out["vif"] = vifs
    results[name] = out
    return m

Xs = df[["log_n_sub", "mean_k", "scale4", "scale6"]]

# --- 预注册指定的主回归: 分歧率 ~ log(投稿量) + 每篇评审数 + 量表版本 ---
m_k2 = ols_report(df["dis_k2"], Xs, "dis_k2__logn_meank_scale")
m_k3 = ols_report(df["dis_k3"], Xs, "dis_k3__logn_meank_scale")
m_rho = ols_report(df["rho_mean"], Xs, "rho__logn_meank_scale")

# --- 预注册判定: 显著系数 (p<0.05) 的 LOO 符号稳健性 (去任一年重跑不变号) ---
def loo_sign_stable(y, X, var):
    base = float(sm.OLS(y, sm.add_constant(X)).fit().params[var])
    signs = []
    for yr in y.index:
        mm = sm.OLS(y.drop(yr), sm.add_constant(X.drop(yr))).fit()
        signs.append(np.sign(mm.params[var]))
    return {"base_sign": float(np.sign(base)),
            "loo_signs": [float(s) for s in signs],
            "stable": bool(all(s == np.sign(base) and s != 0 for s in signs))}

loo = {}
for rname, y in [("dis_k2", df["dis_k2"]), ("dis_k3", df["dis_k3"]), ("rho", df["rho_mean"])]:
    loo[rname] = {}
    for var, info in results[f"{rname}__logn_meank_scale"]["coefs"].items():
        if var == "const":
            continue
        if info["p"] < 0.05:
            loo[rname][var] = loo_sign_stable(y, Xs, var)
results["loo_sign_stability_significant_coefs"] = loo

# --- 正式趋势检验: rho ~ year, dis ~ year ---
def trend_test(y, name):
    X = sm.add_constant(df[["year"]])
    m = sm.OLS(y, X).fit()
    se = float(m.bse["year"])
    # 功效: 80% power, alpha=0.05 双侧, df=n-2
    t_crit = stats.t.ppf(0.975, m.df_resid)
    t_pow = stats.t.ppf(0.80, m.df_resid)
    mde = (t_crit + t_pow) * se
    # 实际观测斜率下的功效
    ncp = abs(float(m.params["year"])) / se
    power = float(1 - stats.nct.cdf(t_crit, m.df_resid, ncp) + stats.nct.cdf(-t_crit, m.df_resid, ncp))
    results[name] = {
        "slope_per_year": float(m.params["year"]),
        "se": se,
        "ci95": [float(x) for x in m.conf_int().loc["year"]],
        "p": float(m.pvalues["year"]),
        "rsquared": float(m.rsquared),
        "mde_80pct_power": float(mde),
        "power_at_observed_slope": power,
        "conclusion": "trend" if float(m.pvalues["year"]) < 0.05 else "no_trend",
    }

trend_test(df["rho_mean"], "trend_rho__year")
trend_test(df["dis_k2"], "trend_dis_k2__year")
trend_test(df["dis_k3"], "trend_dis_k3__year")

# --- 图: fig_rho_trend.png ---
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9, 5))
yrs = df.index.values
ax.errorbar(yrs, df["rho_mean"], yerr=[df["rho_mean"] - df["rho_lo"], df["rho_hi"] - df["rho_mean"]],
            fmt="o-", capsize=4, label="rho (95% CI)")
tr = results["trend_rho__year"]
xs = np.linspace(yrs.min(), yrs.max(), 50)
# 趋势线 (OLS 拟合)
b0 = sm.OLS(df["rho_mean"], sm.add_constant(df[["year"]])).fit().params
ax.plot(xs, b0["const"] + b0["year"] * xs, "--", color="gray",
        label=f"OLS trend: {tr['slope_per_year']:+.4f}/yr, p={tr['p']:.3f} ({tr['conclusion']})")
ax.axvspan(2019.7, 2020.3, color="orange", alpha=0.12, label="2020: 4-pt scale")
ax.set_xlabel("Year")
ax.set_ylabel("rho = paper-quality share of latent score variance")
ax.set_title("RQ-B: rho over time (ICLR 2017-2025)")
ax.set_xticks(yrs)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_rho_trend.png", dpi=150)
plt.close(fig)

# --- 图: fig_disagreement_trend.png (含 NeurIPS 参照) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(yrs, df["dis_k2"], yerr=[df["dis_k2"] - df["dis_k2_lo"], df["dis_k2_hi"] - df["dis_k2"]],
            fmt="o-", capsize=4, label="k=2 (95% CI)")
ax.errorbar(yrs, df["dis_k3"], yerr=[df["dis_k3"] - df["dis_k3_lo"], df["dis_k3_hi"] - df["dis_k3"]],
            fmt="s--", capsize=4, label="k=3 (95% CI)")
ax.axhline(0.259, color="red", linestyle=":", linewidth=2, label="NeurIPS 2014: 25.9% (n=166)")
ax.axhline(0.230, color="green", linestyle=":", linewidth=2, label="NeurIPS 2021: 23.0% (n=882)")
ax.axvspan(2019.7, 2020.3, color="orange", alpha=0.12)
ax.set_xlabel("Year")
ax.set_ylabel("Two-committee disagreement rate")
ax.set_title("RQ-B: simulated disagreement over time, with NeurIPS anchors")
ax.set_xticks(yrs)
ax.set_ylim(0.15, 0.33)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_disagreement_trend.png", dpi=150)
plt.close(fig)

# --- 双变量趋势的 LOO 稳健性 (去任一年重跑) ---
loo_tr = {}
for yname, y in [("dis_k2", df["dis_k2"]), ("dis_k3", df["dis_k3"]), ("rho_mean", df["rho_mean"])]:
    rows = []
    for yr in df.index:
        d2 = df.drop(yr)
        mm = sm.OLS(d2[yname], sm.add_constant(d2[["year"]])).fit()
        rows.append({"dropped": int(yr), "slope": float(mm.params["year"]),
                     "p": float(mm.pvalues["year"])})
    loo_tr[yname] = rows
results["loo_trend_year"] = loo_tr

with open(f"{OUT}/regression_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== 主回归: dis_k2 ~ log_n + mean_k + scale ===")
print(m_k2.summary().tables[1].as_text())
print("\n=== rho ~ year 趋势检验 ===")
print(json.dumps(results["trend_rho__year"], indent=2))
print("\n=== dis_k2 ~ year 趋势检验 ===")
print(json.dumps(results["trend_dis_k2__year"], indent=2))
print("\nVIF:", results["dis_k2__logn_meank_scale"]["vif"])
print("\nLOO:", json.dumps(loo, indent=2)[:1500])
print("\nDONE -> phase3_notes/regression_results.json")
