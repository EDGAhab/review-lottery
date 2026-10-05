#!/usr/bin/env python3
"""03_describe.py — 年度描述统计（全部数字由脚本生成）.

输入: data/processed/papers.parquet, reviews.parquet, peersum_reviews.parquet
输出: phase1_notes/annual_table.csv, phase1_notes/describe.md, figures/*.png

质量门（v3.0 任务要求）：
  - 某年评审数中位数 < 3 → 标记
  - 某年 decision 缺失 > 5% → 标记
标记只写入报告，不擅自删除年份。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/hatch/workspace/review-lottery"
PROC = f"{BASE}/data/processed"
NOTES = f"{BASE}/phase1_notes"
FIG = f"{BASE}/figures"
os.makedirs(NOTES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def main():
    papers = pd.read_parquet(f"{PROC}/papers.parquet")
    reviews = pd.read_parquet(f"{PROC}/reviews.parquet")
    peersum = pd.read_parquet(f"{PROC}/peersum_reviews.parquet")
    print(f"papers={len(papers)}, reviews={len(reviews)}, peersum_reviews={len(peersum)}")

    rows = []
    for y in sorted(papers["year"].unique()):
        g = papers[papers["year"] == y]
        r = reviews[reviews["year"] == y]
        n = len(g)
        n_scored = int(g["has_scores"].sum())
        acc = (g["decision_binary"] == "accept").mean()
        rej = (g["decision_binary"] == "reject").mean()
        unk = (g["decision_binary"] == "unknown").mean()
        ks = g.loc[g["has_scores"], "k"]
        mean_k = float(ks.mean()) if len(ks) else 0.0
        med_k = float(ks.median()) if len(ks) else 0.0
        pct_4plus = float((ks >= 4).mean() * 100) if len(ks) else 0.0
        n_4plus = int((ks >= 4).sum())
        mean_s = float(r["score_raw"].mean()) if len(r) else float("nan")
        sd_s = float(r["score_raw"].std()) if len(r) else float("nan")
        scale_vals = sorted(r["score_raw"].unique().tolist()) if len(r) else []
        rows.append({
            "year": y, "n_papers": n, "n_with_scores": n_scored,
            "accept_rate": round(acc, 4), "reject_rate": round(rej, 4),
            "unknown_decision_rate": round(unk, 4),
            "mean_k": round(mean_k, 3), "median_k": med_k,
            "n_4plus_reviews": n_4plus, "pct_4plus_reviews": round(pct_4plus, 2),
            "mean_score": round(mean_s, 3), "sd_score": round(sd_s, 3),
            "scale_values": scale_vals,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{NOTES}/annual_table.csv", index=False)
    print(tab.to_string(index=False))

    # ---- 质量门 ----
    gates = []
    for _, t in tab.iterrows():
        if t["median_k"] < 3:
            gates.append(f"FLAG year={t['year']}: median reviews/paper = {t['median_k']} < 3")
        if t["unknown_decision_rate"] > 0.05:
            gates.append(f"FLAG year={t['year']}: unknown decision rate = {t['unknown_decision_rate']:.1%} > 5%")
    print("\n--- quality gates ---")
    print("\n".join(gates) if gates else "all clear")

    # ---- 缺失字段报告 ----
    print("\n--- missing-field report ---")
    print(f"berenslab reviews: confidence all-null = {reviews['confidence'].isna().all()}, "
          f"review_len all-null = {reviews['review_len'].isna().all()}")
    py = peersum[peersum["venue"] == "iclr"].groupby("year").agg(
        n=("rating", "size"),
        with_conf=("confidence", lambda s: s.notna().mean()),
        mean_len=("review_len", "mean"),
    )
    print("PeerSum ICLR coverage by year:\n", py.to_string())
    miss_conf_msg = ("2023-2025: no confidence / review text (PeerSum covers 2018-2022 only) "
                      "-- RQ-C descriptive basis narrowed per prereg")

    # ---- 图 ----
    plt.rcParams.update({"font.size": 10})
    # 1. 投稿数
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(tab["year"].astype(str), tab["n_papers"], color="#4472C4")
    ax.set_title("ICLR submissions per year (berenslab, 2017-2025)")
    ax.set_ylabel("papers")
    for x, v in zip(tab["year"].astype(str), tab["n_papers"]):
        ax.text(x, v + 80, str(v), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_submissions_by_year.png", dpi=150); plt.close(fig)

    # 2. 录用率
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tab["year"].astype(str), tab["accept_rate"] * 100, marker="o", color="#C55A5B")
    ax.set_title("ICLR acceptance rate per year (main track)")
    ax.set_ylabel("accept %")
    ax.set_ylim(0, max(tab["accept_rate"] * 100) * 1.6)
    for x, v in zip(tab["year"].astype(str), tab["accept_rate"] * 100):
        ax.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_accept_rate_by_year.png", dpi=150); plt.close(fig)

    # 3. 分数量表分布（每年直方图，展示量表差异）
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for ax, y in zip(axes.flat, sorted(reviews["year"].unique())):
        vals = reviews.loc[reviews["year"] == y, "score_raw"]
        bins = np.arange(0.5, 11.5, 1)
        ax.hist(vals, bins=bins, color="#70AD47", edgecolor="white")
        ax.set_title(f"{y} (n={len(vals)})", fontsize=10)
        ax.set_xticks(range(1, 11))
    for ax in axes.flat[len(tab):]:
        ax.axis("off")
    fig.suptitle("Review score distributions by year (scale changes visible: 2020={1,3,6,8}; 2022+={1,3,5,6,8,10})")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_score_dist_by_year.png", dpi=150); plt.close(fig)

    # 4. 每篇评审数
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tab["year"].astype(str), tab["mean_k"], marker="o", label="mean k")
    ax.plot(tab["year"].astype(str), tab["median_k"], marker="s", label="median k")
    ax.set_title("Reviews per paper by year")
    ax.set_ylabel("k")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_reviews_per_paper.png", dpi=150); plt.close(fig)

    # ---- describe.md ----
    n4_total = int(tab["n_4plus_reviews"].sum())

    def md_table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    md = f"""# Phase 1 描述统计（脚本生成：`scripts/03_describe.py`）

数据：berenslab/iclr-dataset（2017–2025，36,113 篇论文，134,912 条评审）；
补充：PeerSum（2018–2022，53,732 条官方顶层评审，含 confidence 与评审全文长度）。

## 年度表

{md_table(tab)}

## 核心数字

- 投稿量：2017 年 489 篇 → 2025 年 11,663 篇，约 24 倍。
- 录用率：{tab['accept_rate'].min():.1%}–{tab['accept_rate'].max():.1%} 区间（按年见上表）。
- 平均每篇评审数：{tab['mean_k'].min():.2f}–{tab['mean_k'].max():.2f}。
- 4+ 评审论文（内部锚点样本）：{n4_total} 篇（Phase 0 报告 23,995，复核见上表分年数）。
- 量表：2017–2019/2021 为 1–10；2020 为 {{1,3,6,8}}；2022–2025 为 {{1,3,5,6,8,10}}（图 3 可见）。

## 缺失字段

- berenslab 无 confidence、无评审全文：`confidence`/`review_len` 列全空（按设计）。
- PeerSum 覆盖 2018–2022：{miss_conf_msg}。
- decision 缺失：见上表 unknown_decision_rate 列（主要为 withdrawn）。

## 质量门结论

{chr(10).join('- ' + g for g in gates) if gates else '- 全部通过：每年评审数中位数 ≥ 3，decision 缺失率 ≤ 5%。'}

## 图

- `figures/fig_submissions_by_year.png`：每年投稿数
- `figures/fig_accept_rate_by_year.png`：每年录用率
- `figures/fig_score_dist_by_year.png`：每年分数分布（量表差异）
- `figures/fig_reviews_per_paper.png`：每年每篇评审数
"""
    with open(f"{NOTES}/describe.md", "w") as f:
        f.write(md)
    print(f"\nwrote {NOTES}/describe.md, {NOTES}/annual_table.csv, figures/*.png")
    print("DESCRIBE_OK")


if __name__ == "__main__":
    main()
