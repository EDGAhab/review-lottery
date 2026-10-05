#!/usr/bin/env python3
"""02_build_dataset.py — 量表映射与字段统一（G3 结论落实）.

输入: data/processed/berenslab_clean.parquet, data/raw/peersum_huggingface.jsonl
输出:
  data/processed/reviews.parquet         评审级（berenslab 主力）
  data/processed/papers.parquet          论文级（berenslab，2017–2025）
  data/processed/peersum_reviews.parquet  评审级补充（PeerSum，2018–2022，含 confidence/review_len）

decision 归一化规则（14 种变体 + 空）：
  detail                                  binary   说明
  accept-oral                             accept   Oral / Talk / notable-top-5%
  accept-spotlight                        accept   Spotlight / notable-top-25%
  accept-poster                           accept   Poster
  reject                                  reject
  desk-reject                             reject    desk rejection = editorial reject
  workshop-invite                         reject   Invite to Workshop Track：未被主会录用（已记录此判断，Phase 2 可做敏感性分析）
  withdrawn                               unknown  无录用决定，排除
  unknown                                 unknown  空/无法识别，排除

量表（G3 已验证，脚本内断言）：
  2017–2019, 2021: 1–10；2020: {1,3,6,8}；2022–2025: {1,3,5,6,8,10}
score_std = 年内 z-score（当年全部评审分数的 mean/std）。
只用 2017–2025（2026 无分数，跳过）。
"""
import json
import os
import re
import numpy as np
import pandas as pd

BASE = "/home/hatch/workspace/review-lottery"
PROC = f"{BASE}/data/processed"
RAW = f"{BASE}/data/raw"
os.makedirs(PROC, exist_ok=True)

YEARS = list(range(2017, 2026))
SCALE = {
    2017: set(range(1, 11)), 2018: set(range(1, 11)), 2019: set(range(1, 11)),
    2020: {1, 3, 6, 8}, 2021: set(range(1, 11)),
    2022: {1, 3, 5, 6, 8, 10}, 2023: {1, 3, 5, 6, 8, 10},
    2024: {1, 3, 5, 6, 8, 10}, 2025: {1, 3, 5, 6, 8, 10},
}


def normalize_decision(d):
    d = (d or "").strip()
    dl = d.lower()
    if not dl:
        return ("unknown", "unknown")
    if "workshop" in dl:
        return ("workshop-invite", "reject")
    if "desk" in dl:
        return ("desk-reject", "reject")
    if "withdraw" in dl:
        return ("withdrawn", "unknown")
    if "reject" in dl:
        return ("reject", "reject")
    if "oral" in dl or "talk" in dl or "notable-top-5" in dl:
        return ("accept-oral", "accept")
    if "spotlight" in dl or "notable-top-25" in dl:
        return ("accept-spotlight", "accept")
    if "poster" in dl:
        return ("accept-poster", "accept")
    if "accept" in dl:
        return ("accept-unknown-tier", "accept")
    return ("unknown", "unknown")


def test_normalize_decision():
    cases = {
        "": ("unknown", "unknown"),
        "Reject": ("reject", "reject"),
        "Withdrawn": ("withdrawn", "unknown"),
        "Accept (Poster)": ("accept-poster", "accept"),
        "Accept (poster)": ("accept-poster", "accept"),
        "Accept: poster": ("accept-poster", "accept"),
        "Accept (Spotlight)": ("accept-spotlight", "accept"),
        "Accept (spotlight)": ("accept-spotlight", "accept"),
        "Accept: notable-top-25%": ("accept-spotlight", "accept"),
        "Accept (Oral)": ("accept-oral", "accept"),
        "Accept (oral)": ("accept-oral", "accept"),
        "Accept (Talk)": ("accept-oral", "accept"),
        "Accept: notable-top-5%": ("accept-oral", "accept"),
        "Desk rejected": ("desk-reject", "reject"),
        "Invite to Workshop Track": ("workshop-invite", "reject"),
    }
    for raw, expected in cases.items():
        got = normalize_decision(raw)
        assert got == expected, f"FAIL {raw!r}: got {got}, want {expected}"
    print(f"unit test: all {len(cases)} decision variants map correctly")


def build_berenslab():
    df = pd.read_parquet(f"{PROC}/berenslab_clean.parquet")
    df = df[df["year"].isin(YEARS)].copy()
    print(f"berenslab 2017-2025: {len(df)} papers")

    # decision 归一化
    det_bin = df["decision"].apply(normalize_decision)
    df["decision_detail"] = [x[0] for x in det_bin]
    df["decision_binary"] = [x[1] for x in det_bin]

    # 量表断言：每年的观测分数值集必须等于预期量表
    for y in YEARS:
        g = df[df["year"] == y]
        vals = set(s for arr in g["scores"] for s in arr)
        assert vals == SCALE[y], f"scale mismatch {y}: observed {sorted(vals)}"
    print("scale assertion passed for all years 2017-2025")

    # 年内 z-score 参数
    year_stats = {}
    for y in YEARS:
        allscores = [s for arr in df[df["year"] == y]["scores"] for s in arr]
        year_stats[y] = (float(np.mean(allscores)), float(np.std(allscores)))
        print(f"  year {y}: n_scores={len(allscores)}, mean={year_stats[y][0]:.3f}, sd={year_stats[y][1]:.3f}")

    # reviews.parquet（评审级）
    rows = []
    for _, r in df.iterrows():
        mu, sd = year_stats[r["year"]]
        for j, s in enumerate(r["scores"]):
            rows.append({
                "paper_id": r["id"], "year": r["year"], "review_idx": j,
                "score_raw": int(s),
                "score_std": (s - mu) / sd if sd > 0 else 0.0,
                "confidence": None, "review_len": None,
                "decision_detail": r["decision_detail"],
                "decision_binary": r["decision_binary"],
                "source": "berenslab",
            })
    reviews = pd.DataFrame(rows)
    reviews.to_parquet(f"{PROC}/reviews.parquet", index=False)
    print(f"reviews.parquet: {len(reviews)} rows")

    # papers.parquet（论文级）
    prows = []
    for _, r in df.iterrows():
        scores = list(r["scores"])
        k = len(scores)
        labels = r["labels"] if r["labels"] is not None else []
        prows.append({
            "paper_id": r["id"], "year": r["year"],
            "area": labels[0] if len(labels) else None,
            "labels_joined": ";".join(labels) if len(labels) else None,
            "decision_detail": r["decision_detail"],
            "decision_binary": r["decision_binary"],
            "k": k,
            "mean_score": float(np.mean(scores)) if k else None,
            "sd_score": float(np.std(scores, ddof=1)) if k >= 2 else None,
            "has_scores": k > 0,
        })
    papers = pd.DataFrame(prows)
    papers.to_parquet(f"{PROC}/papers.parquet", index=False)
    print(f"papers.parquet: {len(papers)} rows")
    return df


def build_peersum():
    """PeerSum 补充表：2018–2022 官方顶层评审，含 confidence 与 review_len。"""
    rows = []
    with open(f"{RAW}/peersum_huggingface.jsonl") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"([a-z]+)_(\d{4})_", r["paper_id"])
            if not m:
                continue
            venue_raw, year = m.group(1), int(m.group(2))
            venue = "iclr" if venue_raw == "iclr" else ("neurips" if venue_raw in ("nips", "neurips") else venue_raw)
            pid = r["paper_id"]
            for j, (rt, cf, w, rto, content) in enumerate(zip(
                    r["review_ratings"], r["review_confidences"],
                    r["review_writers"], r["review_reply_tos"], r["review_contents"])):
                if w != "official_reviewer" or rto != pid:
                    continue
                if rt == -1:  # 非实质评审条目
                    continue
                rows.append({
                    "paper_id": pid, "venue": venue, "year": year,
                    "review_idx": j,
                    "rating": int(rt),
                    "confidence": int(cf) if cf != -1 else None,
                    "review_len": len(content) if content else 0,
                    "paper_acceptance": r["paper_acceptance"],
                    "source": "peersum",
                })
    pdf = pd.DataFrame(rows)
    pdf.to_parquet(f"{PROC}/peersum_reviews.parquet", index=False)
    print(f"peersum_reviews.parquet: {len(pdf)} rows, "
          f"years={sorted(pdf['year'].unique().tolist())}, "
          f"venues={sorted(pdf['venue'].unique().tolist())}")
    return pdf


def manual_spot_check(df, n=50):
    rng = np.random.default_rng(42)
    sample = df.sample(n, random_state=rng)
    print("\n--- manual spot check (50 random papers: raw -> detail/binary) ---")
    for _, r in sample.iterrows():
        print(f"  {r['decision']!r:35s} -> {r['decision_detail']}/{r['decision_binary']}")
    print("--- end spot check: eyeball above for mis-mapping ---")


def main():
    test_normalize_decision()
    df = build_berenslab()
    manual_spot_check(df)
    build_peersum()
    print("BUILD_DATASET_OK")


if __name__ == "__main__":
    main()
