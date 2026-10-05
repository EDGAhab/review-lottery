#!/usr/bin/env python3
"""01_load_clean.py — 加载层清洗（v3.0 身份字段检查结论落实）.

- 加载 berenslab/iclr-dataset (data/raw/iclr26v1.parquet)
- 加载时立即删除 author_ids / authors 列（含约 12% 作者邮箱，作者 PII）
- 断言：输出中不得含任何可识别评审员身份的列；不做跨论文评审员关联
- 输出: data/processed/berenslab_clean.parquet（PII-free）
"""
import os
import re
import pandas as pd

RAW = "/home/hatch/workspace/review-lottery/data/raw"
OUT = "/home/hatch/workspace/review-lottery/data/processed"
os.makedirs(OUT, exist_ok=True)

PII_COLS = ["author_ids", "authors"]

# 评审员身份相关的列名模式：若命中则报错停下（人工判定后再处理）
REVIEWER_ID_PATTERNS = re.compile(
    r"reviewer.*(id|name|email)|signature|(?<!author_)profile", re.I
)

def main():
    df = pd.read_parquet(f"{RAW}/iclr26v1.parquet")
    print(f"raw: {len(df)} papers, columns={list(df.columns)}")

    # 1. 删除作者 PII 列
    drop = [c for c in PII_COLS if c in df.columns]
    df = df.drop(columns=drop)
    print(f"dropped PII columns: {drop}")

    # 2. 评审员身份字段检查：列名扫描
    suspect = [c for c in df.columns if REVIEWER_ID_PATTERNS.search(c)]
    assert not suspect, f"BLOCKED: reviewer-identity-like columns found: {suspect}"
    print("column scan: no reviewer-identity fields")

    # 3. 值扫描：title/abstract 中找邮箱（抽样 20000）
    email_pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    n_email = sum(
        1 for t in df["title"].head(20000) if email_pat.search(str(t))
    )
    print(f"titles with email-like strings (20k sample): {n_email} (informational only)")

    # 4. 红线重申：不构建任何评审员跨论文标识
    #    scores 列仅为每篇论文的分数列表，无评审员 ID，不做关联。

    out = f"{OUT}/berenslab_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, columns={list(df.columns)}")
    print("LOAD_CLEAN_OK")

if __name__ == "__main__":
    main()
