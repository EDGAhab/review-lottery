#!/usr/bin/env python3
"""Phase 0 checks: G1 (NeurIPS 2021 coverage), G3 (scale comparability),
internal-anchor sample size (papers with 4+ reviews per year),
and identity-field scan. All numbers printed are script-generated.
Outputs: phase0_notes/data_coverage.md (numbers filled by this script).
"""
import json, re, sys
from collections import Counter, defaultdict
import pandas as pd

RAW = "/home/hatch/workspace/review-lottery/data/raw"

print("=" * 70)
print("PART A: berenslab/iclr-dataset (iclr26v1.parquet)")
print("=" * 70)
df = pd.read_parquet(f"{RAW}/iclr26v1.parquet")
print(f"total papers: {len(df)}")
print(f"columns: {list(df.columns)}")
df["n_reviews"] = df["scores"].apply(lambda s: len(s) if s is not None else 0)
df["has_scores"] = df["n_reviews"] > 0

# Normalize decisions
def norm_dec(d):
    d = (d or "").strip().lower()
    if "oral" in d: return "accept-oral"
    if "spotlight" in d or "notable" in d: return "accept-spotlight"
    if "poster" in d: return "accept-poster"
    if "reject" in d: return "reject"
    if "withdraw" in d: return "withdrawn"
    if "desk" in d: return "desk-reject"
    return "unknown/none"
df["dec_norm"] = df["decision"].apply(norm_dec)

print("\n--- per-year counts, decisions, review stats ---")
print(f"{'year':<6}{'n':>7}{'acc%':>7}{'%w4+rev':>9}{'mean_k':>8}{'score_min-max':>14}")
anchor = {}
for y, g in df.groupby("year"):
    n = len(g)
    acc = (g["dec_norm"].str.startswith("accept")).mean() * 100
    gk = g[g["has_scores"]]
    w4 = (gk["n_reviews"] >= 4).sum()
    w4pct = w4 / len(gk) * 100 if len(gk) else 0
    meank = gk["n_reviews"].mean() if len(gk) else 0
    allscores = [s for arr in gk["scores"] for s in arr]
    smin, smax = (min(allscores), max(allscores)) if allscores else ("-", "-")
    anchor[y] = {"n_papers": n, "n_with_scores": len(gk), "n_4plus": int(w4)}
    print(f"{y:<6}{n:>7}{acc:>6.1f}%{w4pct:>8.1f}%{meank:>8.2f}   {smin}-{smax}")

print("\n--- score value distributions per year (for G3 scale mapping) ---")
for y, g in df.groupby("year"):
    gk = g[g["has_scores"]]
    vals = sorted(set(s for arr in gk["scores"] for s in arr))
    print(f"{y}: values={vals}")

print("\n--- decision label variants (normalization check) ---")
print(df["decision"].value_counts().head(20).to_string())

print("\n" + "=" * 70)
print("PART B: oaimli/PeerSum (peersum_huggingface.jsonl)")
print("=" * 70)
venue_year = Counter()
conf_avail = Counter()
n_reviews_dist = Counter()
neurips2021 = {"total": 0, "accepted": 0, "rejected": 0, "with_conf": 0}
with open(f"{RAW}/peersum_huggingface.jsonl") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        pid = r["paper_id"]
        m = re.match(r"([a-z]+)_(\d{4})_", pid)
        vy = f"{m.group(1)}_{m.group(2)}" if m else "unknown"
        venue_year[vy] += 1
        # official reviews only: reply directly to paper
        off = [(rt, cf) for rt, cf, w in zip(r["review_ratings"], r["review_confidences"], r["review_writers"])
               if w == "official_reviewer"]
        # filter to top-level reviews (reply_to == paper_id)
        toplevel = [(rt, cf) for (rt, cf), rto, w in zip(
            zip(r["review_ratings"], r["review_confidences"]), r["review_reply_tos"], r["review_writers"])
            if rto == pid and w == "official_reviewer"]
        n_reviews_dist[(vy, len(toplevel))] += 1
        if any(c is not None for _, c in toplevel):
            conf_avail[vy] += 1
        if vy == "neurips_2021":
            neurips2021["total"] += 1
            acc = r["paper_acceptance"] or ""
            if "reject" in acc.lower() or "withdraw" in acc.lower():
                neurips2021["rejected"] += 1
            elif acc:
                neurips2021["accepted"] += 1
            if any(c is not None for _, c in toplevel):
                neurips2021["with_conf"] += 1
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1} records", file=sys.stderr)

print("\nvenue_year counts:")
for vy in sorted(venue_year):
    print(f"  {vy}: {venue_year[vy]}")
print("\nrecords with any confidence value:", sum(conf_avail.values()))
print("\nNeurIPS 2021 in PeerSum (G1):", neurips2021)
print("\nreview-count distribution sample (venue_year, k -> n):")
for k in sorted(n_reviews_dist)[:20]:
    print(f"  {k}: {n_reviews_dist[k]}")

print("\n" + "=" * 70)
print("PART C: identity-field scan (v3.0 requirement)")
print("=" * 70)
# Column-name scan
suspect_cols = [c for c in df.columns
                if re.search(r"signatur|profile|email|name|author_id|reviewer", c, re.I)]
print("berenslab suspect columns:", suspect_cols)
# author_ids: check format (are they anonymized hashes or emails?)
sample_aids = df["author_ids"].dropna().head(3).tolist()
print("author_ids sample:", [str(a)[:120] for a in sample_aids])
# value scan on a sample of abstracts/titles for email patterns
email_pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
n_email = 0
for t in df["title"].head(20000):
    if email_pat.search(str(t)):
        n_email += 1
print(f"titles with email-like strings (20k sample): {n_email}")
# PeerSum review_writers values
print("\nPeerSum: scanning review_writers values...")
writers = Counter()
with open(f"{RAW}/peersum_huggingface.jsonl") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        writers.update(r["review_writers"])
        if i >= 2000:
            break
print("review_writers values (2k-record sample):", dict(writers))

print("\nDONE")
