# Data Card: ICLR 2017-2025 Processed Review Tables (Review Lottery)

## Dataset summary

Processed, analysis-ready tables of public ICLR peer-review data for ICLR
2017-2025, prepared for the paper *"The Review Lottery: Calibrating an
Observational Estimator of Peer-Review Noise (ICLR 2017-2025)"*
(OSF preregistration: https://osf.io/5vh4e/overview).

- **papers.parquet** (36,113 rows): one row per paper; year, area, decision
  (accept/reject/withdrawn), number of scored reviews `k`, mean/sd of scores.
- **reviews.parquet** (134,912 rows): one row per review; paper id, year,
  raw and standardized score, source dataset. `confidence` and `review_len`
  columns are present but empty for the primary source (verified: 0 of
  134,912 non-empty).
- **peersum_reviews.parquet** (53,732 rows): ICLR 2018-2022 subset from
  PeerSum with reviewer confidence and review text length (used for the
  confidence sensitivity analysis and LLM-era descriptives).

## Sources

- Primary: `berenslab/iclr-dataset` (GitHub mirror of public OpenReview data),
  ICLR 2017-2025.
- Cross-check: `oaimli/PeerSum` (Hugging Face), ICLR 2018-2022 slice.

## Processing rules

1. Papers matched to decisions from the source datasets; 7,474 papers with
   unknown decisions were verified to be withdrawals (not data corruption).
2. Withdrawn papers **with scores** enter the variance decomposition (they
   inform paper-quality estimation) but are **excluded** from decision-model
   fitting, which requires an observed outcome.
3. Papers without any scored review (496) are excluded from model fitting.
4. Author identifiers (`authors`, `author_ids`) were **deleted at load time**.
5. A column-name scan confirmed **no reviewer identity fields**; no
   cross-paper reviewer linkage is performed, so reviewer stringency is folded
   into the noise term.
6. Rating scales were verified per year from observed score values:
   1-10 in 2017-2019 and 2021; {1,3,6,8} in 2020; {1,3,5,6,8,10} in 2022-2025.

## Intended use

Estimating reviewer-noise / two-committee disagreement statistics in the
aggregate. Not suitable for identifying reviewers or authors, or for
evaluating individual papers.

## License

CC0 (derived from public OpenReview data).
