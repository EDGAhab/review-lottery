# The Review Lottery: Calibrating an Observational Estimator of Peer-Review Noise (ICLR 2017-2025)

**OSF preregistration:** https://osf.io/5vh4e/overview (project: https://osf.io/zgqnw/overview)

## What this is

How much of a conference accept/reject decision would change if the same paper were
reviewed by a different set of reviewers? This project builds an estimator of that
counterfactual from public review scores alone: a Bayesian ordered-probit variance
decomposition separates paper quality from reviewer noise year by year, a logistic
decision model approximates the area-chair stage, and a two-committee simulation
(posterior draws B=1,000; k=2,3,4) estimates disagreement rates. The estimator is
calibrated twice: against the NeurIPS 2014/2021 randomized consistency experiments
(external anchor) and against a model-free 2+2 reviewer split-half benchmark
(internal anchor). Applied to ICLR 2017-2025 (36,113 papers, 134,912 reviews).

Headline results: two-committee disagreement 23-30% (k=2); 30-50% of accepted papers
would flip; external bias +0.3pp at the matched caliber; no robust longitudinal trend;
2020's four-point scale measurably raised disagreement (+7pp in a counterfactual).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pymc, arviz, numpy, pandas, scikit-learn, matplotlib, pyarrow
# 1. Load & clean (expects data/raw/iclr26v1.parquet + peersum_huggingface.jsonl;
#    or download the processed tables from the HF dataset below)
python3 scripts/01_load_clean.py
python3 scripts/02_build_dataset.py
# 2. Descriptives
python3 scripts/03_describe.py
# 3. RQ-A: per-year ordered-probit fits (CPU; 2025 takes ~75 min)
python3 scripts/04_model_fit.py            # writes posteriors/draws_YYYY.npz
python3 scripts/04b_conf_sensitivity.py    # PeerSum confidence sensitivity
python3 scripts/05_decision_model.py
python3 scripts/06_simulate.py             # B=1000 two-committee simulation
python3 scripts/07_calibrate.py
python3 scripts/08_build_rho_table.py
# 4. RQ-B / RQ-C
python3 scripts/09_longitudinal.py
python3 scripts/10_anomaly.py && python3 scripts/10b_counterfactual.py && python3 scripts/10c_counterfactual_threshold.py
python3 scripts/11_llm_descriptives.py
# 5. Paper (requires a TeX Live install + the TMLR style files in manuscript/latex/)
cd manuscript && python3 convert_md2tex.py --preprint && cd latex && pdflatex review_lottery.tex
```

Reports: `phase0_report.md`, `phase1_report.md`, `phase2_report.md`, `phase3_report.md`.
Manuscript: `manuscript/manuscript_v1.md` (source of record),
`manuscript/tmlr_review_lottery.pdf` (preprint build),
`manuscript/latex/review_lottery_anon.pdf` (anonymous TMLR-submission build).

## Data

- Primary: `berenslab/iclr-dataset` mirror of public OpenReview ICLR reviews.
- Cross-check: `oaimli/PeerSum` (ICLR 2018-2022 subset; has confidence and text length).
- Processed tables: `data/processed/` (also on Hugging Face, see below).
- Posterior draws (`posteriors/draws_*.npz`, ~1.1 GB total): hosted on Hugging Face,
  not in this repo (exceeds GitHub file limits).

Privacy: author identifiers (`authors`, `author_ids`) were deleted at load; a
column-name scan confirmed no reviewer identity fields; no cross-paper reviewer
linkage is performed. Only aggregate statistics are published.

## Citation

Feilian Huang. The Review Lottery: Calibrating an Observational Estimator of
Peer-Review Noise (ICLR 2017-2025). 2026. Preregistered at
https://osf.io/5vh4e/overview.

## License

Code: MIT (see LICENSE). Data tables: CC0 (derived from public OpenReview data).
Manuscript: CC-BY 4.0.
