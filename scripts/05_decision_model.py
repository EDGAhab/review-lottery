"""05_decision_model.py — Logistic decision model approximating the AC stage.

P(accept | mean_score, sd_score, year). Withdrawn (decision unknown) excluded
(no outcome) per Phase 1 decision. Confidence omitted (berenslab lacks it);
declared in Phase 2 report.

Sensitivity: pure threshold rule per year (threshold on mean_score reproducing
that year's acceptance rate).

Outputs: fitted model (joblib), coefficients JSON, calibration curve plot,
threshold table.
"""
import json, os
import numpy as np
import pandas as pd

WORK = os.path.expanduser("~/workspace/review-lottery")
PROC = os.path.join(WORK, "data/processed")
FIG = os.path.join(WORK, "figures")
NOTES = os.path.join(WORK, "phase2_notes")
os.makedirs(FIG, exist_ok=True); os.makedirs(NOTES, exist_ok=True)

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    import joblib

    p = pd.read_parquet(os.path.join(PROC, "papers.parquet"))
    d = p[(p["decision_binary"] != "unknown") & (p["has_scores"])].copy()
    d["y"] = (d["decision_binary"] == "accept").astype(int)
    d["sd_score"] = d["sd_score"].fillna(0.0)
    print(f"decision-model rows: {len(d)} (accept={int(d['y'].sum())})", flush=True)

    year_dummies = pd.get_dummies(d["year"], prefix="y", dtype=float)
    X = pd.concat([d[["mean_score", "sd_score"]], year_dummies], axis=1)
    feature_names = list(X.columns)
    y = d["y"].to_numpy()

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    proba_cv = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba_cv)
    print(f"5-fold CV AUC: {auc:.4f}", flush=True)

    joblib.dump({"model": clf, "features": feature_names,
                 "year_min": int(d["year"].min()), "year_max": int(d["year"].max())},
                os.path.join(WORK, "decision_model.joblib"))
    coefs = dict(zip(feature_names, [float(c) for c in clf.coef_[0]]))
    coefs["_intercept"] = float(clf.intercept_[0])
    coefs["_cv_auc"] = float(auc)
    with open(os.path.join(NOTES, "decision_model_coefs.json"), "w") as f:
        json.dump(coefs, f, indent=2)

    # calibration curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y, clf.predict_proba(X)[:, 1], n_bins=10)
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, "o-", label="logistic")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    plt.xlabel("predicted P(accept)"); plt.ylabel("observed accept rate")
    plt.title(f"Decision model calibration (CV AUC={auc:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_decision_calibration.png"), dpi=120)
    plt.close()

    # pure threshold rule per year: threshold on mean_score reproducing accept rate
    thresh = {}
    for yy, g in d.groupby("year"):
        acc_rate = g["y"].mean()
        t = float(np.quantile(g["mean_score"], 1 - acc_rate))
        # check reproduced rate
        rep = float((g["mean_score"] >= t).mean())
        thresh[int(yy)] = {"threshold": t, "accept_rate": float(acc_rate),
                           "reproduced_rate": rep, "n": int(len(g))}
    with open(os.path.join(NOTES, "threshold_rule.json"), "w") as f:
        json.dump(thresh, f, indent=2)
    print("thresholds:", {k: round(v["threshold"], 3) for k, v in thresh.items()}, flush=True)
    print("saved decision_model.joblib, coefs, calibration plot, thresholds", flush=True)

if __name__ == "__main__":
    main()
