"""
Vehicle Claim Validity Model - Training
========================================
Refined pipeline:
  1. Feature engineering (src/features.py) inside the sklearn Pipeline,
     so the saved model accepts raw claim rows.
  2. Model + hyperparameter selection via 5-fold cross-validation on the
     training set only (the held-out test set is never used for selection).
  3. Cost-sensitive decision threshold tuned on out-of-fold predictions
     using a dollar-cost matrix:
       - missing a fraudulent claim costs its full claim amount
       - flagging a claim for review costs a fixed investigation fee

Usage:
    python src/train.py

Outputs:
    models/best_model.joblib   - {"pipeline": ..., "threshold": ...}
    reports/metrics.txt        - evaluation + cost analysis report
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import engineer  # noqa: E402

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vehicle_claims.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
TARGET = "fraud_reported"  # 'Y' = invalid/flagged claim, 'N' = valid

# ------------------------------------------------------- dollar-cost matrix
# Cost of sending a claim to manual review (adjuster time, delay).
INVESTIGATION_COST = 500.0
# Missing a fraudulent claim costs its full payout (claim amount).
# Flagged fraudulent claims are assumed caught in review, so their
# payout is avoided at the price of one investigation.
#
# Because the average fraudulent payout dwarfs the investigation fee, an
# unconstrained optimum degenerates to "review everything". Adjuster
# capacity is limited in practice, so the threshold is tuned subject to
# flagging at most this share of incoming claims:
REVIEW_CAPACITY = 0.30


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = (df[TARGET] == "Y").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_pipeline(model) -> Pipeline:
    """Raw claims -> engineered features -> encode/scale -> classifier."""
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=object),
            ),
            (
                "num",
                StandardScaler(),
                make_column_selector(dtype_include=np.number),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer)),
            ("pre", pre),
            ("clf", model),
        ]
    )


def get_candidates() -> dict:
    """Candidate models with small hyperparameter grids."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [None, 8, 12],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_leaf_nodes": [15, 31],
                "clf__max_iter": [150, 300],
            },
        ),
    }


def review_cost(y_true, proba, claim_amounts, threshold) -> float:
    """Total dollar cost of a flag-for-review policy at `threshold`.

    Flagged claims cost one investigation each; fraudulent claims that
    slip through cost their full claim amount.
    """
    flagged = proba >= threshold
    missed_fraud = (~flagged) & (y_true == 1)
    return INVESTIGATION_COST * flagged.sum() + claim_amounts[missed_fraud].sum()


def tune_threshold(y_true, proba, claim_amounts) -> tuple[float, pd.DataFrame]:
    """Pick the threshold that minimizes total expected cost, subject to
    flagging at most REVIEW_CAPACITY of claims."""
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        flag_rate = (proba >= t).mean()
        rows.append(
            (round(t, 2), flag_rate, review_cost(y_true, proba, claim_amounts, t))
        )
    curve = pd.DataFrame(rows, columns=["threshold", "flag_rate", "cost"])
    feasible = curve[curve["flag_rate"] <= REVIEW_CAPACITY]
    if feasible.empty:  # capacity tighter than the coarsest threshold
        feasible = curve.tail(1)
    return float(feasible.loc[feasible["cost"].idxmin(), "threshold"]), curve


def main() -> None:
    X, y = load_data(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- model selection: CV on the training set only
    report_lines = []
    best_cv_auc, best_name, best_search = -1.0, None, None
    for name, (model, grid) in get_candidates().items():
        search = GridSearchCV(
            build_pipeline(model),
            grid,
            scoring="roc_auc",
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        line = (
            f"{name:22s} CV ROC-AUC: {search.best_score_:.3f}  "
            f"params: { {k.removeprefix('clf__'): v for k, v in search.best_params_.items()} }"
        )
        print(line)
        report_lines.append(line)
        if search.best_score_ > best_cv_auc:
            best_cv_auc, best_name, best_search = search.best_score_, name, search

    best_pipe = best_search.best_estimator_
    report_lines.append(f"\nSelected model: {best_name} (CV ROC-AUC {best_cv_auc:.3f})")

    # ---- cost-sensitive threshold, tuned on out-of-fold train predictions
    oof_proba = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=CV_FOLDS, method="predict_proba", n_jobs=-1
    )[:, 1]
    amounts_tr = X_tr["total_claim_amount"].to_numpy()
    threshold, _ = tune_threshold(y_tr.to_numpy(), oof_proba, amounts_tr)
    report_lines.append(
        f"Cost-optimal threshold: {threshold:.2f} "
        f"(investigation ${INVESTIGATION_COST:,.0f}/claim, "
        f"review capacity {REVIEW_CAPACITY:.0%} of claims)"
    )

    # ---- final evaluation on the held-out test set
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= threshold).astype(int)
    amounts_te = X_te["total_claim_amount"].to_numpy()
    y_te_arr = y_te.to_numpy()

    cost_no_model = amounts_te[y_te_arr == 1].sum()  # pay every fraud
    cost_default = review_cost(y_te_arr, proba_te, amounts_te, 0.5)
    cost_tuned = review_cost(y_te_arr, proba_te, amounts_te, threshold)

    block = (
        f"\n{'=' * 60}\n"
        f"Held-out test set ({len(y_te)} claims)\n"
        f"ROC-AUC: {roc_auc_score(y_te, proba_te):.3f}   "
        f"PR-AUC: {average_precision_score(y_te, proba_te):.3f}\n\n"
        f"Confusion matrix at threshold {threshold:.2f} "
        f"(rows=true, cols=pred [Valid, Invalid]):\n"
        f"{confusion_matrix(y_te, pred_te)}\n"
        f"{classification_report(y_te, pred_te, target_names=['Valid', 'Invalid'], digits=3)}\n"
        f"{'-' * 60}\n"
        f"Dollar-cost analysis (test set, investigation ${INVESTIGATION_COST:,.0f}, "
        f"capacity {REVIEW_CAPACITY:.0%}):\n"
        f"  Pay all claims (no model):      ${cost_no_model:>12,.0f}\n"
        f"  Model @ default 0.50 threshold: ${cost_default:>12,.0f}  "
        f"(flags {(proba_te >= 0.5).mean():.0%} of claims)\n"
        f"  Model @ tuned {threshold:.2f} threshold:   ${cost_tuned:>12,.0f}  "
        f"(flags {pred_te.mean():.0%} of claims)\n"
        f"  Estimated savings vs no model:  ${cost_no_model - cost_tuned:>12,.0f} "
        f"({(cost_no_model - cost_tuned) / cost_no_model:.1%})"
    )
    print(block)
    report_lines.append(block)

    # ---- persist model + threshold + report
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "threshold": threshold}, MODEL_PATH)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved pipeline + threshold -> {MODEL_PATH}")
    print(f"Saved metrics report       -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
