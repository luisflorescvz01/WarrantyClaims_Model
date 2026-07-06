# Vehicle Claim Validity Model

Predicts whether a vehicle claim is **valid** or **invalid/flagged**, as a
prototype for automotive warranty-claim cost reduction.

## Dataset
`data/vehicle_claims.csv` — 1,000 auto claims, 39 features, target
`fraud_reported` (Y = invalid/flagged, N = valid; ~25% invalid).
Note: this is public auto *insurance* claims data used as a stand-in for
proprietary OEM warranty data. The pipeline transfers directly to real
warranty features (repair codes, mileage, time-in-service, dealer ID...).

## Project structure
```
warranty-claims-model/
├── data/vehicle_claims.csv    # dataset
├── src/explore.py             # quick EDA
├── src/features.py            # feature engineering (runs inside the pipeline)
├── src/train.py               # CV model selection + cost-tuned threshold
├── src/predict.py             # score raw claims with the saved model
├── src/dashboard.py           # plain-English HTML performance dashboard
├── models/                    # saved {pipeline, threshold} (created by train.py)
├── reports/                   # metrics + cost report (created by train.py)
└── requirements.txt
```

## Setup (VS Code)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/explore.py                          # dataset overview
python src/train.py                            # train + evaluate + tune threshold
python src/predict.py claims.csv scored.csv    # score new raw claims
python src/dashboard.py                        # build + open reports/dashboard.html
```

## Pipeline
1. **Feature engineering** (`src/features.py`, embedded in the sklearn
   Pipeline so the saved model accepts raw claim rows): vehicle age, claim
   composition shares, claim-vs-premium and claim-vs-deductible ratios,
   customer tenure, night-incident flag. "?" values are kept as an explicit
   `UNKNOWN` category — a missing police report is itself a signal.
2. **Model selection** by 5-fold CV ROC-AUC on the training set only
   (LogisticRegression / RandomForest / HistGradientBoosting, each with a
   small hyperparameter grid). The held-out test set is never used for
   selection.
3. **Cost-sensitive threshold**: tuned on out-of-fold train predictions to
   minimize total dollar cost — a missed fraudulent claim costs its full
   payout, a flagged claim costs a $500 investigation — subject to a 30%
   review-capacity limit (without the limit the optimum degenerates to
   "review everything").

## Results (25% held-out test set)
| Metric | Baseline (RF, 0.50) | Refined (LogReg C=0.03, 0.55) |
|---|---|---|
| ROC-AUC | 0.763 | 0.776 |
| PR-AUC | — | 0.551 |
| Invalid recall | 0.63 | 0.63 |
| Claims flagged | — | 27% |

Dollar-cost on the test set (250 claims): $3.64M if every claim is paid,
$1.19M with the model at the tuned threshold — **~67% estimated savings**,
catching 63% of invalid claims while reviewing 27% of volume.

## Roadmap
1. SHAP explainability for adjuster-facing flag reasons
2. XGBoost/LightGBM comparison
3. Calibrated probabilities (CalibratedClassifierCV) so scores read as risk
4. Sensitivity analysis on investigation cost / review capacity
