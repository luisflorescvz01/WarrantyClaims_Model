"""
Score raw claims with the trained model.

Usage:
    python src/predict.py <claims.csv> [output.csv]

Input:  CSV with the same columns as data/vehicle_claims.csv
        (the fraud_reported column, if present, is ignored).
Output: input rows plus `fraud_score` and `decision`
        (FLAG_FOR_REVIEW / AUTO_APPROVE), written to output.csv
        or printed as a summary if no output path is given.

Note: the saved pipeline references src/features.py, so the model must be
loaded with this directory on sys.path - which this script handles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    bundle = joblib.load(MODEL_PATH)
    pipe, threshold = bundle["pipeline"], bundle["threshold"]

    claims = pd.read_csv(sys.argv[1])
    claims = claims.drop(columns=["fraud_reported"], errors="ignore")

    scores = pipe.predict_proba(claims)[:, 1]
    out = claims.copy()
    out["fraud_score"] = scores.round(4)
    out["decision"] = pd.Series(scores >= threshold).map(
        {True: "FLAG_FOR_REVIEW", False: "AUTO_APPROVE"}
    )

    flagged = (scores >= threshold).sum()
    print(
        f"Scored {len(out)} claims at threshold {threshold:.2f}: "
        f"{flagged} flagged for review, {len(out) - flagged} auto-approved."
    )
    if len(sys.argv) > 2:
        out.to_csv(sys.argv[2], index=False)
        print(f"Wrote {sys.argv[2]}")
    else:
        print(out[["fraud_score", "decision"]].head(10).to_string())


if __name__ == "__main__":
    main()
