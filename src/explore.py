"""
Quick exploratory data analysis for the vehicle claims dataset.

Usage:
    python src/explore.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vehicle_claims.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH).replace("?", np.nan)

    print("Shape:", df.shape)
    print("\nTarget distribution:")
    print(df["fraud_reported"].value_counts(normalize=True).round(3))

    print("\nMissing values per column (only columns with any):")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\nInvalid-claim rate by incident severity:")
    print(
        df.groupby("incident_severity")["fraud_reported"]
        .apply(lambda s: (s == "Y").mean())
        .round(3)
        .sort_values(ascending=False)
    )

    print("\nInvalid-claim rate by incident type:")
    print(
        df.groupby("incident_type")["fraud_reported"]
        .apply(lambda s: (s == "Y").mean())
        .round(3)
        .sort_values(ascending=False)
    )

    print("\nMean claim amounts by validity:")
    print(
        df.groupby("fraud_reported")[
            ["total_claim_amount", "vehicle_claim", "injury_claim", "property_claim"]
        ].mean().round(0)
    )


if __name__ == "__main__":
    main()
