"""
Feature engineering for the vehicle claim validity model.

`engineer()` takes a raw claims DataFrame (all original columns except the
target) and returns a model-ready frame:
  - derived features (vehicle age, claim ratios, tenure, night incident...)
  - "?" placeholders kept as an explicit "UNKNOWN" category, since a missing
    police report / damage confirmation is itself a fraud signal
  - identifier / leaky columns dropped

It is used as the first step of the sklearn Pipeline (via FunctionTransformer)
so the saved joblib model accepts raw claim rows directly.
"""

import numpy as np
import pandas as pd

# Identifiers / free text with no generalizable signal, plus insured_hobbies
# (known leaky artifact of this public dataset). incident_date and auto_year
# are consumed by derived features below, then dropped.
DROP_COLS = [
    "policy_number",
    "policy_bind_date",
    "insured_zip",
    "incident_date",
    "incident_location",
    "insured_hobbies",
    "auto_year",
]


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # "Not reported" is signal, not noise - keep it as its own category.
    obj_cols = X.select_dtypes(include=["object"]).columns
    X[obj_cols] = X[obj_cols].replace("?", "UNKNOWN").fillna("UNKNOWN")

    incident_year = pd.to_datetime(
        X["incident_date"], format="%d/%m/%Y", errors="coerce"
    ).dt.year

    # Vehicle / customer profile
    X["vehicle_age"] = (incident_year - X["auto_year"]).clip(lower=0)
    X["tenure_years"] = X["months_as_customer"] / 12.0
    X["age_at_signup"] = X["age"] - X["tenure_years"]

    # Claim composition - unusual splits between injury/property/vehicle
    # components are a classic soft-fraud pattern.
    total = X["total_claim_amount"].replace(0, np.nan)
    X["injury_share"] = (X["injury_claim"] / total).fillna(0)
    X["property_share"] = (X["property_claim"] / total).fillna(0)
    X["vehicle_share"] = (X["vehicle_claim"] / total).fillna(0)

    # Claim size relative to policy economics
    X["claim_per_premium"] = X["total_claim_amount"] / X[
        "policy_annual_premium"
    ].replace(0, np.nan)
    X["claim_per_premium"] = X["claim_per_premium"].fillna(0)
    X["claim_vs_deductible"] = X["total_claim_amount"] / X[
        "policy_deductable"
    ].replace(0, np.nan)
    X["claim_vs_deductible"] = X["claim_vs_deductible"].fillna(0)

    X["net_capital"] = X["capital-gains"] + X["capital-loss"]
    X["night_incident"] = (
        (X["incident_hour_of_the_day"] >= 22) | (X["incident_hour_of_the_day"] <= 5)
    ).astype(int)

    return X.drop(columns=DROP_COLS)
