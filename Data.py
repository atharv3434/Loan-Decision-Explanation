"""Synthetic loan applicant data with a KNOWN ground-truth decision rule —
the only way to verify an explanation method is actually attributing
importance correctly, rather than just producing plausible-looking output.

Features, by design:
  income              — genuinely causal (higher income -> lower risk)
  debt_to_income      — genuinely causal (higher ratio -> higher risk)
  credit_score        — genuinely causal (higher score -> lower risk)
  employment_years    — genuinely causal (more tenure -> lower risk)
  zip_code_risk_score — a PROXY: correlated with income (as real
                         neighborhood-level risk scores often are with
                         income), but does NOT enter the true decision rule
                         at all. A trustworthy explanation method should
                         assign this little to no credit — if it doesn't,
                         that's a real, actionable finding: the model may be
                         leaning on a proxy variable, a genuine fair-lending
                         concern in real underwriting models.
  lucky_number         — pure noise, independent of everything; the sanity
                         check every method should rank at the bottom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = ["income", "debt_to_income", "credit_score", "employment_years", "zip_code_risk_score", "lucky_number"]

GROUND_TRUTH_ROLE = {
    "income": "causal",
    "debt_to_income": "causal",
    "credit_score": "causal",
    "employment_years": "causal",
    "zip_code_risk_score": "proxy (correlated with income, not causal)",
    "lucky_number": "noise",
}

# Direction each feature should move to REDUCE risk (used by the
# counterfactual search) — None for features that aren't actionable /
# shouldn't be used to justify a decision at all.
ACTIONABLE_DIRECTION = {
    "income": +1,
    "debt_to_income": -1,
    "credit_score": +1,
    "employment_years": +1,
    "zip_code_risk_score": None,
    "lucky_number": None,
}


def generate_applicants(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    income = rng.uniform(20_000, 150_000, n_samples)
    debt_to_income = rng.uniform(0.0, 0.6, n_samples)
    credit_score = rng.uniform(500, 850, n_samples)
    employment_years = rng.uniform(0, 30, n_samples)
    lucky_number = rng.uniform(0, 100, n_samples)

    # Proxy: correlated with income, but absent from the true risk equation below.
    zip_code_risk_score = -income / 40_000 + rng.normal(0, 0.3, n_samples)

    risk = (
        -income / 30_000
        + debt_to_income * 6
        - (credit_score - 650) / 80
        - employment_years / 10
        + rng.normal(0, 0.4, n_samples)
    )
    # Calibrated against the actual risk distribution (median ~ -2.86) so
    # roughly half of applicants are approved, giving both classes plenty of
    # examples for training and for later picking a realistic denied
    # applicant to explain.
    approved = (risk < -2.85).astype(int)

    return pd.DataFrame({
        "income": income,
        "debt_to_income": debt_to_income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "zip_code_risk_score": zip_code_risk_score,
        "lucky_number": lucky_number,
        "approved": approved,
    })