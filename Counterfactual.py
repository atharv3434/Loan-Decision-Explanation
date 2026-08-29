"""Counterfactual search: for a denied applicant, find the smallest realistic
change to their ACTIONABLE features (income, debt-to-income, credit score,
employment years — never the proxy or noise features, which shouldn't be
"advice" at all) that would flip the model's decision to approved.

Uses a simple greedy coordinate search: repeatedly nudge whichever
actionable feature currently gives the biggest reduction in predicted risk
per unit of (normalized) change, until the decision flips or a step budget
is exhausted. Not globally optimal, but transparent and fast — appropriate
for a prototype whose job is to demonstrate the idea working end-to-end.
"""
from __future__ import annotations

import pandas as pd

from decision_explainer.data import ACTIONABLE_DIRECTION


def find_counterfactual(
    model,
    instance: pd.Series,
    feature_names: list[str],
    feature_ranges: dict[str, tuple[float, float]],
    max_steps: int = 200,
    step_fraction: float = 0.01,
) -> dict | None:
    """Returns a dict describing the counterfactual (changed features and
    their new values, plus the resulting approval probability), or None if
    no counterfactual was found within the step budget.
    """
    current = instance.copy()
    actionable_features = [f for f in feature_names if ACTIONABLE_DIRECTION.get(f) is not None]

    if model.predict_proba(pd.DataFrame([current])[feature_names])[0, 1] >= 0.5:
        return {"already_approved": True}

    changes: dict[str, float] = {}

    for _ in range(max_steps):
        best_feature = None
        best_gain = 0.0
        best_new_value = None

        current_prob = model.predict_proba(pd.DataFrame([current])[feature_names])[0, 1]

        for feature in actionable_features:
            direction = ACTIONABLE_DIRECTION[feature]
            low, high = feature_ranges[feature]
            step_size = (high - low) * step_fraction * direction

            candidate = current.copy()
            candidate[feature] = float(candidate[feature]) + step_size
            candidate[feature] = min(max(candidate[feature], low), high)

            candidate_prob = model.predict_proba(pd.DataFrame([candidate])[feature_names])[0, 1]
            gain = candidate_prob - current_prob

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_new_value = candidate[feature]

        if best_feature is None:
            break  # no actionable feature can improve things further

        current[best_feature] = best_new_value
        changes[best_feature] = best_new_value

        new_prob = model.predict_proba(pd.DataFrame([current])[feature_names])[0, 1]
        if new_prob >= 0.5:
            return {
                "already_approved": False,
                "changed_features": {
                    f: {"from": float(instance[f]), "to": float(current[f])} for f in changes
                },
                "new_approval_probability": float(new_prob),
            }

    return None  # exhausted the step budget without flipping the decision
