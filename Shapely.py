"""Exact Shapley value computation from first principles (game theory), not
an approximation like KernelSHAP's sampling or an assumption specific to
tree models like TreeSHAP. Feasible here because the feature set is small
(brute-force over 2^n coalitions, n=6 -> 64 subsets — instant).

The core idea: treat each feature as a "player" in a cooperative game where
the "payout" of a coalition S (a subset of features) is what the model would
predict if it only had access to features in S, with the rest replaced by
values drawn from a background dataset (the standard interventional/
marginal definition of a coalition's value). The Shapley value for feature i
is i's average marginal contribution across every possible ORDER those
features could be revealed in — which is exactly what makes it the unique
attribution method satisfying a specific, well-defined set of fairness
axioms (efficiency, symmetry, dummy, additivity) from cooperative game theory.

This implementation's correctness is checked directly against the defining
"efficiency" property: the Shapley values for one instance must sum exactly
to (that instance's prediction) - (the average background prediction). See
tests/test_shapley.py — not just a plausibility check, an exact mathematical
identity this implementation must satisfy.
"""
from __future__ import annotations

import math
from itertools import combinations

import pandas as pd


def _coalition_value(model, instance: pd.Series, background: pd.DataFrame, feature_indices: tuple[int, ...], feature_names: list[str]) -> float:
    """v(S): average model output when features in `feature_indices` are
    fixed to `instance`'s values and every other feature is drawn from each
    row of `background` in turn (marginalizing out the "unknown" features)."""
    n_background = len(background)
    rows = pd.DataFrame([instance.to_dict()] * n_background, index=background.index)

    unknown_columns = [feature_names[i] for i in range(len(feature_names)) if i not in feature_indices]
    if unknown_columns:
        rows[unknown_columns] = background[unknown_columns].to_numpy()

    predictions = model.predict_proba(rows[feature_names])[:, 1]
    return float(predictions.mean())


def exact_shapley_values(
    model, instance: pd.Series, background: pd.DataFrame, feature_names: list[str]
) -> dict[str, float]:
    """Returns {feature_name: shapley_value} for one instance."""
    n = len(feature_names)

    # Precompute v(S) for every one of the 2^n subsets once, since each
    # feature's Shapley value needs v(S) and v(S union {i}) for many subsets.
    coalition_values: dict[frozenset, float] = {}
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            coalition_values[frozenset(subset)] = _coalition_value(model, instance, background, subset, feature_names)

    shapley_values = {}
    all_indices = set(range(n))

    for i in range(n):
        other_indices = all_indices - {i}
        total = 0.0

        for size in range(len(other_indices) + 1):
            weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)
            for subset in combinations(sorted(other_indices), size):
                s = frozenset(subset)
                s_with_i = frozenset(subset) | {i}
                marginal_contribution = coalition_values[s_with_i] - coalition_values[s]
                total += weight * marginal_contribution

        shapley_values[feature_names[i]] = total

    return shapley_values


def baseline_prediction(model, background: pd.DataFrame, feature_names: list[str]) -> float:
    """v(empty set): the model's average prediction with every feature
    replaced by background values — the reference point Shapley values are
    measured relative to."""
    return float(model.predict_proba(background[feature_names])[:, 1].mean())
