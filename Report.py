"""Turns Shapley values + a counterfactual into a plain-language decision
report, in the spirit of a real adverse-action notice (US ECOA requires
lenders to give specific "principal reasons" for a denial, not just a
score) — the actual point of computing an explanation at all is presenting
it in a form a person can act on, not just a table of numbers.
"""
from __future__ import annotations

import pandas as pd

FEATURE_DESCRIPTIONS = {
    "income": "annual income of {value:,.0f} dollars",
    "debt_to_income": "debt-to-income ratio of {value:.1%}",
    "credit_score": "credit score of {value:.0f}",
    "employment_years": "{value:.1f} years of employment history",
    "zip_code_risk_score": "location-based risk score",
    "lucky_number": "lucky number",
}

# Features that should never be presented as a legitimate "reason" in a
# customer-facing report, even if the model's Shapley values assign them
# nontrivial weight — that assignment itself is the finding to flag, not
# something to launder into a plausible-sounding sentence.
NON_JUSTIFIABLE_FEATURES = {"zip_code_risk_score", "lucky_number"}


def _describe_feature(feature: str, value: float) -> str:
    template = FEATURE_DESCRIPTIONS.get(feature, feature)
    return template.format(value=value)


def generate_report(
    instance: pd.Series,
    shapley_values: dict[str, float],
    approval_probability: float,
    counterfactual: dict | None,
    top_n_reasons: int = 3,
) -> str:
    decision = "APPROVED" if approval_probability >= 0.5 else "DENIED"

    sorted_features = sorted(shapley_values.items(), key=lambda kv: abs(kv[1]), reverse=True)

    lines = [
        f"DECISION: {decision}",
        f"Model confidence: {approval_probability:.0%} probability of approval",
        "",
        "PRINCIPAL REASONS FOR THIS DECISION:",
    ]

    reasons_shown = 0
    flagged_proxy_reliance = []
    for feature, shap_value in sorted_features:
        if reasons_shown >= top_n_reasons:
            break

        direction = "increased" if shap_value > 0 else "decreased"
        description = _describe_feature(feature, instance[feature])

        if feature in NON_JUSTIFIABLE_FEATURES and abs(shap_value) > 0.02:
            flagged_proxy_reliance.append((feature, shap_value))
            continue  # don't present a proxy variable as a customer-facing "reason"

        lines.append(f"  {reasons_shown + 1}. Your {description} {direction} your approval likelihood.")
        reasons_shown += 1

    if flagged_proxy_reliance:
        lines.append("")
        lines.append("MODEL GOVERNANCE FLAG (not shown to the applicant):")
        for feature, shap_value in flagged_proxy_reliance:
            lines.append(
                f"  The model's decision was meaningfully influenced by '{feature}' "
                f"(Shapley contribution: {shap_value:+.3f}), which is not a legitimate "
                f"basis for a credit decision and may function as a proxy for a "
                f"protected characteristic. Recommend review."
            )

    if decision == "DENIED":
        lines.append("")
        if counterfactual is None:
            lines.append("No realistic combination of changes to actionable factors was found within the search budget.")
        elif counterfactual.get("already_approved"):
            pass
        else:
            lines.append("WHAT COULD CHANGE THIS DECISION:")
            for feature, change in counterfactual["changed_features"].items():
                lines.append(
                    f"  - {_describe_feature(feature, change['to'])} "
                    f"(currently {_describe_feature(feature, change['from'])})"
                )
            lines.append(
                f"  With these changes, the estimated approval probability would rise to "
                f"{counterfactual['new_approval_probability']:.0%}."
            )

    return "\n".join(lines)
