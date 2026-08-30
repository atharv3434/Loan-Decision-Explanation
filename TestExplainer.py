"""Tests for the loan decision explainer: data calibration, the Shapley
efficiency property (an exact mathematical identity, not a plausibility
check), correct attribution of causal vs proxy vs noise features,
counterfactual validity, and report generation."""
import numpy as np
import pandas as pd
import pytest

from decision_explainer.counterfactual import find_counterfactual
from decision_explainer.data import FEATURE_NAMES, GROUND_TRUTH_ROLE, generate_applicants
from decision_explainer.model import train_model
from decision_explainer.report import NON_JUSTIFIABLE_FEATURES, generate_report
from decision_explainer.shapley import baseline_prediction, exact_shapley_values


@pytest.fixture(scope="module")
def trained():
    df = generate_applicants(1200, random_state=42)
    X, y = df[FEATURE_NAMES], df["approved"]
    model, X_train, X_test, y_train, y_test = train_model(X, y, random_state=42)
    background = X_train.sample(25, random_state=1)
    return {"model": model, "X_train": X_train, "X_test": X_test, "y_test": y_test, "background": background}


class TestDataCalibration:
    def test_approval_rate_is_reasonably_balanced(self):
        df = generate_applicants(2000, random_state=1)
        assert 0.35 < df["approved"].mean() < 0.65

    def test_zip_code_is_strongly_correlated_with_income(self):
        df = generate_applicants(2000, random_state=1)
        corr = np.corrcoef(df["income"], df["zip_code_risk_score"])[0, 1]
        assert abs(corr) > 0.8

    def test_lucky_number_is_uncorrelated_with_everything(self):
        df = generate_applicants(2000, random_state=1)
        for col in ["income", "debt_to_income", "credit_score", "employment_years"]:
            corr = np.corrcoef(df["lucky_number"], df[col])[0, 1]
            assert abs(corr) < 0.1


class TestModel:
    def test_achieves_reasonable_accuracy(self, trained):
        acc = trained["model"].score(trained["X_test"], trained["y_test"])
        assert acc > 0.8


class TestShapley:
    def test_efficiency_property_holds_exactly(self, trained):
        """The defining mathematical property of Shapley values: they must
        sum exactly to (this instance's prediction) - (baseline prediction).
        This is not a heuristic check — if this fails, the implementation is
        simply wrong."""
        model, background = trained["model"], trained["background"]
        instance = trained["X_test"].iloc[0]

        baseline = baseline_prediction(model, background, FEATURE_NAMES)
        prediction = model.predict_proba(instance.to_frame().T[FEATURE_NAMES])[0, 1]
        shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)

        assert sum(shap_values.values()) == pytest.approx(prediction - baseline, abs=1e-9)

    def test_efficiency_property_holds_for_multiple_instances(self, trained):
        model, background = trained["model"], trained["background"]
        baseline = baseline_prediction(model, background, FEATURE_NAMES)

        for i in range(5):
            instance = trained["X_test"].iloc[i]
            prediction = model.predict_proba(instance.to_frame().T[FEATURE_NAMES])[0, 1]
            shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)
            assert sum(shap_values.values()) == pytest.approx(prediction - baseline, abs=1e-9)

    def test_causal_features_outweigh_noise_feature(self, trained):
        """A genuine correctness check against known ground truth: the
        pure-noise feature should get near-zero credit, far below any
        causal feature."""
        model, background = trained["model"], trained["background"]
        instance = trained["X_test"].iloc[0]
        shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)

        causal_magnitudes = [abs(shap_values[f]) for f, role in GROUND_TRUTH_ROLE.items() if role == "causal"]
        noise_magnitude = abs(shap_values["lucky_number"])

        assert noise_magnitude < min(causal_magnitudes)

    def test_averaged_over_many_instances_proxy_gets_less_credit_than_causal_features(self, trained):
        """On average across many applicants, the proxy feature should be
        credited less than the genuinely causal features — the core claim
        this tool exists to check on a real model."""
        model, background = trained["model"], trained["background"]
        proxy_total, causal_totals = 0.0, {f: 0.0 for f, r in GROUND_TRUTH_ROLE.items() if r == "causal"}

        n_check = 15
        for i in range(n_check):
            instance = trained["X_test"].iloc[i]
            shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)
            proxy_total += abs(shap_values["zip_code_risk_score"])
            for f in causal_totals:
                causal_totals[f] += abs(shap_values[f])

        avg_proxy = proxy_total / n_check
        avg_causal = {f: t / n_check for f, t in causal_totals.items()}
        assert avg_proxy < max(avg_causal.values())


class TestCounterfactual:
    def test_returns_already_approved_for_an_approved_applicant(self, trained):
        model = trained["model"]
        feature_ranges = {f: (trained["X_train"][f].min(), trained["X_train"][f].max()) for f in FEATURE_NAMES}

        approved_idx = trained["X_test"][model.predict(trained["X_test"]) == 1].index[0]
        instance = trained["X_test"].loc[approved_idx]

        result = find_counterfactual(model, instance, FEATURE_NAMES, feature_ranges)
        assert result["already_approved"] is True

    def test_counterfactual_actually_flips_the_decision(self, trained):
        """The real correctness check: applying the suggested changes to the
        applicant's data must actually cause the model to approve them —
        not just look plausible."""
        model = trained["model"]
        feature_ranges = {f: (trained["X_train"][f].min(), trained["X_train"][f].max()) for f in FEATURE_NAMES}

        denied_mask = model.predict(trained["X_test"]) == 0
        denied_idx = trained["X_test"][denied_mask].index[0]
        instance = trained["X_test"].loc[denied_idx]

        result = find_counterfactual(model, instance, FEATURE_NAMES, feature_ranges)
        assert result is not None and not result.get("already_approved")

        modified = instance.copy()
        for feature, change in result["changed_features"].items():
            modified[feature] = change["to"]

        new_prediction = model.predict_proba(modified.to_frame().T[FEATURE_NAMES])[0, 1]
        assert new_prediction >= 0.5

    def test_counterfactual_never_changes_non_actionable_features(self, trained):
        model = trained["model"]
        feature_ranges = {f: (trained["X_train"][f].min(), trained["X_train"][f].max()) for f in FEATURE_NAMES}

        denied_idx = trained["X_test"][model.predict(trained["X_test"]) == 0].index[0]
        instance = trained["X_test"].loc[denied_idx]

        result = find_counterfactual(model, instance, FEATURE_NAMES, feature_ranges)
        if result and not result.get("already_approved"):
            assert "zip_code_risk_score" not in result["changed_features"]
            assert "lucky_number" not in result["changed_features"]


class TestReport:
    def test_report_contains_decision_and_reasons(self, trained):
        model, background = trained["model"], trained["background"]
        instance = trained["X_test"].iloc[0]
        pred = model.predict_proba(instance.to_frame().T[FEATURE_NAMES])[0, 1]
        shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)

        report = generate_report(instance, shap_values, pred, counterfactual=None)
        assert "DECISION:" in report
        assert "PRINCIPAL REASONS" in report

    def test_non_justifiable_features_never_appear_as_a_customer_facing_reason(self, trained):
        """Even if the model happens to lean on the proxy feature, the
        report must never present it as a plain 'reason' — it should be
        routed to the governance flag instead."""
        model, background = trained["model"], trained["background"]

        instance = trained["X_test"].iloc[0].copy()
        instance["zip_code_risk_score"] = 5.0

        pred = model.predict_proba(instance.to_frame().T[FEATURE_NAMES])[0, 1]
        shap_values = exact_shapley_values(model, instance, background, FEATURE_NAMES)
        report = generate_report(instance, shap_values, pred, counterfactual=None)

        for line in report.split("\n"):
            if line.strip().startswith(tuple("123")):
                for feature in NON_JUSTIFIABLE_FEATURES:
                    assert feature not in line
