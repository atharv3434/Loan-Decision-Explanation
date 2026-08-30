"""Command-line interface for the loan decision explainer."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from decision_explainer.config import Config
from decision_explainer.counterfactual import find_counterfactual
from decision_explainer.engine import load_bundle, train_and_save
from decision_explainer.report import generate_report
from decision_explainer.shapley import exact_shapley_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


@click.group()
def cli() -> None:
    """Explain individual loan approval/denial decisions: Shapley values,
    counterfactuals, and a plain-language report."""


@cli.command()
@click.option("--n-samples", default=3000, show_default=True)
def train(n_samples: int) -> None:
    """Train the loan approval model and save it, a background sample, and
    a pool of sample applicants to explain."""
    config = Config(n_samples=n_samples)
    metrics = train_and_save(config)
    click.echo(json.dumps(metrics, indent=2))


@cli.command()
@click.option("--index", "sample_index", default=0, show_default=True, help="Which sample applicant to explain (0-19).")
@click.option("--applicant-json", default=None, help="Path to a JSON file with a single applicant's feature values, instead of a sample index.")
def explain(sample_index: int, applicant_json: str | None) -> None:
    """Explain one applicant's decision: principal reasons + counterfactual."""
    config = Config()
    bundle = load_bundle(config)
    model = bundle["model"]
    background = bundle["background"]
    feature_ranges = bundle["feature_ranges"]
    feature_names = bundle["feature_names"]

    if applicant_json:
        with open(applicant_json) as f:
            record = json.load(f)
        instance = pd.Series(record)[feature_names]
    else:
        sample_path = Path(config.model_dir) / "sample_applicants.json"
        with open(sample_path) as f:
            samples = json.load(f)
        if not 0 <= sample_index < len(samples):
            raise click.ClickException(f"--index must be between 0 and {len(samples) - 1}")
        instance = pd.Series(samples[sample_index])[feature_names]

    approval_probability = float(model.predict_proba(instance.to_frame().T[feature_names])[0, 1])
    shap_values = exact_shapley_values(model, instance, background, feature_names)

    counterfactual = None
    if approval_probability < 0.5:
        counterfactual = find_counterfactual(model, instance, feature_names, feature_ranges)

    report = generate_report(instance, shap_values, approval_probability, counterfactual)
    click.echo(report)


if __name__ == "__main__":
    cli()
