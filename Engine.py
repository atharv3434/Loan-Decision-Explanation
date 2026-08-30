"""Trains the model and persists everything the explainer needs later:
the model itself, a background sample for Shapley marginalization, and
each feature's observed range for the counterfactual search.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from decision_explainer.config import Config
from decision_explainer.data import FEATURE_NAMES, generate_applicants
from decision_explainer.model import train_model

logger = logging.getLogger(__name__)


def train_and_save(config: Config) -> dict:
    df = generate_applicants(config.n_samples, config.random_state)
    X, y = df[FEATURE_NAMES], df["approved"]

    model, X_train, X_test, y_train, y_test = train_model(X, y, random_state=config.random_state)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Trained model: test accuracy = {accuracy:.3f}")

    background = X_train.sample(config.background_size, random_state=config.random_state)
    feature_ranges = {f: (float(X_train[f].min()), float(X_train[f].max())) for f in FEATURE_NAMES}

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / config.model_name
    joblib.dump({
        "model": model,
        "background": background,
        "feature_ranges": feature_ranges,
        "feature_names": FEATURE_NAMES,
    }, bundle_path)
    logger.info(f"Saved model bundle to {bundle_path}")

    sample_path = model_dir / "sample_applicants.json"
    sample_records = X_test.head(20).reset_index(drop=True).to_dict(orient="records")
    with open(sample_path, "w") as f:
        json.dump(sample_records, f, indent=2)
    logger.info(f"Saved {len(sample_records)} sample applicants to {sample_path}")

    return {"accuracy": float(accuracy), "n_train": len(X_train), "n_test": len(X_test)}


def load_bundle(config: Config) -> dict:
    bundle_path = Path(config.model_dir) / config.model_name
    if not bundle_path.exists():
        raise FileNotFoundError(f"No trained model found at {bundle_path}. Run `decision-explainer train` first.")
    return joblib.load(bundle_path)
