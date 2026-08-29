"""Configuration for the decision explainer pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    n_samples: int = 3000
    random_state: int = 42
    background_size: int = 30
    model_dir: str = "checkpoints"
    model_name: str = "model.joblib"
