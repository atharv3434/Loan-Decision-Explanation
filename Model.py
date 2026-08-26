"""Trains the loan approval classifier used throughout the tool. Gradient
boosting is a realistic choice — this is genuinely the kind of model class
used in real underwriting systems (nonlinear, handles feature interactions),
which is exactly why post-hoc explanation methods like Shapley values matter
here: unlike a plain logistic regression, you can't just read the
coefficients to see what the model is doing.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def train_model(X, y, random_state: int = 42, test_size: float = 0.2):
    """Returns (model, X_train, X_test, y_train, y_test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test