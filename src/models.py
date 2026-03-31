from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_regression_models(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """
    Return a dictionary of regression models for comparison.
    """
    models: Dict[str, RegressorMixin] = {
        "ols": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=10000))
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }
    return models


def get_classification_models(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """
    Return a dictionary of classification models for later experiments.
    """
    models: Dict[str, ClassifierMixin] = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }
    return models


def fit_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Fit all models in a dictionary and return the fitted versions.
    """
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def predict_models(fitted_models: Dict[str, Any], X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate predictions from fitted models.
    """
    preds = {}
    for name, model in fitted_models.items():
        preds[name] = model.predict(X_test)
    return preds


def predict_prob_models(fitted_models: Dict[str, Any], X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate predicted probabilities for classification models.
    """
    probs = {}
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            probs[name] = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Model '{name}' does not support predict_proba().")
    return probs
