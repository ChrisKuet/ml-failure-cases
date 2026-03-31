from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_regression_predictions(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Evaluate multiple regression predictions and return a tidy DataFrame.
    """
    rows: List[Dict[str, float | str]] = []

    for model_name, y_pred in predictions.items():
        metrics = regression_metrics(y_true, y_pred)
        row = {"model": model_name, **metrics}
        rows.append(row)

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def compare_to_truth_function(f_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare predictions to the underlying true signal, not the noisy response.
    Useful in simulation settings.
    """
    rows = []

    for model_name, y_pred in predictions.items():
        rmse_f = float(np.sqrt(mean_squared_error(f_true, y_pred)))
        mae_f = float(mean_absolute_error(f_true, y_pred))
        rows.append({
            "model": model_name,
            "rmse_vs_truth": rmse_f,
            "mae_vs_truth": mae_f,
        })

    return pd.DataFrame(rows).sort_values("rmse_vs_truth").reset_index(drop=True)
