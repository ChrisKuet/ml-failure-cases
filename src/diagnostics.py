from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute residuals y - yhat.
    """
    return y_true - y_pred


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Summarize residual behavior.
    """
    r = residuals(y_true, y_pred)
    return {
        "mean_residual": float(np.mean(r)),
        "std_residual": float(np.std(r, ddof=1)),
        "median_residual": float(np.median(r)),
        "max_abs_residual": float(np.max(np.abs(r))),
        "q90_abs_residual": float(np.quantile(np.abs(r), 0.90)),
        "q95_abs_residual": float(np.quantile(np.abs(r), 0.95)),
    }


def diagnostics_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Create a diagnostics summary table for multiple models.
    """
    rows = []
    for model_name, y_pred in predictions.items():
        row = {"model": model_name, **residual_summary(y_true, y_pred)}
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def error_by_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_groups: int = 10,
) -> pd.DataFrame:
    """
    Summarize absolute error by quantiles of fitted values.
    Useful for checking whether error worsens in certain prediction regions.
    """
    pred = np.asarray(y_pred)
    abs_err = np.abs(np.asarray(y_true) - pred)

    df = pd.DataFrame({"pred": pred, "abs_error": abs_err})
    df["bin"] = pd.qcut(df["pred"], q=n_groups, duplicates="drop")

    out = (
        df.groupby("bin", observed=False)
        .agg(mean_pred=("pred", "mean"), mean_abs_error=("abs_error", "mean"), n=("abs_error", "size"))
        .reset_index(drop=True)
    )
    return out
