from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str | None = None,
):
    """
    Create a bar plot comparing one metric across models.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing at least columns ['model', metric].
    metric : str, default='rmse'
        Metric column to plot.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    if "model" not in results_df.columns:
        raise ValueError("results_df must contain a 'model' column.")
    if metric not in results_df.columns:
        raise ValueError(f"results_df must contain a '{metric}' column.")

    df = results_df.copy().sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df[metric])
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(title if title is not None else f"Model Comparison: {metric.upper()}")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig, ax


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 30,
    title: str = "Residual Histogram",
):
    """
    Plot histogram of residuals.
    """
    resid = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=bins)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
):
    """
    Scatter plot of actual vs predicted values with 45-degree reference line.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)

    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], linestyle="--")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_error_by_quantile(
    error_df: pd.DataFrame,
    title: str = "Absolute Error by Prediction Quantile",
):
    """
    Plot mean absolute error against mean predicted value across quantile bins.

    Parameters
    ----------
    error_df : pd.DataFrame
        DataFrame with columns ['mean_pred', 'mean_abs_error'].
    title : str
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    required_cols = {"mean_pred", "mean_abs_error"}
    missing = required_cols - set(error_df.columns)
    if missing:
        raise ValueError(f"error_df is missing required columns: {missing}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(error_df["mean_pred"], error_df["mean_abs_error"], marker="o")
    ax.set_xlabel("Mean Predicted Value")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_multi_model_metric(
    metrics_by_scenario: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "model",
    title: str | None = None,
):
    """
    Plot one metric across scenarios for multiple models.

    Parameters
    ----------
    metrics_by_scenario : pd.DataFrame
        DataFrame containing columns for x, y, and hue.
    x : str
        Column to use on the x-axis (e.g. scenario).
    y : str
        Metric column to use on the y-axis (e.g. rmse).
    hue : str, default='model'
        Column defining model groups.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    required_cols = {x, y, hue}
    missing = required_cols - set(metrics_by_scenario.columns)
    if missing:
        raise ValueError(f"metrics_by_scenario is missing required columns: {missing}")

    df = metrics_by_scenario.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, subdf in df.groupby(hue):
        ax.plot(subdf[x], subdf[y], marker="o", label=model_name)

    ax.set_xlabel(x)
    ax.set_ylabel(y.upper())
    ax.set_title(title if title is not None else f"{y.upper()} across {x}")
    ax.legend()
    plt.tight_layout()
    return fig, ax
