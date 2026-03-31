from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple
from sklearn.model_selection import train_test_split


def nonlinear_signal(X: np.ndarray) -> np.ndarray:
    """
    Default nonlinear regression signal.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Mean response f(X).
    """
    if X.shape[1] < 5:
        raise ValueError("X must have at least 5 columns for nonlinear_signal().")

    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def normal_errors(n: int, scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=0.0, scale=scale, size=n)


def t_errors(n: int, df: float, scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.standard_t(df=df, size=n) * scale


def skewed_errors(n: int, scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simple skewed error generator using centered chi-square noise.
    """
    rng = np.random.default_rng() if rng is None else rng
    e = rng.chisquare(df=3, size=n) - 3.0
    return e * scale


def simulate_regression(
    n: int = 1000,
    p: int = 10,
    signal_fn: Callable[[np.ndarray], np.ndarray] = nonlinear_signal,
    error_dist: str = "normal",
    error_scale: float = 1.0,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a regression dataset Y = f(X) + epsilon.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of predictors.
    signal_fn : callable
        Function mapping X to f(X).
    error_dist : str
        One of {"normal", "t3", "t1", "skewed"}.
    error_scale : float
        Noise scale multiplier.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n, p).
    y : np.ndarray
        Observed response.
    f : np.ndarray
        True signal values.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    f = signal_fn(X)

    if error_dist == "normal":
        eps = normal_errors(n, scale=error_scale, rng=rng)
    elif error_dist == "t3":
        eps = t_errors(n, df=3, scale=error_scale, rng=rng)
    elif error_dist == "t1":
        eps = t_errors(n, df=1, scale=error_scale, rng=rng)
    elif error_dist == "skewed":
        eps = skewed_errors(n, scale=error_scale, rng=rng)
    else:
        raise ValueError(f"Unsupported error_dist: {error_dist}")

    y = f + eps
    return X, y, f


def make_train_test_regression(
    n: int = 1000,
    p: int = 10,
    test_size: float = 0.3,
    signal_fn: Callable[[np.ndarray], np.ndarray] = nonlinear_signal,
    error_dist: str = "normal",
    error_scale: float = 1.0,
    random_state: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a train/test split for regression data.
    """
    X, y, f = simulate_regression(
        n=n,
        p=p,
        signal_fn=signal_fn,
        error_dist=error_dist,
        error_scale=error_scale,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, f, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "f_train": f_train,
        "f_test": f_test,
    }


def regression_data_to_frame(X: np.ndarray, y: np.ndarray, f: np.ndarray | None = None) -> pd.DataFrame:
    """
    Convert simulated regression data to a pandas DataFrame.
    """
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    df["y"] = y
    if f is not None:
        df["f_true"] = f
    return df
