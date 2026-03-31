from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_generation import make_train_test_regression
from src.models import get_regression_models, fit_models, predict_models
from src.evaluation import evaluate_regression_predictions, compare_to_truth_function
from src.diagnostics import diagnostics_table
from src.plotting import plot_metric_comparison, plot_multi_model_metric
from src.utils import ensure_dir, set_seed


def run_single_scenario(
    error_dist: str,
    n: int = 1000,
    p: int = 10,
    test_size: float = 0.3,
    error_scale: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one heavy-tail scenario and return:
    1. performance against observed y
    2. performance against true signal f(x)
    3. residual diagnostics
    """
    data = make_train_test_regression(
        n=n,
        p=p,
        test_size=test_size,
        error_dist=error_dist,
        error_scale=error_scale,
        random_state=random_state,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    f_test = data["f_test"]

    models = get_regression_models(random_state=random_state)
    fitted_models = fit_models(models, X_train, y_train)
    predictions = predict_models(fitted_models, X_test)

    performance_y = evaluate_regression_predictions(y_test, predictions)
    performance_y["scenario"] = error_dist

    performance_f = compare_to_truth_function(f_test, predictions)
    performance_f["scenario"] = error_dist

    diagnostics = diagnostics_table(y_test, predictions)
    diagnostics["scenario"] = error_dist

    return performance_y, performance_f, diagnostics


def save_plot(fig, filepath: Path) -> None:
    """
    Save a matplotlib figure and close it.
    """
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    fig.clf()


def main() -> None:
    set_seed(42)

    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    ensure_dir(str(results_dir))
    ensure_dir(str(figures_dir))
    ensure_dir(str(tables_dir))

    scenarios = ["normal", "t3", "t1", "skewed"]

    all_perf_y = []
    all_perf_f = []
    all_diagnostics = []

    print("Running heavy-tail experiment...\n")

    for scenario in scenarios:
        print(f"Scenario: {scenario}")
        perf_y, perf_f, diag = run_single_scenario(
            error_dist=scenario,
            n=1000,
            p=10,
            test_size=0.3,
            error_scale=1.0,
            random_state=42,
        )

        all_perf_y.append(perf_y)
        all_perf_f.append(perf_f)
        all_diagnostics.append(diag)

        # Per-scenario RMSE figure
        fig, _ = plot_metric_comparison(
            perf_y.sort_values("rmse"),
            metric="rmse",
            title=f"Test RMSE by Model ({scenario} errors)",
        )
        save_plot(fig, figures_dir / f"rmse_{scenario}.png")

    # Combine results
    perf_y_df = pd.concat(all_perf_y, ignore_index=True)
    perf_f_df = pd.concat(all_perf_f, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)

    # Save tables
    perf_y_df.to_csv(tables_dir / "heavy_tails_performance_vs_observed.csv", index=False)
    perf_f_df.to_csv(tables_dir / "heavy_tails_performance_vs_truth.csv", index=False)
    diagnostics_df.to_csv(tables_dir / "heavy_tails_diagnostics.csv", index=False)

    # Summary tables
    rmse_summary = perf_y_df.pivot(index="scenario", columns="model", values="rmse")
    mae_summary = perf_y_df.pivot(index="scenario", columns="model", values="mae")
    truth_rmse_summary = perf_f_df.pivot(index="scenario", columns="model", values="rmse_vs_truth")

    rmse_summary.to_csv(tables_dir / "heavy_tails_rmse_summary.csv")
    mae_summary.to_csv(tables_dir / "heavy_tails_mae_summary.csv")
    truth_rmse_summary.to_csv(tables_dir / "heavy_tails_truth_rmse_summary.csv")

    # Cross-scenario plots
    perf_y_ordered = perf_y_df.copy()
    perf_y_ordered["scenario"] = pd.Categorical(
        perf_y_ordered["scenario"],
        categories=scenarios,
        ordered=True,
    )
    perf_y_ordered = perf_y_ordered.sort_values(["model", "scenario"])

    fig, _ = plot_multi_model_metric(
        metrics_by_scenario=perf_y_ordered,
        x="scenario",
        y="rmse",
        hue="model",
        title="RMSE Across Error Distributions",
    )
    save_plot(fig, figures_dir / "rmse_across_scenarios.png")

    fig, _ = plot_multi_model_metric(
        metrics_by_scenario=perf_y_ordered,
        x="scenario",
        y="mae",
        hue="model",
        title="MAE Across Error Distributions",
    )
    save_plot(fig, figures_dir / "mae_across_scenarios.png")

    print("\nFinished.")
    print(f"Saved tables to:  {tables_dir}")
    print(f"Saved figures to: {figures_dir}")

    print("\nPreview: performance against observed response")
    print(perf_y_df.sort_values(["scenario", "rmse"]).to_string(index=False))

    print("\nPreview: performance against true signal")
    print(perf_f_df.sort_values(["scenario", "rmse_vs_truth"]).to_string(index=False))


if __name__ == "__main__":
    main()
