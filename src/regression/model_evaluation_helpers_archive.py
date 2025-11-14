# ** TO DO delete unused imports in this file **
import sys
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY

# format title for plots
def format_title(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

# calculate 95%-confidence intervals
def confidence_intervals(samples, ci=default_ci):
    arr = np.asarray(samples, dtype=float)
    alpha = 1.0 - ci
    return np.nanquantile(arr, alpha / 2), np.nanquantile(arr, 1 - alpha / 2)

# compute BCa CI with scipy
def bca_ci(stat_fn, data_tuple, ci=default_ci, n_resamples=default_nboot, random_state=default_randomstate):
    res = bootstrap(
        data=data_tuple,
        statistic=lambda *arrs: stat_fn(*arrs),
        paired=True,
        axis=0,
        n_resamples=n_resamples,
        confidence_level=ci,
        method="BCa",
        vectorized=False,
        random_state=random_state,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)

# from old cross-validation setup
def save_crossval_results(
        r2_list, rmse_list, mae_list,
        r2_mean, r2_std, r2_se, r2_ci_low, r2_ci_high,
        task_name, target, output_dir, n_folds=5,
        all_preds=None,
        model_type=None
):

    os.makedirs(output_dir, exist_ok=True)

    # save per-fold results
    cv_df = pd.DataFrame({
        "Fold": list(range(1, n_folds + 1)),
        "R2": r2_list,
        "RMSE": rmse_list,
        "MAE": mae_list,
    })
    cv_df_path = os.path.join(output_dir, f"cv_folds_{task_name}_{target}_{model_type.__name__}.csv")
    cv_df.to_csv(cv_df_path, index=False)

    # save summary
    summary_df = pd.DataFrame({
        "R2_Mean": [r2_mean],
        "R2_Std": [r2_std],
        "R2_SE": [r2_se],
        "R2_CI_Low": [r2_ci_low],
        "R2_CI_High": [r2_ci_high],
        "RMSE_Mean": [np.mean(rmse_list)],
        "MAE_Mean": [np.mean(mae_list)]
    })
    summary_path = os.path.join(output_dir, f"cv_summary_{task_name}_{target}_{model_type.__name__}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"saved per-fold cross-validation results to: {cv_df_path}")
    print(f"saved cross-validation summary to: {summary_path}")

    # optional: save scatterplot
    matplotlib.rcParams['font.family'] = 'Arial'
    # map model types for better readability in titles
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest Regression"
    }
    model_type_display_name = model_name_mapping.get(model_type.__name__, model_type.__name__)
    formatted_task_name = format_title(task_name)

    if all_preds is not None:
        plt.figure(figsize=(7, 6))
        for fold in range(1, n_folds + 1):
            fold_df = all_preds[all_preds['fold'] == fold]
            plt.scatter(fold_df["y_test"], fold_df["y_pred"], label=f"Fold {fold}", alpha=0.7)

        plt.plot(
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            linestyle='--', color='gray', label="Perfect Prediction"
        )

        plt.xlabel("Actual Score", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted Score", fontsize=12, fontweight='bold')
        plt.title(f"{formatted_task_name}: Cross-Validated Predictions ({model_type_display_name})", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"cv_prediction_plot_{task_name}_{target}_{model_type.__name__}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"cross-validation prediction plot saved to {plot_path}")