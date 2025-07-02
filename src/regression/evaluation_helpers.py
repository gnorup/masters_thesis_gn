import re
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

def format_title(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals'
    return name.title().strip()

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

def plot_per_fold_predictions(all_preds_df, task_name, target, model_type, output_dir, n_folds=5):

    # map model types for better readability in titles
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest Regression"
    }
    model_type_display_name = model_name_mapping.get(model_type.__name__, model_type.__name__)
    format_task_name = format_title(task_name)
    format_target_name = format_title(target)

    matplotlib.rcParams['font.family'] = 'Arial'

    # create plot
    for fold in range(1, n_folds + 1):
        fold_df = all_preds_df[all_preds_df['fold'] == fold]
        y_test, y_pred = fold_df['y_test'], fold_df['y_pred']
        r2 = r2_score(y_test, y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, s=20, alpha=0.7, color='steelblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='gray')

        # legend and labels
        custom_legend = [plt.Line2D([0], [0], color='gray', linestyle='--', label='Perfect Prediction')]
        plt.legend(handles=custom_legend, loc="upper left", fontsize=10, frameon=True)
        plt.text(x=0.05, y=0.90, s=f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=10)

        plt.xlabel("Actual Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Predicted Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f"Fold {fold}: Predicted vs. Actual {format_target_name}\n({model_type_display_name}, {format_task_name})", fontsize=14, fontweight='bold', pad=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plot_path = os.path.join(output_dir, f"fold{fold}_actual_vs_predicted_{target}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"plot: actual vs predicted scores for fold {fold} saved to {plot_path}")
