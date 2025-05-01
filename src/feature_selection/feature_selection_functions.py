# steps: 1) load features & preprocessed scores, 2) correlation-matrix, 3) VIF, 4) forward-selection, 5) save selected features for each task-score-combination, 6) compare all selected features, 7) build general feature set

# setup
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.patches import Patch


def compute_correlation_matrix(X_scaled, y, task_name, target, output_dir):
    """
    Saves the correlation matrix CSV and heatmap.
    """
    X_with_target = X_scaled.copy()
    X_with_target[target] = y.values

    corr = X_with_target.corr()

    # Save matrix as CSV
    csv_path = os.path.join(output_dir, f"correlation_matrix_{task_name}_{target}.csv")
    corr.to_csv(csv_path)

    # Plot heatmap
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title(f"Correlation Matrix with {target}", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"full_correlation_matrix_{task_name}_{target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation matrix CSV to:\n{csv_path}")
    print(f"Saved correlation matrix plot to:\n{plot_path}")
    return corr


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    def categorize_vif(vif_value):
        if vif_value < 5:
            return "Low"
        elif vif_value < 10:
            return "Moderate"
        else:
            return "High"

    vif_data["VIF_Category"] = vif_data["VIF"].apply(categorize_vif)
    return vif_data


def forward_selection(X, y, task_name, target, output_dir, verbose=True):
    """
    Perform forward feature selection using adjusted R², AIC, and BIC.
    Saves selected features and returns them with a performance summary.
    """

    remaining_features = list(X.columns)
    selected_features = []
    current_score, best_new_score = -float("inf"), -float("inf")
    summary = []

    while remaining_features:
        scores_with_candidates = []

        for candidate in remaining_features:
            features_to_test = selected_features + [candidate]
            X_const = sm.add_constant(X[features_to_test].reset_index(drop=True))
            y_aligned = y.reset_index(drop=True)
            model = sm.OLS(y_aligned, X_const).fit()

            scores_with_candidates.append({
                "feature": candidate,
                "R2_adj": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic
            })

        # Convert to DataFrame
        scores_df = pd.DataFrame(scores_with_candidates)

        # Select the best candidate (based on R² adjusted)
        best_candidate = scores_df.sort_values("R2_adj", ascending=False).iloc[0]

        if best_candidate["R2_adj"] > current_score:
            remaining_features.remove(best_candidate["feature"])
            selected_features.append(best_candidate["feature"])
            current_score = best_candidate["R2_adj"]
            summary.append(best_candidate)
            if verbose:
                print(f"Added: {best_candidate['feature']} | R²_adj: {best_candidate['R2_adj']:.4f}")
        else:
            if verbose:
                print("No improvement. Stopping.")
            break

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)

    # Save selected features
    os.makedirs(output_dir, exist_ok=True)
    filename = f"selected_features_{task_name}_{target}.csv"
    summary_df.to_csv(os.path.join(output_dir, filename), index=False)

    # Final model with selected features
    X_final = sm.add_constant(X[selected_features].reset_index(drop=True))
    y_final = y.reset_index(drop=True)
    final_model = sm.OLS(y_final, X_final).fit()

    return selected_features, summary_df, final_model



def evaluate_on_test_set(X_train, y_train, X_test, y_test, selected_features, task_name, target, output_dir):
    """
    Trains a Linear Regression model on selected features (X_train) and evaluates on X_test.
    Saves R², RMSE, and MAE as a CSV to output_dir.
    """

    # select only relevant features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # train model
    model = LinearRegression().fit(X_train_sel, y_train)

    # predict
    y_pred = model.predict(X_test_sel)

    # evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # save metrics
    metrics_df = pd.DataFrame({
        "task": [task_name],
        "target": [target],
        "R2": [r2],
        "RMSE": [rmse],
        "MAE": [mae]
    })

    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, f"model_test_evaluation_{task_name}_{target}.csv")
    metrics_df.to_csv(eval_path, index=False)

    print(f"final test evaluation:")
    print(metrics_df.to_string(index=False))
    print(f"saved to: {eval_path}")

    # plot: Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.rcParams["font.family"] = "Arial"
    plt.scatter(y_test, y_pred, color="steelblue", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             linestyle='--', color='gray', label="Perfect Prediction")

    plt.xlabel("Actual Score", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Score", fontsize=12, fontweight="bold")
    plt.title(f"{task_name} – {target}: Test Set Prediction", fontsize=14, fontweight="bold")
    plt.text(x=0.05, y=0.9, s=f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=10)
    plt.legend(loc="upper left", fontsize=10, frameon=True)

    plot_path = os.path.join(output_dir, f"prediction_plot_{task_name}_{target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"prediction plot saved to: {plot_path}")

    # combined scatterplot for train and test
    y_train_pred = model.predict(X_train_sel)

    plt.figure(figsize=(6, 6))
    plt.rcParams["font.family"] = "Arial"
    plt.scatter(y_train, y_train_pred, color="steelblue", alpha=0.6, label="Train")
    plt.scatter(y_test, y_pred, color="darkorange", alpha=0.8, label="Test")
    plt.plot([min(y_test.min(), y_train.min()), max(y_test.max(), y_train.max())],
             [min(y_test.min(), y_train.min()), max(y_test.max(), y_train.max())],
             linestyle='--', color='gray', label="Perfect Prediction")

    plt.xlabel("Actual Score", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Score", fontsize=12, fontweight="bold")
    plt.title(f"{task_name} – {target}: Train vs. Test Predictions", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10, frameon=True)

    plot_path_combined = os.path.join(output_dir, f"prediction_plot_train_test_{task_name}_{target}.png")
    plt.savefig(plot_path_combined, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"combined train+test prediction plot saved to: {plot_path_combined}")


    return model, y_pred, metrics_df

import pandas as pd
import os


def save_crossval_results(
    r2_list, rmse_list, mae_list,
    r2_mean, r2_std, r2_se, r2_ci_low, r2_ci_high,
    task_name, target, output_dir,
    all_preds=None  # optional: pass predictions for plotting
):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save per-fold results
    cv_df = pd.DataFrame({
        "Fold": list(range(1, len(r2_list) + 1)),
        "R2": r2_list,
        "RMSE": rmse_list,
        "MAE": mae_list
    })
    cv_df_path = os.path.join(output_dir, f"cv_folds_{task_name}_{target}.csv")
    cv_df.to_csv(cv_df_path, index=False)

    # 2. Save summary
    summary_df = pd.DataFrame({
        "R2_Mean": [r2_mean],
        "R2_Std": [r2_std],
        "R2_SE": [r2_se],
        "R2_CI_Low": [r2_ci_low],
        "R2_CI_High": [r2_ci_high],
        "RMSE_Mean": [np.mean(rmse_list)],
        "MAE_Mean": [np.mean(mae_list)]
    })
    summary_path = os.path.join(output_dir, f"cv_summary_{task_name}_{target}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"saved per-fold CV results to:\n{cv_df_path}")
    print(f"saved CV summary to:\n{summary_path}")

    # 3. Save scatterplot
    if all_preds is not None:
        plt.figure(figsize=(7, 6))
        for fold in all_preds['fold'].unique():
            fold_df = all_preds[all_preds['fold'] == fold]
            plt.scatter(fold_df["y_test"], fold_df["y_pred"], label=f"Fold {fold + 1}", alpha=0.7)

        plt.plot(
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            linestyle='--', color='gray', label="Perfect Prediction"
        )

        plt.xlabel("Actual Score", fontsize=12, fontweight="bold")
        plt.ylabel("Predicted Score", fontsize=12, fontweight="bold")
        plt.title(f"{task_name.title()}: Cross-Validated Predictions", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"cv_prediction_plot_{task_name}_{target}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"CV prediction plot saved to: {plot_path}")







