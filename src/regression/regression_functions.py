import re
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
contains helper functions for: 
- formatting titles
- saving regression results and prediction plots
- running manual K-Fold cross-validation 
- saving cross-validation results 
- new: cross-validation function using saved stratified folds
"""


def format_title(name):
    """formats variable names for more readable title."""
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()


def save_regression_outputs(
        model, X_train, X_test, y_test, y_train,
        y_pred_train, y_pred_test,
        metrics,
        task_name, target, output_dir, model_type):
    """
    save evaluation results and prediction plots
    """

    os.makedirs(output_dir, exist_ok=True)

    # map model type names to more readable names
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest Regression"
    }

    if model_type is None:
        model_type_raw = model.__class__.__name__
        model_type_display_name = model_name_mapping.get(model_type_raw, format_title(model_type_raw))
    else:
        model_type_display_name = model_name_mapping.get(model_type, format_title(model_type))

    # formatted names for plot
    format_task_name = format_title(task_name)
    format_target_name = format_title(target)

    # base filename
    base_filename = f"{task_name}_{target}_{model_type}"

    # save evaluation metrics
    eval_path = os.path.join(output_dir, f"{base_filename}_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write(f"Regression Evaluation ({model_type_display_name}) for {task_name} → {target}\n")
        f.write(f"R²: {metrics['R2']:.3f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.3f}\n")
        f.write(f"MAE: {metrics['MAE']:.3f}\n")

    print(f"evaluation metrics saved to: {eval_path}")

    # create predicted vs. actual plot
    matplotlib.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, s=20, alpha=0.7, color='steelblue')
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--', color='gray'
    )

    r_squared = metrics['R2']

    # legend and labels
    custom_legend = [Patch(color='gray', label='Perfect Prediction')]
    plt.legend(handles=custom_legend, loc="upper left", fontsize=10, frameon=True )

    plt.text(
        x=0.05, y=0.90,
        s=f"$R^2$ = {r_squared:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
    )

    plt.xlabel("Actual Score", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Predicted Score", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(f"{format_task_name}: Predicted vs. Actual {format_target_name}\n({model_type_display_name})", fontsize=14,
              fontweight="bold", pad=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plot_path = os.path.join(output_dir, f"{base_filename}_prediction_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"saved prediction plot to: {plot_path}")

    # plot 2: Train and Test combined
    plt.figure(figsize=(6, 6))

    # scatter plots
    plt.scatter(y_train, y_pred_train, s=20, alpha=0.6, color='steelblue', label="Train")
    plt.scatter(y_test, y_pred_test, s=20, alpha=0.8, color='darkorange', label="Test")

    # diagonal line for perfect prediction
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='Perfect Prediction')

    # labels and title
    plt.xlabel("Actual Score", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Score", fontsize=12, fontweight="bold")
    plt.title(f"{format_task_name}: Train & Test Predictions\n({model_type_display_name})", fontsize=14,
              fontweight="bold")

    # legend with all 3 elements
    plt.legend(loc="upper left", fontsize=10, frameon=True)

    # add R² as separate annotation below the legend
    plt.text(
        x=0.05, y=0.78,
        s=f"Test $R^2$ = {metrics['R2']:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10
    )

    plt.grid(True)

    plot_path_combined = os.path.join(output_dir, f"{base_filename}_train_test_plot.png")
    plt.savefig(plot_path_combined, dpi=300)
    plt.close()
    print(f"saved train+test combined prediction plot to:\n{plot_path_combined}")



def save_crossval_results(
    r2_list, rmse_list, mae_list,
    r2_mean, r2_std, r2_se, r2_ci_low, r2_ci_high,
    task_name, target, output_dir,
    all_preds=None, # optional: pass predictions for plotting
    model_type="LinearRegression"

):

    os.makedirs(output_dir, exist_ok=True)

    # save per-fold results
    cv_df = pd.DataFrame({
        "Fold": list(range(1, len(r2_list) + 1)),
        "R2": r2_list,
        "RMSE": rmse_list,
        "MAE": mae_list
    })
    cv_df_path = os.path.join(output_dir, f"cv_folds_{task_name}_{target}_{model_type}.csv")
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
    summary_path = os.path.join(output_dir, f"cv_summary_{task_name}_{target}_{model_type}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"saved per-fold CV results to:\n{cv_df_path}")
    print(f"saved CV summary to:\n{summary_path}")

    # save scatterplot
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

        plot_path = os.path.join(output_dir, f"cv_prediction_plot_{task_name}_{target}_{model_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"CV prediction plot saved to: {plot_path}")



def stratified_cross_validation(
    df, fold_column, model_class, model_params,
    target_column, n_folds=5, feature_columns=None, model_name=""
):
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    all_preds = []

    for fold in range(1, n_folds + 1):
        train_df = df[df[fold_column] != fold] # all the other folds
        test_df = df[df[fold_column] == fold] # current fold

        # train test split (80/20)
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # standardize
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # train model
        model = model_class(**model_params) if model_params else model_class()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # model evaluation
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        fold_df = pd.DataFrame({
            "y_test": y_test.values,
            "y_pred": y_pred,
            "fold": fold,
            "model": model_name
        })
        all_preds.append(fold_df)

        print(f"Fold {fold}: R² = {r2:.3f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")

    all_preds_df = pd.concat(all_preds, ignore_index=True)

    return r2_scores, rmse_scores, mae_scores, all_preds_df
