import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import re
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer



def format_title(name):
    """Formats variable names for more readable title."""
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

def run_multiple_regression(
        features_path,
        scores_path,
        target,
        output_dir,
        task_name,
        id_column="Subject_ID",
        save_outputs=True,
        evaluation=None,  # "split" = train/test split, "cv" = k-fold cross-validation
        cv_folds=5,       # number of folds for cross-validation (default 5)
        test_size=0.2,    # proportion of data for test set in train/test split (default 0.2 = 20%)
        random_state=42   # random seed for reproducibility (42 as default)
):
    """
    Run multiple linear regression on selected features and a language score,
    with optional evaluation via train/test split or cross-validation.

    Parameters:
    - features_path (str): Path to CSV file with input features.
    - scores_path (str): Path to CSV file with language test scores.
    - target (str): Name of the score column to predict.
    - output_dir (str): Directory to save outputs (plots, CSVs, summaries).
    - task_name (str): Name of the task (used in filenames and plots).
    - id_column (str): Column used to merge data (default: "Subject_ID").
    - save_outputs (bool): Whether to save outputs (default: True).
    - evaluation (str or None): "split" for train/test split or "cv" for cross-validation.
    - cv_folds (int): Number of folds for cross-validation (default: 5).
    - test_size (float): Proportion of test data in train/test split (default: 0.2).
    - random_state (int): Seed for reproducibility (default: 42).

    Returns:
    - If evaluation is None: statsmodels OLS model, results_df, VIF, X, y.
    - If evaluation is "split" or "cv": prints and optionally saves evaluation metrics.
    """

    # load data
    features = pd.read_csv(features_path)
    scores = pd.read_csv(scores_path)

    # merge on subject ID
    df = pd.merge(features, scores[[id_column, target]], on=id_column)
    df = df.dropna()

    # prepare variables
    X = df.drop(columns=[id_column, target])  # input features
    y = df[target]  # score to predict
    X = sm.add_constant(X)  # adds intercept term to regression

    # base filename for saving results
    base_filename = f"{task_name}_{target}"

    # optional: Evaluation mode for train/test split or cross-validation
    if evaluation == "split":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model_split = LinearRegression().fit(X_train, y_train)
        y_pred = model_split.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        if save_outputs:
            eval_path = os.path.join(output_dir, f"{task_name}_{target}_split_evaluation.txt")
            with open(eval_path, "w") as f:
                f.write(f"Train/Test Split Evaluation for {task_name} → {target}\n")
                f.write(f"R²: {r2:.3f}\n")
                f.write(f"RMSE: {rmse:.3f}\n")

        print(f"Train/Test Split — R²: {r2:.3f}, RMSE: {rmse:.3f}")

        return None # avoids running full regression afterward

    elif evaluation == "cv":
        model_cv = LinearRegression()
        # R²
        r2_scores = cross_val_score(model_cv, X, y, cv=cv_folds, scoring='r2')
        # MSE
        mse_scores = cross_val_score(
            model_cv, X, y, cv=cv_folds,
            scoring=make_scorer(mean_squared_error, greater_is_better=False)
        )
        #  MAE
        mae_scores = cross_val_score(
            model_cv, X, y, cv=cv_folds,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False)
        )

        if save_outputs:
            eval_path = os.path.join(output_dir, f"{task_name}_{target}_crossval_evaluation.txt")
            with open(eval_path, "w") as f:
                f.write(f"{cv_folds}-Fold Cross-Validation for {task_name} → {target}\n")
                f.write("--------------------------------------------------\n")
                for i in range(cv_folds):
                    f.write(f"Fold {i + 1}:\n")
                    f.write(f"  R²  = {r2_scores[i]:.3f}\n")
                    f.write(f"  MSE = {-mse_scores[i]:.3f}\n")
                    f.write(f"  MAE = {-mae_scores[i]:.3f}\n\n")
                f.write("--------------------------------------------------\n")
                f.write(f"Mean R² : {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}\n")
                f.write(f"Mean MSE: {abs(np.mean(mse_scores)):.3f} ± {np.std(mse_scores):.3f}\n")
                f.write(f"Mean MAE: {abs(np.mean(mae_scores)):.3f} ± {np.std(mae_scores):.3f}\n")

        print(f"{cv_folds}-Fold CV Results for {task_name} → {target}")
        print(f"  Mean R²  = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Mean MSE = {abs(np.mean(mse_scores)):.3f} ± {np.std(mse_scores):.3f}")
        print(f"  Mean MAE = {abs(np.mean(mae_scores)):.3f} ± {np.std(mae_scores):.3f}")

        return None  # avoids running full regression afterward

    # fit regression model using statsmodels
    model = sm.OLS(y, X).fit()

    # extract confidence intervals
    conf_int = model.conf_int()
    conf_int.columns = ["conf_low", "conf_high"]

    # standardize features for standardized coefficients
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X.drop(columns=["const"]))
    X_std = sm.add_constant(X_std)
    model_std = sm.OLS(y, X_std).fit()

    # coefficients and p-values
    results_df = pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "standardized_coefficient": model_std.params.values,
        "p_value": model.pvalues.values,
        "std_err": model.bse.values,
        "t_value": model.tvalues.values
    })

    # add confidence intervals
    results_df = pd.concat([results_df.set_index("feature"), conf_int], axis=1).reset_index()

    # calculate multicollinearity (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # save regression output and plot (optional)
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

        # save full regression output as CSV
        coef_path = os.path.join(output_dir, f"{base_filename}_regression_coefficients.csv")
        results_df.to_csv(coef_path, index=False)

        # save regression summary
        summary_path = os.path.join(output_dir, f"{base_filename}_regression_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Task: {task_name}\n")
            f.write(f"Target: {target}\n\n")
            f.write(model.summary().as_text())

        # plot
        matplotlib.rcParams['font.family'] = 'Arial'

        # formatted names for the plot
        format_task = format_title(task_name)
        format_target = format_title(target)

        plt.figure(figsize=(6, 6))
        plt.scatter(y, model.predict(X), s=20, alpha=0.7, color='steelblue')
        plt.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            linestyle='-',
            linewidth=2,
            color='gray',
            label='Perfect Prediction'
        )

        # R² value
        r_squared = model.rsquared

        # legend
        custom_legend = [Patch(color='gray', label='Perfect Prediction')]
        plt.legend(handles=custom_legend, loc="upper left", fontsize=10, frameon=True)

        # R² text directly on plot (just below legend box)
        plt.text(
            x=0.05, y=0.90,
            s=f"$R^2$ = {r_squared:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10
        )

        # axes and title
        plt.xlabel("Predicted Score", fontsize=12, fontweight="bold", labelpad=10)
        plt.ylabel("Actual Score", fontsize=12, fontweight="bold", labelpad=10)
        plt.title(f"{format_task}: Predicted vs. Actual {format_target}", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)

        # save plot
        plot_path = os.path.join(output_dir, f"{base_filename}_prediction_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # save multicollinearity (VIF)
        vif_path = os.path.join(output_dir, f"{base_filename}_vif.csv")
        vif_data.to_csv(vif_path, index=False)

    return model, results_df, vif_data, X, y



def plot_single_feature(df, feature, target, output_dir=None, task_name=None):
    """
    Creates a scatter plot for a single feature vs. target score.

    Arguments:
    - df: DataFrame with your data
    - feature: name of the feature column
    - target: name of the target column (e.g., language score)
    - output_dir: where to save the plot (optional)
    - task_name: optional name of the task (for labeling)
    """
    os.makedirs(output_dir, exist_ok=True)

    # labels
    feature_label = format_title(feature)
    target_label = format_title(target)
    task_label = format_title(task_name) if task_name else ""

    x = df[feature]
    y = df[target]

    # fit regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

    plt.figure(figsize=(6, 6))
    plt.scatter(df[feature], df[target], alpha=0.7, color='steelblue', s=20)
    plt.plot(x, line, color="darkred", linewidth=2, linestyle="-", label=f"Regression line (R²={r_value ** 2:.2f})")

    plt.xlabel(feature_label, fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel(target_label, fontsize=12, fontweight="bold", labelpad=10)
    title = f"{task_label}: {feature_label} vs. {target_label}" if task_label else f"{feature_label} vs. {target_label}"
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plot.legend(fontsize=10)

    filename = f"{task_name}_{feature}_vs_{target}.png" if task_name else f"{feature}_vs_{target}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def check_regression_assumptions(model, X, y, output_dir):
    """
    Generate diagnostic plots and statistics for linear regression assumptions:
    - Residuals vs. Fitted
    - Histogram of residuals
    - Q-Q Plot
    - Durbin-Watson test for autocorrelation

    Saves all plots and a text file summarizing the assumption checks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # calculate residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid
    standardized_residuals = model.get_influence().resid_studentized_internal

    # Plot: Residuals vs Fitted
    plt.figure(figsize=(6, 4))
    plt.scatter(fitted_vals, residuals, alpha=0.7, edgecolor='k')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_vs_fitted.png"), dpi=300)
    plt.close()

    # Plot: Histogram of residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.75)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_histogram.png"), dpi=300)
    plt.close()

    # Plot: Q-Q Plot
    plt.figure(figsize=(6, 4))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot.png"), dpi=300)
    plt.close()

    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)

    # save summary
    summary_path = os.path.join(output_dir, "assumption_checks_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Regression Assumption Checks\n")
        f.write("==============================\n")
        f.write(f"Durbin-Watson Statistic: {dw_stat:.4f}\n")
        f.write("\n")
        f.write("See generated plots for:")
        f.write("\n - Residuals vs. Fitted")
        f.write("\n - Histogram of Residuals")
        f.write("\n - Q-Q Plot")

    print(f"Assumption checks saved to: {output_dir}")
