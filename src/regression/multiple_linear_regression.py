import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def format_title(name):
    """Formats variable names for more readable title."""
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Trains a linear regression model and evaluates it on test data.
    """

    # create and train  model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict on test data
    y_pred = model.predict(X_test)

    # evaluate model
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return model, y_pred, r2, rmse, mae


def run_multiple_regression(
        features_path,
        scores_path,
        target,
        output_dir,
        task_name,
        id_column="Subject_ID",
        save_outputs=True,
        test_size=0.2,    # proportion of data for test set in train/test split (default 0.2 = 20%)
        random_state=42   # random seed for reproducibility (42 as default)
):
    """
    """

    # load data
    features = pd.read_csv(features_path)
    scores = pd.read_csv(scores_path)

    # merge on subject ID
    df = pd.merge(features, scores[[id_column, target]], on=id_column)
    df = df.dropna()

    # prepare variables
    X_raw = df.drop(columns=[id_column, target])  # input features
    y = df[target]  # score to predict

    # standardize features (for scikit)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # train and evaluate
    model, y_pred, r2, rmse, mae = train_and_evaluate(X_train, y_train, X_test, y_test)

    # base filename for saving results
    base_filename = f"{task_name}_{target}"

    # save regression output and plot (optional)
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

        eval_path = os.path.join(output_dir, f"{base_filename}_split_evaluation.txt")
        with open(eval_path, "w") as f:
            f.write(f"Train/Test Split Evaluation for {task_name} → {target}\n")
            f.write(f"R²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\n")

        # plot
        matplotlib.rcParams['font.family'] = 'Arial'

        # formatted names for the plot
        format_task = format_title(task_name)
        format_target = format_title(target)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, s=20, alpha=0.7, color='steelblue')
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle='-',
            linewidth=2,
            color='gray',
            label='Perfect Prediction'
        )

        # R² value
        r_squared = r2

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

    return model, X_scaled, y, X_train, X_test, y_train, y_test
