import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib


def format_title(name):
    """Formats variable names for more readable title."""
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

    # map model type names to nice readable names
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest"
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
    plt.title(f"{format_task_name}: Predicted vs. Actual {format_target_name}\n({model_type_display_name} Regression)", fontsize=14,
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
    plt.scatter(y_train, y_pred_train, s=20, alpha=0.6, color='steelblue', label="Train")
    plt.scatter(y_test, y_pred_test, s=20, alpha=0.8, color='darkorange', label="Test")

    plt.plot(
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        linestyle='--', color='gray'
    )

    plt.xlabel("Actual Score", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Score", fontsize=12, fontweight="bold")
    plt.title(f"{format_task_name}: Train & Test Predictions\n({model_type_display_name} Regression)", fontsize=14,
              fontweight="bold")

    plt.legend(loc="upper left", fontsize=10, frameon=True)

    # add R² text (for Test set only)
    plt.text(
        x=0.05, y=0.90,
        s=f"Test $R^2$ = {metrics['R2']:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10
    )

    plt.grid(True)

    plot_path_combined = os.path.join(output_dir, f"{base_filename}_train_test_plot.png")
    plt.savefig(plot_path_combined, dpi=300)
    plt.close()
    print(f"saved train+test combined prediction plot to:\n{plot_path_combined}")