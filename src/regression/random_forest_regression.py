import sys
import os

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from sklearn.ensemble import RandomForestRegressor

from config.constants import GIT_DIRECTORY
from regression.regression_models import run_regression
from regression.hyperparameter_tuning import load_tuned_rf_params_regression
from regression.statistical_comparisons import run_statistical_comparisons

# run Random Forest regression
def run_rf_regression():
    rf_params = load_tuned_rf_params_regression()
    oof_path = run_regression(
        model_type=RandomForestRegressor,
        model_params=rf_params,
        outdir="random_forest",
    )
    return oof_path


if __name__ == "__main__":
    # run RF + statistical comparisons when called directly
    oof_path = run_rf_regression()
    results_dir = os.path.join(
        GIT_DIRECTORY,
        "results",
        "regression",
        "random_forest",
    )
    run_statistical_comparisons(oof_path=oof_path, results_path=results_dir)
