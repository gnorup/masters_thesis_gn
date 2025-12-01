import sys
import os

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from sklearn.linear_model import LinearRegression

from config.constants import GIT_DIRECTORY
from regression.regression_models import run_regression
from regression.statistical_comparisons import run_statistical_comparisons


def run_multiple_linear_regression():
    oof_path = run_regression(
        model_type=LinearRegression,
        model_params=None,
        outdir="linear",
    )
    return oof_path


if __name__ == "__main__":
    oof_path = run_multiple_linear_regression()
    results_dir = os.path.join(
        GIT_DIRECTORY,
        "results",
        "regression",
        "linear",
    )
