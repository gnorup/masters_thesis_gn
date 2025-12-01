import os
import sys

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from regression.statistical_comparisons import run_statistical_comparisons

if __name__ == "__main__":
    oof_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest", "oof_preds_all_scores.csv")
    results_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest")
    run_statistical_comparisons(oof_path=oof_dir, results_path=results_dir)