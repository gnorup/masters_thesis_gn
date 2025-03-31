import sys
import os

# manually add project root to sys.path
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from regression.multiple_regression import run_multiple_regression, check_regression_assumptions

# define task name
task_name = "cookieTheft"

# run regression
run_multiple_regression(
    features_path=os.path.join(GIT_DIRECTORY, f"results/features/{task_name}.csv"),
    scores_path=os.path.join(GIT_DIRECTORY, "resources/language_scores_all_subjects.csv"),
    target="PhonemicFluencyScore",
    output_dir=os.path.join(GIT_DIRECTORY, "results/regression"),
    task_name=task_name  # passes the task to the function for labeling purposes
)
