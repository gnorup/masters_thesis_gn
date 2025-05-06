# setup

import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from regression.train_regression_models import train_and_evaluate_regression_model
from regression.regression_functions import save_regression_outputs

# load features and scores
task_name = "cookieTheft"
target = "PhonemicFluencyScore"

# load features and target
features = pd.read_csv(os.path.join(GIT_DIRECTORY, f"results/features/filtered/{task_name}_filtered.csv"))
scores = pd.read_csv(os.path.join(GIT_DIRECTORY, "resources/language_scores_all_subjects.csv"))

df = pd.merge(features, scores[["Subject_ID", target]], on="Subject_ID").dropna()

X = df.drop(columns=["Subject_ID", target])
y = df[target]

# train and evaluate
model, metrics, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_and_evaluate_regression_model(
    X, y,
    model_class=LinearRegression,   # options: LinearRegression, Ridge, Lasso, RandomForestRegressor
    model_params=None,
    test_size=0.2,
    random_state=42
)

# get model type automatically
model_type = model.__class__.__name__

# save results and plot
save_regression_outputs(
    model,
    X_train, X_test, y_test, y_train,
    y_pred_train, y_pred_test,
    metrics,
    task_name, target,
    output_dir=os.path.join(GIT_DIRECTORY, "results/regression"),
    model_type=model_type
)
