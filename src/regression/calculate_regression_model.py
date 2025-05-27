# setup

import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from regression.regression_functions import save_regression_outputs, stratified_cross_validation

# load features and scores
task_name = "cookieTheft"
target = "PhonemicFluencyScore"

# load features and target
features = pd.read_csv(os.path.join(GIT_DIRECTORY, f"results/features/filtered/{task_name}_filtered.csv"))
scores = pd.read_csv(os.path.join(GIT_DIRECTORY, "resources/language_scores_all_subjects.csv"))

df = pd.merge(features, scores[["Subject_ID", target]], on="Subject_ID").dropna()

X = df.drop(columns=["Subject_ID", target]) # all feature columns
y = df[target]

# load fold information
folds_path = os.path.join(GIT_DIRECTORY, "data", f"{task_name}_stratified_folds.csv")
fold_df = pd.read_csv(folds_path)

# merge folds with main dataframe
df = pd.merge(df, fold_df[["Subject_ID", "fold"]], on="Subject_ID")

# use stratified cross-validation
r2_list, rmse_list, mae_list, all_preds = stratified_cross_validation(
    df=df,
    fold_column="fold",
    model_class=LinearRegression,
    model_params=None,
    target_column=target,
    feature_columns=X.columns,
    mdoel_name=model_name
)

# calculate average metrics
r2_mean = np.mean(r2_list)
r2_std = np.std(r2_list)
r2_se = r2_std / np.sqrt(len(r2_list))
r2_ci_low = r2_mean - 1.96 * r2_se
r2_ci_high = r2_mean + 1.96 * r2_se

# get model type automatically
model_type = model.__class__.__name__

# save results and plot
save_crossval_results(
    r2_list, rmse_list, mae_list,
    r2_mean, r2_std, r2_se, r2_ci_low, r2_ci_high,
    task_name, target,
    output_dir=os.path.join(GIT_DIRECTORY, "results/regression"),
    all_preds=all_preds,
    model_type="LinearRegression"
)
