# tune hyperparameters for Random Forest regression

import os
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV


from config.constants import GIT_DIRECTORY, ID_COL, RANDOM_STATE
from data_preparation.data_handling import (
    load_demographics,
    load_task_dataframe,
    get_model_feature_list,
    build_feature_sets
)

DATA_DIR = os.path.join(GIT_DIRECTORY, "data")
SCORES_PATH = os.path.join(DATA_DIR, "language_scores_all_subjects.csv")
HALF_SPLIT_PATH = os.path.join(DATA_DIR, "stratified_folds_half.csv")

REG_RESULTS_DIR = os.path.join(GIT_DIRECTORY, "results", "regression")
GRID_SEARCH_DIR = os.path.join(REG_RESULTS_DIR, "grid_search")
os.makedirs(GRID_SEARCH_DIR, exist_ok=True)

BEST_PARAMS_PATH = os.path.join(GRID_SEARCH_DIR, "best_params_final.csv")

DEFAULT_RF_REG_PARAMS = dict(
    n_estimators=625,
    random_state=RANDOM_STATE,
    min_samples_leaf=4,
    max_features="sqrt",
    max_depth=14,
    min_samples_split=9,
    bootstrap=True,
)

# window around parameter set from RandomizedSearchCV for GridSearchCV
def parameter_window(center, radius, low, high, step=1):
    if center is None:
        return [None]
    center = int(center)
    a = max(low, center - radius)
    b = min(high, center + radius)
    if a > b:
        return [center]
    return list(range(a, b + 1, step))

# run hyperparameter-tuning
def regression_hyperparameter_tuning(
    task_name="picnicScene",
    target="SemanticFluencyScore",
    model_name="full",
    test_half_label=1,
    n_iter_random=100,
    refine_with_grid=True,
    random_state=RANDOM_STATE,
):
    # load data
    scores = pd.read_csv(SCORES_PATH)
    demographics = load_demographics()
    df_task = load_task_dataframe(task_name, target, scores, demographics)

    # add half-split info
    half_df = pd.read_csv(HALF_SPLIT_PATH)[[ID_COL, "fold"]].rename(columns={"fold": "half"})
    for df in (df_task, half_df):
        df[ID_COL] = df[ID_COL].astype(str)
    df = df_task.merge(half_df, on=ID_COL, how="inner")

    # features in different models
    model_features = build_feature_sets(df.columns)
    feature_cols = model_features[model_name]

    df_use = df.dropna(subset=[target] + feature_cols).reset_index(drop=True)

    # outer half-split
    train_df = df_use[df_use["half"] != test_half_label].reset_index(drop=True)
    test_df = df_use[df_use["half"] == test_half_label].reset_index(drop=True)

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[target].astype(float)

    # drop any remaining NaNs
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]

    print(f"{task_name} | {model_name} -> train N={len(X_train)}, features={len(feature_cols)}")

    # inner 5-fold CV
    inner_cv = list(
        KFold(n_splits=5, shuffle=True, random_state=random_state)
        .split(np.arange(len(X_train)))
    )

    # RandomizedSearchCV
    param_dist = {
        "n_estimators": randint(100, 1001),
        "max_depth": [None] + list(range(10, 31)),
        "max_features": ["sqrt", "log2"],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "bootstrap": [True, False],
    }

    rs = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter_random,
        scoring="r2",
        cv=inner_cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        refit=True,
        return_train_score=True,
    )

    print("\n[RandomizedSearchCV] running ...")
    rs.fit(X_train, y_train)

    pd.DataFrame(rs.cv_results_).to_csv(
        os.path.join(GRID_SEARCH_DIR, "randomized_search_cv_results.csv"),
        index=False,
    )

    best_params = rs.best_params_
    print("RandomizedSearch best params:", best_params)
    print("RandomizedSearch best inner-CV R²:", f"{rs.best_score_:.3f}")

    final_params = dict(best_params)

    # GridSearchCV refinement
    if refine_with_grid:
        bp = best_params
        grid = {
            "n_estimators": parameter_window(bp["n_estimators"], radius=200, low=100, high=1000, step=50),
            "max_depth": [None] if bp["max_depth"] is None else parameter_window(bp["max_depth"], radius=3, low=10, high=30),
            "max_features": ["sqrt", "log2"],
            "min_samples_split": parameter_window(bp["min_samples_split"], radius=1, low=2, high=10),
            "min_samples_leaf": parameter_window(bp["min_samples_leaf"], radius=1, low=1, high=5),
            "bootstrap": [True],
        }

        gs = GridSearchCV(
            estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
            param_grid=grid,
            scoring="r2",
            cv=inner_cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
            return_train_score=False,
        )

        print("\n[GridSearchCV] running ...")
        gs.fit(X_train, y_train)

        pd.DataFrame(gs.cv_results_).to_csv(
            os.path.join(GRID_SEARCH_DIR, "grid_search_cv_results.csv"),
            index=False,
        )

        best_params = gs.best_params_
        print("GridSearch best params:", best_params)
        print("GridSearch best inner-CV R²:", f"{gs.best_score_:.3f}")
        final_params = dict(best_params)

    # save resulting parameters
    pd.DataFrame([final_params]).to_csv(BEST_PARAMS_PATH, index=False)
    print(f"[tuned params] saved to {BEST_PARAMS_PATH}")

    return final_params

# load results from hyperparameter tuning
def load_tuned_rf_params_regression(default_params=None):
    if default_params is None:
        default_params = dict(DEFAULT_RF_REG_PARAMS)
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            df = pd.read_csv(BEST_PARAMS_PATH)
            params = df.iloc[0].to_dict()
            params["random_state"] = RANDOM_STATE
            params["n_jobs"] = -1
            return params
        except Exception as e:
            print("failed to read tuned regression params, using defaults. Error:", e)
    return dict(default_params)