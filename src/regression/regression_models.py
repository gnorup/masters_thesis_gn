import os
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor

from config.constants import GIT_DIRECTORY, TASKS, SCORES, ID_COL, N_BOOT, CI, RANDOM_STATE

from data_preparation.data_handling import (
    load_demographics,
    load_task_dataframe,
    complete_subjects,
    build_feature_sets,
)

from regression.regression_setup import stratified_cross_validation
from regression.model_evaluation import (
    normalize_oof_df,
    bootstrap_r2_summary,
    bootstrap_metrics,
)


# run main regression
def run_regression(model_type, model_params, outdir):
    """
    Run regression for all scores, tasks and models using stratified 5-fold cross-validation.
    Output: test-fold predictions for each subject, fold-wise metrics, bootstrap summaries and metrics
    """
    # path to predictions and summaries
    oof_dir = os.path.join(GIT_DIRECTORY, "results", "regression", outdir)
    os.makedirs(oof_dir, exist_ok=True)

    # language scores for all subjects
    scores_df = pd.read_csv(
        os.path.join(GIT_DIRECTORY, "data", "language_scores_all_subjects.csv")
    )

    # load demographic information
    demographics = load_demographics(demographics_csv=None)

    all_oof_rows = []
    fold_rows = []

    # loop over scores
    for target in SCORES:
        # per-task dataframes for this score
        task_dfs = {
            t: load_task_dataframe(t, target, scores_df, demographics)
            for t in TASKS
        }

        # feature-complete intersection of subjects (based on full model)
        full_cols_by_task = {
            t: build_feature_sets(task_dfs[t].columns)["full"]
            for t in TASKS
        }
        subject_sets = [
            complete_subjects(task_dfs[t], full_cols_by_task[t], target)
            for t in TASKS
        ]
        full_subjects = set.intersection(*subject_sets)

        print(
            f"{target}: intersection of subjects across all tasks: "
            f"N={len(full_subjects)}"
        )

        # fit all models for this score & each task
        for t in TASKS:
            df_t = task_dfs[t]
            df_t = df_t[df_t[ID_COL].isin(full_subjects)].copy()

            model_features = build_feature_sets(df_t.columns)
            model_features["baseline"] = []

            for model_name, fcols in model_features.items():
                # baseline via DummyRegressor (mean prediction)
                if model_name == "baseline":
                    df_use = df_t.dropna(subset=[target]).copy()
                    if df_use.empty:
                        continue

                    X = pd.DataFrame(
                        np.ones((len(df_use), 1)),
                        index=df_use.index,
                        columns=["__dummy__"],
                    )
                    fcols = ["__dummy__"]
                    mtype, mparams = DummyRegressor, {"strategy": "mean"}

                else:
                    # other models using features
                    if not fcols:
                        continue
                    df_use = df_t.dropna(subset=[target] + fcols).copy()

                    if df_use.empty:
                        continue

                    X = df_use[fcols]
                    mtype, mparams = model_type, model_params

                # dataframe passed into stratified_cross_validation
                model_df = pd.concat(
                    [
                        df_use[[ID_COL, "fold"]],
                        X,
                        df_use[target].rename(target),
                    ],
                    axis=1,
                )

                print(
                    f"{t} | {model_name} | N={len(model_df)} | "
                    f"features used={0 if model_name == 'baseline' else len(fcols)}"
                )

                # cross-validation (pre-stratified folds)
                r2_list, rmse_list, mae_list, all_preds = stratified_cross_validation(
                    df=model_df,
                    fold_column="fold",
                    model_type=mtype,
                    model_params=mparams,
                    target_column=target,
                    n_folds=5,
                    feature_columns=fcols,
                )

                # collect fold metrics
                for k, (r2, rmse, mae) in enumerate(
                    zip(r2_list, rmse_list, mae_list)
                ):
                    fold_rows.append(
                        {
                            "target": target,
                            "task": t,
                            "model": model_name,
                            "fold": k,
                            "r2": r2,
                            "rmse": rmse,
                            "mae": mae,
                            "estimator": mtype.__name__,
                        }
                    )

                # normalized out-of-fold (OOF) predictions per subject
                all_preds = all_preds.rename(columns={"y_test": "y_true"})
                oof_df = normalize_oof_df(all_preds, target_col=target)
                oof_df["task"] = t
                oof_df["model"] = model_name
                oof_df["target"] = target

                # merge demographics into OOF rows for later subsetting
                cols_keep = [
                    ID_COL,
                    "Age",
                    "Gender",
                    "Education_level",
                    "Country",
                    "Socioeconomic",
                ]
                oof_df[ID_COL] = oof_df[ID_COL].astype(str)
                demo_merge = demographics[cols_keep].copy()
                demo_merge[ID_COL] = demo_merge[ID_COL].astype(str)

                oof_df = oof_df.merge(
                    demo_merge,
                    on=ID_COL,
                    how="left",
                )

                # labels for group-based analyses
                oof_df["Gender_label"] = oof_df["Gender"].map({0: "f", 1: "m"})
                oof_df["Country_label"] = oof_df["Country"].map({0: "uk", 1: "usa"})
                oof_df["AgeGroup"] = pd.cut(
                    oof_df["Age"],
                    bins=[-np.inf, 65, 75, np.inf],
                    labels=["<65", "65–75", ">75"],
                )

                all_oof_rows.append(oof_df)

        # per-score bootstrap summaries
        oof_all = pd.concat(all_oof_rows, ignore_index=True)
        oof_score = oof_all[oof_all["target"] == target].copy()

        # R² summaries
        _, summ_df = bootstrap_r2_summary(
            oof_score,
            group_cols=("target", "task", "model"),
            n_boot=N_BOOT,
            ci=CI,
            random_state=RANDOM_STATE,
        )
        summ_df.to_csv(
            os.path.join(oof_dir, f"bootstrap_summary_{target}.csv"),
            index=False,
        )

        # R² / RMSE / MAE with CIs
        met = bootstrap_metrics(
            oof_score,
            group_cols=("target", "task", "model"),
            n_boot=N_BOOT,
            ci=CI,
            random_state=RANDOM_STATE,
        )
        met.to_csv(
            os.path.join(oof_dir, f"bootstrap_metrics_{target}.csv"),
            index=False,
        )

    # store all OOF predictions + fold-wise metrics
    oof_all = pd.concat(all_oof_rows, ignore_index=True)
    oof_path = os.path.join(oof_dir, "oof_preds_all_scores.csv")
    oof_all.to_csv(oof_path, index=False)

    pd.DataFrame(fold_rows).to_csv(
        os.path.join(oof_dir, "cv_folds_all_scores.csv"),
        index=False,
    )

    return oof_path
