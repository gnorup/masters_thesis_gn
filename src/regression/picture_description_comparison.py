# compare different picture description sample lengths (1min, 2min, ≤5min)

import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY, SCORES, ID_COL, N_BOOT, RANDOM_STATE, PD_TASKS
from config.feature_sets import get_linguistic_features, get_acoustic_features, get_demographic_features

from data_preparation.data_handling import (
    load_demographics,
    get_model_feature_list,
    complete_subjects,
    subject_intersection_for_score,
    load_task_dataframe,
)
from regression.regression_setup import stratified_cross_validation
from regression.plots import plot_pd_duration_box_with_pvals
from regression.model_evaluation import (
    normalize_oof_df,
    bootstrap_r2_summary,
    bootstrap_metrics,
    compare_tasks_for_one_score,
)
from regression.hyperparameter_tuning import load_tuned_rf_params_regression


# config
MODEL_TYPE = RandomForestRegressor
MODEL_PARAMS = load_tuned_rf_params_regression()

DISPLAY_LABELS = {
    "picture_description_1min": "Picture Description (1min)",
    "picture_description_2min": "Picture Description (2min)",
    "picture_description": "Picture Description (≤5min)",
}


def run_picture_description_variants_comparison():
    """
    Compare picture description variants (1min, 2min, ≤5min) using Random Forest regression.
    """
    results_dir = os.path.join(
        GIT_DIRECTORY,
        "results",
        "regression",
        "random_forest",
        "picture_description",
    )
    os.makedirs(results_dir, exist_ok=True)

    # data
    scores_df = pd.read_csv(os.path.join(GIT_DIRECTORY, "data", "language_scores_all_subjects.csv"))
    scores_df[ID_COL] = scores_df[ID_COL].astype(str)

    demographics = load_demographics()
    demographics[ID_COL] = demographics[ID_COL].astype(str)

    # model configs: full model
    linguistic = get_linguistic_features()
    acoustic = get_acoustic_features()
    demographic = get_demographic_features()
    model_configs = {"full": sorted(list(linguistic | acoustic)) + demographic}

    # cross-validation + test fold predictions
    all_oof_rows = []
    fold_rows = []

    for target in SCORES:
        task_dfs = {t: load_task_dataframe(t, target, scores_df, demographics) for t in PD_TASKS}
        for t in PD_TASKS:
            task_dfs[t][ID_COL] = task_dfs[t][ID_COL].astype(str)

        full_cols_by_task = {
            t: get_model_feature_list(task_dfs[t].columns, model_configs["full"], target)
            for t in PD_TASKS
        }

        subject_sets = [
            set(map(str, complete_subjects(task_dfs[t], full_cols_by_task[t], target)))
            for t in PD_TASKS
        ]
        full_subjects = set.intersection(*subject_sets)
        print(f"\n{target}: intersection of subjects across all tasks: N={len(full_subjects)}")

        if len(full_subjects) == 0:
            for t in PD_TASKS:
                n_rows = len(task_dfs[t])
                n_feats = len(full_cols_by_task[t])
                n_complete = len(
                    set(map(str, complete_subjects(task_dfs[t], full_cols_by_task[t], target)))
                )
                print(
                    f"  - {t}: rows={n_rows}, feat_cols={n_feats}, complete_subjects={n_complete}"
                )
            raise RuntimeError(f"[STOP] No subject intersection for target={target}.")

        for t in PD_TASKS:
            df_t = task_dfs[t].copy()
            before = len(df_t)
            df_t = df_t[df_t[ID_COL].isin(full_subjects)].copy()
            print(f"{target} | {t}: kept {len(df_t)}/{before} rows after subject intersection.")

            model_name, selected = "full", model_configs["full"]
            fcols = get_model_feature_list(df_t.columns, selected, target)
            if not fcols:
                print(f"{target} | {t} | {model_name}: 0 feature columns matched.")
                continue

            miss_tgt = df_t[target].isna().sum()
            miss_feat = df_t[fcols].isna().any(axis=1).sum()
            df_use = df_t.dropna(subset=[target] + fcols).copy()
            print(
                f"{target} | {t} | {model_name}: fcols={len(fcols)}, "
                f"NaN target rows={miss_tgt}, any NaN in features rows={miss_feat}, "
                f"usable={len(df_use)}"
            )

            if df_use.empty:
                continue

            X = df_use[fcols]
            mtype, mparams = MODEL_TYPE, MODEL_PARAMS

            model_df = pd.concat(
                [df_use[[ID_COL, "fold"]], X, df_use[target].rename(target)], axis=1
            )
            print(f"{t} | {model_name} | N={len(model_df)} | features={len(fcols)}")

            r2_list, rmse_list, mae_list, all_preds = stratified_cross_validation(
                df=model_df,
                fold_column="fold",
                model_type=mtype,
                model_params=mparams,
                target_column=target,
                feature_columns=fcols,
            )

            if all_preds is None or all_preds.empty:
                print(f"empty predictions for {t} / {target} — skipping.")
                continue

            for k, (r2, rmse, mae) in enumerate(zip(r2_list, rmse_list, mae_list)):
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

            all_preds = all_preds.rename(columns={"y_test": "y_true"})
            oof_df = normalize_oof_df(all_preds, target_col=target)
            oof_df["task"] = t
            oof_df["model"] = model_name
            oof_df["target"] = target

            cols_keep = [ID_COL, "Age", "Gender", "Education_level", "Country"]
            oof_df = oof_df.merge(demographics[cols_keep], on=ID_COL, how="left")
            all_oof_rows.append(oof_df)

    if not all_oof_rows:
        raise RuntimeError("No OOF rows were produced.")

    oof_all = pd.concat(all_oof_rows, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    oof_all.to_csv(os.path.join(results_dir, "oof_preds_all_scores.csv"), index=False)
    fold_df.to_csv(os.path.join(results_dir, "cv_folds_all_scores.csv"), index=False)
    print("\nSaved OOF and fold metrics.")

    # bootstrap summaries per target
    for target in SCORES:
        oof_score = oof_all[oof_all["target"] == target].copy()
        _, summ_df = bootstrap_r2_summary(
            oof_score,
            group_cols=("target", "task", "model"),
            n_boot=N_BOOT,
            ci=0.95,
            random_state=RANDOM_STATE,
        )
        summ_df.to_csv(
            os.path.join(results_dir, f"bootstrap_summary_{target}.csv"),
            index=False,
        )

        met = bootstrap_metrics(
            oof_score,
            group_cols=("target", "task", "model"),
            n_boot=N_BOOT,
            ci=0.95,
            random_state=RANDOM_STATE,
        )
        met.to_csv(
            os.path.join(results_dir, f"bootstrap_metrics_{target}.csv"),
            index=False,
        )

    # duration comparisons
    output_directory = os.path.join(results_dir, "picture_description_analyses")
    os.makedirs(output_directory, exist_ok=True)

    for target in SCORES:
        subject_intersection = subject_intersection_for_score(
            oof_all=oof_all,
            target=target,
            tasks=PD_TASKS,
            models=["full"],
        )
        tests = compare_tasks_for_one_score(
            oof_all,
            target=target,
            model="full",
            tasks=tuple(PD_TASKS),
            n_boot=N_BOOT,
            adjust="holm",
            random_state=RANDOM_STATE,
            subject_set=subject_intersection,
        )

        tests.to_csv(
            os.path.join(output_directory, f"{target}_picture_description_sample_length_comparison.csv"),
            index=False,
        )

        significant_pairs = [(r.task_A, r.task_B) for _, r in tests.iterrows() if r.p_adj < 0.05]
        _ = plot_pd_duration_box_with_pvals(
            oof_all,
            target=target,
            model="full",
            order_tasks=PD_TASKS,
            pairs_to_show=significant_pairs,
            subject_set=subject_intersection,
            n_boot=N_BOOT,
            random_state=RANDOM_STATE,
            save_path=output_directory,
            display_labels=DISPLAY_LABELS,
        )


if __name__ == "__main__":
    run_picture_description_variants_comparison()
