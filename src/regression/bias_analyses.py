# Evaluation of potential performance biases across demographic subgroups

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.constants import GIT_DIRECTORY, SCORES, RANDOM_STATE, N_BOOT, CI, METRICS
from regression.model_evaluation import compare_subgroups, metrics_summary_subgroups
from regression.plots import plot_bias_panels
from data_preparation.data_handling import make_age_binary_groups

# style
plt.rcParams["font.family"] = "Arial"
sns.set_theme(context="paper", style="white")

# config
TASK = "picnicScene"
MODEL = "full"

# run bias analyses for one score
def run_bias_for_score(
    oof_preds,
    outdir,
    target,
    task=TASK,
    model=MODEL,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
):
    os.makedirs(outdir, exist_ok=True)

    current_subset = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target)
    ].copy()

    # summaries
    g_summary = metrics_summary_subgroups(
        oof_preds, task, model, target, "Gender_label", ("f","m"),
        n_boot=n_boot, ci=ci, random_state=random_state
    )
    c_summary = metrics_summary_subgroups(
        oof_preds, task, model, target, "Country_label", ("uk","usa"),
        n_boot=n_boot, ci=ci, random_state=random_state+10
    )
    # age: <65 vs ≥65
    current_subset = current_subset.drop(columns=["AgeGroup", "AgeGroup2"], errors="ignore")
    age_df, age_levels = make_age_binary_groups(
        current_subset, age_col="Age", label_col="AgeGroup2"
    )
    a_summary = metrics_summary_subgroups(
        age_df, task, model, target, "AgeGroup2", age_levels,
        n_boot=n_boot, ci=ci, random_state=random_state+20
    )

    metrics_by_subgroup = pd.concat([
        g_summary.assign(group_col="Gender_label"),
        c_summary.assign(group_col="Country_label"),
        a_summary.assign(group_col="AgeGroup2"),
    ], ignore_index=True)

    metrics_csv = os.path.join(outdir, f"bias_metrics_by_subgroup_{target}.csv")
    metrics_by_subgroup.to_csv(metrics_csv, index=False)

    # pairwise tests per metric
    paths = {"metrics_by_subgroup_csv": metrics_csv}
    for metric in METRICS:
        tests = []
        tests.append(compare_subgroups(oof_preds, "Gender_label", ("f","m"),
                                           task, target, model, metric, n_boot, ci, random_state))
        tests.append(compare_subgroups(oof_preds, "Country_label", ("uk","usa"),
                                           task, target, model, metric, n_boot, ci, random_state))
        tests.append(compare_subgroups(age_df, "AgeGroup2", age_levels,
                                           task, target, model, metric, n_boot, ci, random_state))

        tests_all = pd.concat(tests, ignore_index=True)
        out = os.path.join(outdir, f"bias_pairwise_tests_{metric}_{target}.csv")
        tests_all.to_csv(out, index=False)
        paths[f"pairwise_tests_{metric}_csv"] = out

    # panel plots per metric
    for metric in METRICS:
        img = os.path.join(outdir, f"bias_panels_{metric}_{target}.png")
        plot_bias_panels(
            oof_preds, target, task, model, metric=metric,
            save_path=img
        )
        paths[f"panels_{metric}_png"] = img

    return paths

# run bias analyses for all language scores
def run_bias_all_scores(
    oof_path,
    out_path,
    task = TASK,
    model = MODEL
):
    os.makedirs(out_path, exist_ok=True)
    oof_all = pd.read_csv(oof_path)

    results = {}
    for target in SCORES:
        score_outdir = os.path.join(out_path, target)
        paths = run_bias_for_score(
            oof_preds=oof_all,
            outdir=score_outdir,
            target=target,
            task=task,
            model=model,
            n_boot=N_BOOT,
            ci=CI,
            random_state=RANDOM_STATE,
        )
        results[target] = paths

    return results


if __name__ == "__main__":
    oof_path = os.path.join(
        GIT_DIRECTORY,
        "results", "regression", "random_forest",
        "oof_preds_all_scores.csv"
    )
    results_dir = os.path.join(
        GIT_DIRECTORY,
        "results", "regression", "random_forest"
    )
    bias_out_path = os.path.join(results_dir, "bias")
    run_bias_all_scores(
        oof_path=oof_path,
        out_path=bias_out_path,
        task="picnicScene",
        model="full",
    )