import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests

from config.constants import RANDOM_STATE, N_BOOT, CI, ID_COL, SCORES, TASKS, METRICS

from data_preparation.data_handling import subset_in_order


def p_value_bootstrap(boot_stats, observed_stat):
    """
    Calculate a two-sided bootstrap p-value for an observed test statistic
    """
    bootstrapped_stats = np.asarray(boot_stats, dtype=float)
    bootstrapped_stats = bootstrapped_stats[~np.isnan(bootstrapped_stats)]

    n_bootstrap_samples = bootstrapped_stats.size
    if n_bootstrap_samples == 0:
        return np.nan

    # observed test statistic (T0 = ΔR² from original sample)
    T0 = float(observed_stat)

    # center bootstrapped estimates around 0 for null distribution (H0: ΔR² = 0)
    T_star = bootstrapped_stats - np.mean(bootstrapped_stats)  # (T* = ΔR² - mean(ΔR²))

    # two-sided p-value: add both tails (T ≥ T0|H0 and T ≤ -T0|H0)
    extreme_count = np.sum(np.abs(T_star) >= np.abs(T0))
    p_value = extreme_count / n_bootstrap_samples

    return p_value


def adjust_pvals(pvals, method="holm", alpha=0.05):
    """
    Correct p-values for multiple testing using Holm-Bonferroni method
    """
    p = np.asarray(pvals, dtype=float)
    return multipletests(p, alpha=alpha, method=method)[1]


def confidence_intervals(samples, ci=CI):
    """
    Calculate percentile bootstrap confidence intervals (default CI = 0.95)
    """
    arr = np.asarray(samples, dtype=float)
    alpha = 1.0 - ci
    return np.nanquantile(arr, alpha / 2), np.nanquantile(arr, 1 - alpha / 2)


def bootstrap_index_generator(n, n_boot, random_state):
    """
    Draw random samples for bootstrapping
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(n_boot, n))


def normalize_oof_df(all_preds, target_col=None):
    """
    Get cross-validated predictions across all subjects; safeguard: only one prediction per subject
    """
    if isinstance(all_preds, list):
        df = pd.concat(all_preds, ignore_index=True)
    else:
        df = all_preds.copy()

    required_cols = {ID_COL, "y_true", "y_pred"}
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"OOF predictions missing columns: {missing_cols}. Existing {list(df.columns)}")

    # check: one test prediction per subject expected
    subject_counts = df[ID_COL].value_counts()
    duplicated_subjects = subject_counts[subject_counts > 1]
    if len(duplicated_subjects) > 0:
        example_subjects = duplicated_subjects.index.tolist()[:5]
        raise ValueError(
            f"{len(duplicated_subjects)} subjects with multiple rows in OOF (e.g., {example_subjects})."
        )

    return df[[ID_COL, "y_true", "y_pred"]].reset_index(drop=True)


def bootstrap_r2_summary(
        oof_df,
        group_cols=("model", "task"),
        n_boot=N_BOOT,
        ci=CI,
        subject_set=None,
        random_state=RANDOM_STATE
):
    """
    Create summary of bootstrapped R² results (mean±CI) per model and task
    """
    oof_df_copy = oof_df.copy()
    if subject_set is not None:
        oof_df_copy = oof_df_copy[oof_df_copy[ID_COL].isin(subject_set)].copy()

    required = {ID_COL, "y_true", "y_pred"}
    if not required.issubset(oof_df_copy.columns):
        missing = required - set(oof_df_copy.columns)
        raise ValueError(f"OOF df missing required columns: {missing}")

    boot_rows = []
    summary_rows = []

    for group_keys, group_df in oof_df_copy.groupby(list(group_cols), observed=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        y_true = group_df["y_true"].to_numpy()
        y_pred = group_df["y_pred"].to_numpy()
        n_subjects = len(y_true)

        # bootstrapping
        r2_oof = r2_score(y_true, y_pred)
        bootstrap_indices = bootstrap_index_generator(n_subjects, n_boot, random_state)
        bootstrap_r2_values = np.array(
            [r2_score(y_true[idx], y_pred[idx]) for idx in bootstrap_indices],
            dtype=float
        )

        # add percentile CI for plotting
        ci_low, ci_high = confidence_intervals(bootstrap_r2_values, ci=ci)

        for bootstrap_idx, bootstrap_r2 in enumerate(bootstrap_r2_values):
            row = dict(zip(group_cols, group_keys))
            row.update({"bootstrap": bootstrap_idx, "r2": bootstrap_r2})
            boot_rows.append(row)

        r2_mean = float(np.nanmean(bootstrap_r2_values))

        summary_row = dict(zip(group_cols, group_keys))
        summary_row.update({
            "r2_oof": r2_oof,
            "r2_mean": r2_mean,
            "r2_ci_low": ci_low,
            "r2_ci_high": ci_high,
            "n_subjects": n_subjects,
            "n_boot": n_boot
        })
        summary_rows.append(summary_row)

    boot_df = pd.DataFrame(boot_rows)
    summ_df = pd.DataFrame(summary_rows)
    return boot_df, summ_df


def bootstrap_metrics(
    oof_df, group_cols=("model", "task"),
    n_boot=N_BOOT, ci=CI, subject_set=None, random_state=RANDOM_STATE
):
    """
    Table with all bootstrapped means + CIs for metrics: R², RMSE, MAE
    """
    oof_df_copy = oof_df.copy()
    if subject_set is not None:
        oof_df_copy = oof_df_copy[oof_df_copy[ID_COL].isin(subject_set)].copy()

    summary_rows = []

    for group_keys, group_df in oof_df_copy.groupby(list(group_cols), observed=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        y_true = group_df["y_true"].to_numpy()
        y_pred = group_df["y_pred"].to_numpy()
        n_subjects = len(y_true)

        bootstrap_indices = bootstrap_index_generator(n_subjects, n_boot, random_state)
        bootstrap_metrics_dict = {
            "r2": np.array([r2_score(y_true[idx], y_pred[idx]) for idx in bootstrap_indices]),
            "rmse": np.array(
                [np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])) for idx in bootstrap_indices]
            ),
            "mae": np.array(
                [mean_absolute_error(y_true[idx], y_pred[idx]) for idx in bootstrap_indices]
            ),
        }

        for metric_name in ("r2", "rmse", "mae"):
            draws = bootstrap_metrics_dict[metric_name]
            est = float(np.nanmean(draws))
            ci_low, ci_high = confidence_intervals(draws, ci=ci)

            summary_rows.append({
                **dict(zip(group_cols, group_keys)),
                "metric": metric_name,
                "estimate": est,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_subjects": n_subjects,
            })

    return pd.DataFrame(summary_rows)


def metric_value(y_true, y_pred, metric):
    """
    Calculate metrics: R², RMSE and MAE
    """
    m = metric.upper()
    if m == "R2":
        return float(r2_score(y_true, y_pred))
    if m == "RMSE":
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))
    if m == "MAE":
        return float(mean_absolute_error(y_true, y_pred))


def metric_label(metric):
    return "R\u00B2" if metric.upper() == 'R2' else metric


def bootstrap_metric_unpaired(df, metric, n_boot=N_BOOT, random_state=RANDOM_STATE):
    """
    Bootstrap metrics unpaired; use when subjects from different subgroups are compared (-> bias analyses)
    """

    rng = np.random.default_rng(random_state)  # generate random numbers
    idx_by_sid = df.groupby(ID_COL).indices
    subject_ids = np.array(list(idx_by_sid.keys()))  # identify subjects in the group
    n_subjects = len(subject_ids)  # -> resample same size as original group

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    # bootstrapping
    bootstrap_draws = np.empty(n_boot, dtype=float)
    for bootstrap_idx in range(n_boot):
        sampled_subject_ids = rng.choice(subject_ids, size=n_subjects, replace=True)
        sampled_indices = np.concatenate([idx_by_sid[s] for s in sampled_subject_ids])
        bootstrap_draws[bootstrap_idx] = metric_value(y_true[sampled_indices], y_pred[sampled_indices], metric)
    return bootstrap_draws


def metrics_summary_subgroups(
    oof_preds,
    task,
    model,
    target,
    group_col,
    levels,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
):
    """
    Create summary per subgroup with all metrics & CIs; variance (y_pred), mean error (y_true - y_pred)
    """

    subset_df, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    summary_rows = []
    for i, lvl in enumerate(levels):
        subgroup_df = subset_df[subset_df[group_col] == lvl]
        if subgroup_df.empty:
            summary_rows.append({
                group_col: lvl, "n_subjects": 0,
                "R2_mean": np.nan, "R2_ci_low": np.nan, "R2_ci_high": np.nan,
                "MAE_mean": np.nan, "MAE_ci_low": np.nan, "MAE_ci_high": np.nan,
                "RMSE_mean": np.nan, "RMSE_ci_low": np.nan, "RMSE_ci_high": np.nan,
                "var_pred": np.nan, "mean_error": np.nan
            })
            continue

        # bootstrap draws for each metric
        subgroup_summary = {group_col: lvl, "n_subjects": int(subgroup_df[ID_COL].nunique())}
        for j, metric in enumerate(METRICS):
            draws = bootstrap_metric_unpaired(
                subgroup_df, metric, n_boot=n_boot, random_state=random_state + 1000 * i
            )
            metric_mean = float(np.mean(draws))
            lo, hi = confidence_intervals(draws, ci=ci)
            subgroup_summary[f"{metric}_mean"] = metric_mean
            subgroup_summary[f"{metric}_ci_low"] = float(lo)
            subgroup_summary[f"{metric}_ci_high"] = float(hi)

        subgroup_summary["var_pred"] = float(
            np.nanvar(subgroup_df["y_pred"].to_numpy(), ddof=1)
        ) if len(subgroup_df) > 1 else 0.0
        subgroup_summary["mean_error"] = float(
            np.nanmean(subgroup_df["y_true"].to_numpy() - subgroup_df["y_pred"].to_numpy())
        )
        summary_rows.append(subgroup_summary)

    summary_df = pd.DataFrame(summary_rows)

    # column order
    fixed = [group_col, "n_subjects",
             "R2_mean", "R2_ci_low", "R2_ci_high",
             "MAE_mean", "MAE_ci_low", "MAE_ci_high",
             "RMSE_mean", "RMSE_ci_low", "RMSE_ci_high",
             "var_pred", "mean_error"]

    return summary_df[fixed]


def bootstrap_by_model(oof_preds, target, task, order_models, drop_models=None,
                        n_boot=N_BOOT, ci=CI, random_state=RANDOM_STATE, subject_set=None):
    """
    Create bootstrap summary for one target, task and selected models
    """
    # one target-task-combination
    subset_df = oof_preds[(oof_preds["target"] == target) & (oof_preds["task"] == task)].copy()

    # optionally drop models (baseline)
    if drop_models:
        subset_df = subset_df[~subset_df["model"].isin(drop_models)]
        order_models = [m for m in order_models if m not in set(drop_models)]

    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    subset_df["model"] = subset_df["model"].astype(cat_model)

    # run bootstrap
    boot_df, summ_df = bootstrap_r2_summary(
        subset_df, group_cols=("model",), n_boot=n_boot, ci=ci,
        random_state=random_state, subject_set=subject_set
    )

    # keep order for plotting
    boot_df["model"] = boot_df["model"].astype(cat_model)
    summ_df["model"] = summ_df["model"].astype(cat_model)

    return boot_df, summ_df, order_models


def bootstrap_r2_diff(oof_A, oof_B, n_boot=N_BOOT, random_state=RANDOM_STATE, require_same_y_true=True):
    """
    Paired bootstrap test of the difference in R² between two models, tasks or scores
    - If require_same_y_true=True: y_true identical in A and B (model/task comparison for the same target)
    - If require_same_y_true=False: y_true differs (score comparison)
    """
    # align A and B on Subject_ID to ensure that same subjects are in both sets (paired)
    merged_df = pd.merge(
        oof_A[[ID_COL, "y_true", "y_pred"]],
        oof_B[[ID_COL, "y_true", "y_pred"]],
        on=ID_COL, suffixes=("_A", "_B")
    )
    n_subjects = len(merged_df)

    # enforce same y_true only when requested (model-vs-model / task-vs-task)
    if require_same_y_true and not np.allclose(merged_df["y_true_A"], merged_df["y_true_B"], equal_nan=True):
        raise ValueError("y_true mismatch; set require_same_y_true=False for target-vs-target.")

    # extract true and predicted for both models
    y_true_A = merged_df["y_true_A"].to_numpy()
    y_true_B = merged_df["y_true_B"].to_numpy()
    y_pred_A = merged_df["y_pred_A"].to_numpy()
    y_pred_B = merged_df["y_pred_B"].to_numpy()

    # observed difference
    if require_same_y_true:
        y_true = y_true_A
        obs_diff = r2_score(y_true, y_pred_A) - r2_score(y_true, y_pred_B)
    else:
        obs_diff = r2_score(y_true_A, y_pred_A) - r2_score(y_true_B, y_pred_B)

    # for each bootstrap sample: compute R²(A) - R²(B) -> ΔR²
    bootstrap_indices = bootstrap_index_generator(n_subjects, n_boot, random_state)
    bootstrap_diffs = np.empty(n_boot, dtype=float)
    for bootstrap_idx in range(n_boot):
        sample_indices = bootstrap_indices[bootstrap_idx]
        if require_same_y_true:
            r2A = r2_score(y_true[sample_indices], y_pred_A[sample_indices])
            r2B = r2_score(y_true[sample_indices], y_pred_B[sample_indices])
        else:
            r2A = r2_score(y_true_A[sample_indices], y_pred_A[sample_indices])
            r2B = r2_score(y_true_B[sample_indices], y_pred_B[sample_indices])
        bootstrap_diffs[bootstrap_idx] = r2A - r2B

    # summary for plots
    ci_low, ci_high = confidence_intervals(bootstrap_diffs, ci=CI)
    mean_diff = bootstrap_diffs.mean()

    # two-sided bootstrap p-value
    p_boot = p_value_bootstrap(bootstrap_diffs, obs_diff)

    return bootstrap_diffs, {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_boot": p_boot,
        "n_subjects": n_subjects,
    }


def compare_scores(
    oof_preds, task="picnicScene", model="full",
    targets=SCORES,
    n_boot=N_BOOT, adjust="holm", random_state=RANDOM_STATE
):
    """
    Statistically compare scores within one task and model
    """
    # filter OOF results to the chosen task + model
    subset_df = oof_preds[(oof_preds["task"] == task) & (oof_preds["model"] == model)].copy()

    results = []
    score_pairs = list(combinations(targets, 2))  # all pairwise score comparisons
    pvals = []
    pair_summaries = []  # to align with adjusted p-values later

    # for each score pair, compute bootstrap R² difference and store raw p
    for A, B in score_pairs:
        oofA = subset_df[subset_df["target"] == A]
        oofB = subset_df[subset_df["target"] == B]
        _, summary_stats = bootstrap_r2_diff(
            oofA, oofB, n_boot=n_boot, random_state=random_state, require_same_y_true=False
        )
        pvals.append(summary_stats["p_boot"])
        pair_summaries.append((A, B, summary_stats))

    # adjust p-values across all pairs
    p_adj = adjust_pvals(pvals, method=adjust)

    # results
    for ((A, B, summary_stats), p_corr) in zip(pair_summaries, p_adj):
        results.append({
            "task": task, "model": model, "A": A, "B": B,
            **summary_stats, "p_raw": summary_stats["p_boot"], "p_adj": float(p_corr)
        })

    cols_order = [
        "task", "model",
        "A", "B",
        "mean_diff", "ci_low", "ci_high",
        "p_raw", "p_adj",
        "n_subjects",
    ]

    return pd.DataFrame(results)[cols_order].sort_values("p_adj")


def compare_tasks(
    oof_preds,
    scores=SCORES,
    tasks=TASKS,
    model="full", n_boot=N_BOOT, adjust="holm", random_state=RANDOM_STATE
):
    """
    Statistically compare tasks within scores and a selected model
    """
    # filter OOF results to the chosen model
    subset_df = oof_preds[oof_preds["model"] == model].copy()

    results = []
    pvals = []
    task_pair_summaries = []

    # for each score, test all task pairs (A vs B)
    for target in scores:
        target_df = subset_df[subset_df["target"] == target]
        for A, B in combinations(tasks, 2):
            oofA = target_df[target_df["task"] == A]
            oofB = target_df[target_df["task"] == B]
            _, summary_stats = bootstrap_r2_diff(
                oofA, oofB, n_boot=n_boot, random_state=random_state, require_same_y_true=True
            )
            pvals.append(summary_stats["p_boot"])
            task_pair_summaries.append((target, A, B, summary_stats))

    # adjust p-values across every comparison performed above
    p_adj = adjust_pvals(pvals, method=adjust)

    # results
    for ((target, A, B, summary_stats), p_corr) in zip(task_pair_summaries, p_adj):
        results.append({
            "target": target, "model": model, "A": A, "B": B,
            **summary_stats, "p_raw": summary_stats["p_boot"], "p_adj": float(p_corr)
        })

    cols_order = [
        "target", "model", "A", "B",
        "mean_diff", "ci_low", "ci_high",
        "p_raw", "p_adj", "n_subjects"
    ]
    return pd.DataFrame(results)[cols_order].sort_values(["target", "p_adj"])


def compare_tasks_for_one_score(
    oof_preds,
    target="SemanticFluencyScore",
    model="full",
    tasks=None,
    n_boot=N_BOOT,
    adjust="holm",
    random_state=RANDOM_STATE,
    subject_set=None,
):
    """
    Statistically compare tasks for one score and a selected model (for picture description tasks)
    """
    preds_copy = oof_preds.copy()

    if subject_set is not None:
        preds_copy = preds_copy[preds_copy[ID_COL].isin(subject_set)].copy()

    out = compare_tasks(
        preds_copy,
        scores=(target,),
        tasks=tuple(tasks),
        model=model,
        n_boot=n_boot,
        adjust=adjust,
        random_state=random_state,
    )
    out = out[out["target"] == target].copy()
    out = out.rename(columns={"A": "task_A", "B": "task_B"})

    return out.sort_values("p_adj").reset_index(drop=True)


def compare_models(
    oof_preds, task="picnicScene", target="SemanticFluencyScore",
    models=None, drop_models=None, n_boot=N_BOOT, adjust="holm", random_state=RANDOM_STATE
):
    """
    Statistically compare models within one score and task
    """
    # filter to the chosen task + target
    subset_df = oof_preds[(oof_preds["task"] == task) & (oof_preds["target"] == target)].copy()

    if models is None:
        models = list(pd.unique(subset_df["model"]))

    # optionally drop some models from the comparison set
    if drop_models:
        models = [m for m in models if m not in set(drop_models)]
        subset_df = subset_df[~subset_df["model"].isin(drop_models)]

    results = []
    pvals = []
    model_pair_summaries = []

    # for each model pair compute bootstrap R² difference and raw p
    for A, B in combinations(models, 2):
        oofA = subset_df[subset_df["model"] == A]
        oofB = subset_df[subset_df["model"] == B]
        _, summary_stats = bootstrap_r2_diff(oofA, oofB, n_boot=n_boot, random_state=random_state)
        pvals.append(summary_stats["p_boot"])
        model_pair_summaries.append((A, B, summary_stats))

    # adjust p-values for multiple testing
    p_adj = adjust_pvals(pvals, method=adjust)

    result_rows = []
    for ((A, B, summary_stats), p_corr) in zip(model_pair_summaries, p_adj):
        result_rows.append({
            "task": task, "target": target, "A": A, "B": B,
            **summary_stats, "p_raw": summary_stats["p_boot"], "p_adj": float(p_corr)
        })
    results_df = pd.DataFrame(result_rows)

    cols_order = [
        "task", "target", "A", "B",
        "mean_diff", "ci_low", "ci_high",
        "p_raw", "p_adj", "n_subjects"
    ]
    cols_order = [c for c in cols_order if c in results_df.columns]

    return results_df[cols_order].sort_values("p_adj")


def compare_subgroups(
    oof_preds,
    group_col,  # select variable to split on (Gender, Country, Age)
    levels,  # allowed levels and order
    task,
    target,
    model="full",
    metric="R2",
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
    pairs_to_test=None  # select what levels to test against each other (if None -> all)
):
    """
    Pairwise comparisons for demographic subgroups for one metric at a time -> Δ(A-B), CI, p-value
    """
    metric = metric.upper()
    subset_df, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    level_index = {lvl: i for i, lvl in enumerate(levels)}

    pairs = list(combinations(levels, 2)) if (pairs_to_test is None) else [tuple(p) for p in pairs_to_test]
    result_rows = []
    for (A, B) in pairs:
        subgroup_A = subset_df[subset_df[group_col] == A]
        subgroup_B = subset_df[subset_df[group_col] == B]

        if subgroup_A.empty or subgroup_B.empty:
            result_rows.append({"A": A, "B": B, "mean_diff": np.nan,
                                "ci_low": np.nan, "ci_high": np.nan,
                                "p_raw": np.nan, "p_adj": np.nan})
            continue

        # observed difference
        observed_diff = metric_value(
            subgroup_A["y_true"].to_numpy(), subgroup_A["y_pred"].to_numpy(), metric
        ) - metric_value(
            subgroup_B["y_true"].to_numpy(), subgroup_B["y_pred"].to_numpy(), metric
        )

        # use the same per-level seeds as summaries/plots
        seed_A = random_state + 1000 * level_index[A]
        seed_B = random_state + 1000 * level_index[B]

        # bootstrap draws
        mA = bootstrap_metric_unpaired(subgroup_A, metric, n_boot=n_boot, random_state=seed_A)
        mB = bootstrap_metric_unpaired(subgroup_B, metric, n_boot=n_boot, random_state=seed_B)
        diff_draws = mA - mB

        mean_d = float(np.nanmean(diff_draws))
        lo, hi = confidence_intervals(diff_draws, ci=ci)
        p = float(p_value_bootstrap(diff_draws, observed_diff))

        result_rows.append({"A": A, "B": B, "mean_diff": mean_d,
                            "ci_low": float(lo), "ci_high": float(hi),
                            "p_raw": p, "p_adj": None})

    results_df = pd.DataFrame(result_rows)
    # correct p-values for multiple testing
    results_df["p_adj"] = adjust_pvals(results_df["p_raw"].to_numpy(), method="holm")
    results_df.insert(0, "group_col", group_col)
    cols = ["group_col", "A", "B", "mean_diff", "ci_low", "ci_high", "p_raw", "p_adj"]

    return results_df[cols].sort_values(["group_col", "p_adj", "A", "B"])
