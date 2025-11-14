import sys
import os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from config.feature_sets import get_linguistic_features, get_acoustic_features

# default values
default_ci = 0.95
default_nboot = 1000
default_randomstate = 42

# calculate two-sided p-value from a bootstrapped difference-distribution
def p_value_bootstrap(boot_stats, observed_stat): # bootstrapped difference & observed difference
    bootstrapped_stats = np.asarray(boot_stats, dtype=float)
    bootstrapped_stats = bootstrapped_stats[~np.isnan(bootstrapped_stats)]
    B = bootstrapped_stats.size # number of bootstrap samples
    centered = bootstrapped_stats - observed_stat # center to null ~ sampling distribution under H0 (true difference = 0)
    count = np.sum(np.abs(centered) >= np.abs(observed_stat)) # count how many bootstrap values are at least as extreme as the observed one
    return count / B

# calculate 95%-confidence intervals
def confidence_intervals(samples, ci=default_ci):
    arr = np.asarray(samples, dtype=float)
    alpha = 1.0 - ci
    return np.nanquantile(arr, alpha / 2), np.nanquantile(arr, 1 - alpha / 2)

# draw random samples for bootstrapping
def bootstrap_index_generator(n, n_boot, random_state):
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(n_boot, n))

### model comparison helpers

# load demographics and turn into categories for models
def load_demographics(demographics_csv):

    if demographics_csv is None:
        demographics_csv = os.path.join(GIT_DIRECTORY, "data", "demographics_data.csv")

    df = pd.read_csv(demographics_csv)

    # normalize selected string columns
    for col in ["Gender","Education","Country","Language"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    # make sure Age is numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # make sure Socioeconomic is numeric (MacArthur scale)
    if "Socioeconomic" in df.columns:
        df["Socioeconomic"] = pd.to_numeric(df["Socioeconomic"], errors="coerce")

    # map Gender -> female (f):0, male (m):1
    if "Gender" in df.columns:
        gender_map = {"f": 0, "female": 0, "m": 1, "male": 1}
        df["Gender"] = df["Gender"].map(gender_map)

    # map Education to ordered levels and later to low, middle high
    if "Education" in df.columns:
        education_map = {
            "less_than_highschool": 1, "less than highschool": 1, "less than high school": 1,
            "high_school": 2, "high school": 2,
            "vocational": 3,
            "bachelor": 4,
            "master": 5, "msc": 5, "ma": 5,
            "phd": 6, "doctorate": 6,
            "no_answer": np.nan, "na": np.nan, "": np.nan
        }
        df["Education"] = df["Education"].map(education_map)
        # 0=low(1), 1=mid(2–3), 2=high(4–6)
        df["Education_level"] = df["Education"].map({1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})

    # map Country -> uk:0, usa:1
    if "Country" in df.columns:
        country_map = {
            "uk": 0,
            "usa": 1, "us": 1,
        }
        df["Country"] = df["Country"].map(country_map)

    # drop Language if present (should carry same information as country)
    df = df.drop(columns=["Language"], errors="ignore")

    for col in ["Gender","Education","Education_level","Country"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    keep = ["Subject_ID","Age","Gender","Education","Education_level","Socioeconomic","Country"]
    cols = ["Subject_ID"] + [c for c in keep if c in df.columns and c != "Subject_ID"]
    return df[cols].copy()

# load and merge features and stratified folds
def load_task_dataframe(task_name, target, scores, demographics):
    features_path = os.path.join(GIT_DIRECTORY, f"results/features/filtered/{task_name}_filtered2.csv")
    folds_path = os.path.join(GIT_DIRECTORY, "data/stratified_folds.csv")
    features = pd.read_csv(features_path)
    folds = pd.read_csv(folds_path)
    df = pd.merge(features, scores[["Subject_ID", target]], on="Subject_ID")
    df = pd.merge(df, demographics, on="Subject_ID")
    df = pd.merge(df, folds[["Subject_ID", "fold"]], on="Subject_ID")
    return df

# get features for certain model
def get_model_feature_list(df_columns, selected_features, target_name):
    drop = {"Subject_ID", "fold", target_name}
    safe = []
    for f in selected_features:
        if f in drop:
            continue
        if f in df_columns:
            safe.append(f)
    return safe

# define models (what features belong into what model)
def build_feature_sets(df_columns, demographic_variables):
    linguistic = get_linguistic_features()
    acoustic   = get_acoustic_features()
    configs = {
        "demographics": demographic_variables,
        "acoustic": sorted(list(acoustic)),
        "linguistic": sorted(list(linguistic)),
        "acoustic+linguistic": sorted(list(acoustic | linguistic)),
        "full": sorted(list(acoustic | linguistic)) + demographic_variables
    }
    def get_model_feature_list(df_columns, selected_features):
        drop = {"Subject_ID","fold"}
        df_cols = set(df_columns)
        return [f for f in selected_features if f not in drop and f in df_cols]

    for k in list(configs.keys()):
        configs[k] = get_model_feature_list(df_columns, configs[k])
    return configs


# get Subject_IDs with no missing in target & all features
def complete_subjects(df, feature_cols, target_name):
    need = [target_name] + (feature_cols if len(feature_cols) > 0 else [])
    return set(df.dropna(subset=need)["Subject_ID"])

# get Subject_IDs with no missing in target, all features and scores
def subjects_with_all_features_and_scores(full_subjects, scores_df, score_cols):
    if "Subject_ID" not in scores_df.columns:
        raise ValueError("scores_df must contain 'Subject_ID'")
    all_scores = set(scores_df.dropna(subset=list(score_cols))["Subject_ID"])
    return set(full_subjects) & all_scores

# get cross-validated test-R² across all subjects -> cleans and validates
def normalize_oof_df(all_preds, target_col=None):
    if isinstance(all_preds, list):
        df = pd.concat(all_preds, ignore_index=True)
    else:
        df = all_preds.copy()

    # fix error
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"subject_id","subject","participant","participant_id"}:
            rename_map[c] = "Subject_ID"
        elif cl in {"y_true","y","target","true","y_test","y_actual"}:
            rename_map[c] = "y_true"
        elif cl in {"y_pred","pred","prediction","yhat","y_hat"}:
            rename_map[c] = "y_pred"
        elif cl == "fold":
            rename_map[c] = "fold"
    df = df.rename(columns=rename_map)

    if "y_true" not in df.columns and target_col is not None and target_col in df.columns:
        df = df.rename(columns={target_col: "y_true"})

    need = {"Subject_ID","y_true","y_pred"}
    if not need.issubset(df.columns):
        miss = need - set(df.columns)
        raise ValueError(f"OOF predictions missing columns: {miss}. Have {list(df.columns)}")

    # check: one test prediction per subject expected
    counts = df["Subject_ID"].value_counts()
    bad = counts[counts > 1]
    if len(bad) > 0:
        sample = bad.index.tolist()[:5]
        raise ValueError(
            f"{len(bad)} subjects with multiple rows in OOF (e.g., {sample})."
        )

    return df[["Subject_ID","y_true","y_pred"]].reset_index(drop=True)

# creates summary of bootstrapped results per model and task
def bootstrap_summary_from_oof(
        oof_df: pd.DataFrame,
        group_cols=("model", "task"),
        n_boot=default_nboot,
        ci=default_ci,
        subject_set=None,
        random_state=default_randomstate
):
    df = oof_df.copy()
    if subject_set is not None:
        df = df[df["Subject_ID"].isin(subject_set)].copy()

    required = {"Subject_ID", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OOF df missing required columns: {missing}")

    boot_rows = []
    summ_rows = []

    for keys, g in df.groupby(list(group_cols), observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y = g["y_true"].to_numpy()
        ypred = g["y_pred"].to_numpy()
        n = len(y)

        # bootstrapping
        r2_oof = r2_score(y, ypred)
        idxs = bootstrap_index_generator(n, n_boot, random_state)
        boots = np.array([r2_score(y[idx], ypred[idx]) for idx in idxs], dtype=float)

        # add percentile CI for plotting
        ci_low, ci_high = confidence_intervals(boots, ci=ci)

        for b_idx, r2b in enumerate(boots):
            row = dict(zip(group_cols, keys))
            row.update({"bootstrap": b_idx, "r2": r2b})
            boot_rows.append(row)

        r2_mean = float(np.nanmean(boots))

        srow = dict(zip(group_cols, keys))
        srow.update({
            "r2_oof": r2_oof,
            "r2_mean": r2_mean,
            "r2_ci_low": ci_low,
            "r2_ci_high": ci_high,
            "n_subjects": n,
            "n_boot": n_boot
        })
        summ_rows.append(srow)

    boot_df = pd.DataFrame(boot_rows)
    summ_df = pd.DataFrame(summ_rows)
    return boot_df, summ_df

# store more bootstrapped metrics like rmse and mae for tables
def bootstrap_metrics_from_oof(
    oof_df: pd.DataFrame,
    group_cols=("model","task"),
    n_boot=default_nboot, ci=default_ci, subject_set=None, random_state=default_randomstate
):

    df = oof_df.copy()
    if subject_set is not None:
        df = df[df["Subject_ID"].isin(subject_set)].copy()

    rows = []

    for keys, g in df.groupby(list(group_cols), observed=True):
        if not isinstance(keys, tuple): keys = (keys,)
        y = g["y_true"].to_numpy(); ypred = g["y_pred"].to_numpy()
        n = len(y)

        est = {
            "r2": r2_score(y, ypred),
            "rmse": np.sqrt(mean_squared_error(y, ypred)),
            "mae": mean_absolute_error(y, ypred),
        }
        idxs = bootstrap_index_generator(n, n_boot, random_state)
        boots = {
            "r2": np.array([r2_score(y[idx], ypred[idx]) for idx in idxs]),
            "rmse": np.array([np.sqrt(mean_squared_error(y[idx], ypred[idx])) for idx in idxs]),
            "mae": np.array([mean_absolute_error(y[idx], ypred[idx]) for idx in idxs]),
        }

        for m in ("r2","rmse","mae"):
            rows.append({
                **dict(zip(group_cols, keys)),
                "metric": m,
                "estimate": est[m],
                "ci_low": confidence_intervals(boots[m], ci=ci)[0],
                "ci_high": confidence_intervals(boots[m], ci=ci)[1],
                "n_subjects": n,
                "n_boot": n_boot,
            })

    return pd.DataFrame(rows)

# paired bootstrap test of the difference in R² between two models
def bootstrap_r2_diff(oof_A, oof_B, n_boot=default_nboot, random_state=default_randomstate, require_same_y_true=True):
    # align A and B on Subject_ID to ensure that same subjects are in both sets (paired)
    m = pd.merge(oof_A[["Subject_ID","y_true","y_pred"]],
                 oof_B[["Subject_ID","y_true","y_pred"]],
                 on="Subject_ID", suffixes=("_A","_B"))
    n = len(m)

    # enforce same y_true only when requested (model-vs-model / task-vs-task)
    if require_same_y_true and not np.allclose(m["y_true_A"], m["y_true_B"], equal_nan=True):
        raise ValueError("y_true mismatch; set require_same_y_true=False for target-vs-target.")

    # extract true and predicted for both models
    y_true_A = m["y_true_A"].to_numpy()
    y_true_B = m["y_true_B"].to_numpy()
    y_pred_A = m["y_pred_A"].to_numpy()
    y_pred_B = m["y_pred_B"].to_numpy()

    # observed difference
    if require_same_y_true:
        y_true = y_true_A
        obs_diff = r2_score(y_true, y_pred_A) - r2_score(y_true, y_pred_B)
    else:
        obs_diff = r2_score(y_true_A, y_pred_A) - r2_score(y_true_B, y_pred_B)

    # for each bootstrap sample: compute R²(A) - R²(B) -> ΔR²
    idxs = bootstrap_index_generator(n, n_boot, random_state)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = idxs[b]
        if require_same_y_true:
            r2A = r2_score(y_true[idx], y_pred_A[idx])
            r2B = r2_score(y_true[idx], y_pred_B[idx])
        else:
            r2A = r2_score(y_true_A[idx], y_pred_A[idx])
            r2B = r2_score(y_true_B[idx], y_pred_B[idx])
        diffs[b] = r2A - r2B

    # summary for plots
    ci_low, ci_high = confidence_intervals(diffs, ci=default_ci)
    mean_d  = diffs.mean()

    # two-sided bootstrap p-value
    p_boot = p_value_bootstrap(diffs, obs_diff)

    return diffs, {
        "mean_diff": mean_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_boot": p_boot,
        "n_subjects": n,
        "n_boot": n_boot
    }

# compare models for specific task and target using bootstrapping
def compare_models_bootstrap(oof_preds, task, target, models=None, n_boot=default_nboot, random_state=default_randomstate, preferred_order=None):
    df = oof_preds[(oof_preds["task"] == task) & (oof_preds["target"] == target)].copy()
    if models is None:
        if preferred_order:
            models = sorted(df["model"].unique(), key=lambda m: preferred_order.index(m) if m in preferred_order else 999)
        else:
            models = list(pd.unique(df["model"]))
    rows = []
    for a, b in combinations(models, 2):
        oofA = df[df["model"] == a]
        oofB = df[df["model"] == b]
        diffs, summ = bootstrap_r2_diff(oofA, oofB, n_boot=n_boot, random_state=random_state)
        rows.append({"task": task, "target": target, "model_a": a, "model_b": b, **summ})
    return pd.DataFrame(rows)

# bootstrapping for one target and task
def bootstrap_by_model(oof_preds, target, task, order_models, drop_models=None,
                        n_boot=default_nboot, ci=default_ci, random_state=default_randomstate, subject_set=None):
    # one target-task-combination
    sub = oof_preds[(oof_preds["target"]==target) & (oof_preds["task"]==task)].copy()

    # optionally drop models (baseline)
    if drop_models:
        sub = sub[~sub["model"].isin(drop_models)]
        order_models = [m for m in order_models if m not in set(drop_models)]

    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    sub["model"] = sub["model"].astype(cat_model)

    # run bootstrap
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("model",), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )

    # keep order for plotting
    boot_df["model"] = boot_df["model"].astype(cat_model)
    summ_df["model"] = summ_df["model"].astype(cat_model)

    return boot_df, summ_df, order_models

# p-value correction: Holm-Bonferroni
def adjust_pvals(pvals, method="holm", alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    return multipletests(p, alpha=alpha, method=method)[1]

#  score comparisons within one task (full model)
def pairwise_score_tests_for_task(
    oof_preds, task="picnicScene", model="full",
    targets=("PictureNamingScore", "SemanticFluencyScore", "PhonemicFluencyScore"),
    n_boot=default_nboot, adjust="holm", random_state=default_randomstate
):
    # filter OOF results to the chosen task + model
    df = oof_preds[(oof_preds["task"] == task) & (oof_preds["model"] == model)].copy()

    results = []
    pairs = list(combinations(targets, 2))  # all pairwise score comparisons
    pvals = []
    tmp = []  # to align with adjusted p-values later

    # for each score pair, compute bootstrap R² difference and store raw p
    for A, B in pairs:
        oofA = df[df["target"] == A]
        oofB = df[df["target"] == B]
        _, summ = bootstrap_r2_diff(oofA, oofB, n_boot=n_boot, random_state=random_state, require_same_y_true=False)
        pvals.append(summ["p_boot"])
        tmp.append((A, B, summ))

    # adjust p-values across all pairs
    p_adj = adjust_pvals(pvals, method=adjust)

    # result dataframe
    for ((A, B, summ), p_corr) in zip(tmp, p_adj):
        results.append({
            "task": task, "model": model, "A": A, "B": B,
            **summ, "p_raw": summ["p_boot"], "p_adj": float(p_corr), "adjust": adjust
        })

    # sort by adjusted p-value
    return pd.DataFrame(results).sort_values("p_adj")

# task comparisons for each of two scores (full model)
def pairwise_task_tests_for_scores(
    oof_preds,
    scores=("PictureNamingScore", "SemanticFluencyScore"),
    tasks=("cookieTheft", "picnicScene", "journaling"),
    model="full", n_boot=default_nboot, adjust="holm", random_state=default_randomstate
):
    # full model
    df = oof_preds[oof_preds["model"] == model].copy()

    results, pvals, tmp = [], [], []

    # for each score, test all task pairs (A vs B)
    for target in scores:
        sub = df[df["target"] == target]
        for A, B in combinations(tasks, 2):
            oofA = sub[sub["task"] == A]
            oofB = sub[sub["task"] == B]
            _, summ = bootstrap_r2_diff(oofA, oofB, n_boot=n_boot, random_state=random_state, require_same_y_true=True)
            pvals.append(summ["p_boot"])
            tmp.append((target, A, B, summ))

    # adjust p-values across every comparison performed above
    p_adj = adjust_pvals(pvals, method=adjust)

    # results table (one row per score × task-pair)
    for ((target, A, B, summ), p_corr) in zip(tmp, p_adj):
        results.append({
            "target": target, "model": model, "A": A, "B": B,
            **summ, "p_raw": summ["p_boot"], "p_adj": float(p_corr), "adjust": adjust
        })

    cols_order = [
        "target", "model", "A", "B",
        "mean_diff", "ci_low", "ci_high",
        "p_raw", "p_adj", "adjust", "n_subjects", "n_boot"
    ]
    return pd.DataFrame(results)[cols_order].sort_values(["target", "p_adj"])

# model comparisons within one task+score
def pairwise_model_tests_for_task_target(
    oof_preds, task="picnicScene", target="SemanticFluencyScore",
    models=None, drop_models=None, n_boot=default_nboot, adjust="holm", random_state=default_randomstate
):
    # filter to the chosen task + target
    sub = oof_preds[(oof_preds["task"] == task) & (oof_preds["target"] == target)].copy()

    if models is None:
        models = list(pd.unique(sub["model"]))

    # optionally drop some models from the comparison set
    if drop_models:
        models = [m for m in models if m not in set(drop_models)]
        sub = sub[~sub["model"].isin(drop_models)]

    results, pvals, tmp = [], [], []

    # for each model pair compute bootstrap R² difference and raw p
    for A, B in combinations(models, 2):
        oofA = sub[sub["model"] == A]
        oofB = sub[sub["model"] == B]
        _, summ = bootstrap_r2_diff(oofA, oofB, n_boot=n_boot, random_state=random_state)
        pvals.append(summ["p_boot"])
        tmp.append((A, B, summ))

    # adjust p-values for multiple testing
    p_adj = adjust_pvals(pvals, method=adjust)

    rows = []
    for ((A, B, summ), p_corr) in zip(tmp, p_adj):
        rows.append({
            "task": task, "target": target, "A": A, "B": B,
            **summ, "p_raw": summ["p_boot"], "p_adj": float(p_corr), "adjust": adjust
        })
    out = pd.DataFrame(rows).sort_values("p_adj")
    return out

