import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
import seaborn as sns

from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
    recall_score, precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)

# add project root
PROJECT_ROOT = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.constants import GIT_DIRECTORY, RANDOM_STATE, CI, N_BOOT, SCORES, ID_COL
from config.feature_sets import get_linguistic_features, get_acoustic_features, get_demographic_features
from data_preparation.data_handling import (load_demographics,
    load_task_dataframe,
    build_feature_sets,
)
from regression.model_evaluation import (
    confidence_intervals,
    bootstrap_index_generator,
)
from regression.hyperparameter_tuning import parameter_window


# default settings
TASK = "picnicScene"
MODELS = ["demographics", "acoustic", "linguistic", "linguistic+acoustic", "full"]
BASELINE_THRESHOLD = 0.5

DEFAULT_RF_CL_PARAMS = dict(
    n_estimators=975,
    random_state=RANDOM_STATE,
    bootstrap=True,
    class_weight="balanced",
    max_depth=16,
    max_features="log2",
    min_samples_leaf=3,
    min_samples_split=3
)

label_map = {
        "full": "full",
        "linguistic+acoustic": "linguistic + acoustic",
        "linguistic": "linguistic",
        "acoustic": "acoustic",
        "demographics": "demographic",
    }

# config
DATA_DIR = os.path.join(GIT_DIRECTORY, "data")
SCORES_PATH = os.path.join(DATA_DIR, "language_scores_all_subjects.csv")
DEMO_PATH = os.path.join(DATA_DIR, "demographics_data.csv")
HALF_SPLIT_PATH = os.path.join(GIT_DIRECTORY, "data", "stratified_folds_half.csv")
CL_RESULTS_DIR = os.path.join(GIT_DIRECTORY, "results", "classification")
os.makedirs(CL_RESULTS_DIR, exist_ok=True)
NORMS_PATH = os.path.join(CL_RESULTS_DIR, "population_norms.csv")

scores = pd.read_csv(SCORES_PATH)
demographics = load_demographics(DEMO_PATH)
demographic_variables = get_demographic_features()
half_split = pd.read_csv(HALF_SPLIT_PATH)
half_split = half_split.rename(columns={"fold": "half_split"})

# plot style
palette = sns.color_palette("Set2", n_colors=len(MODELS))
colors = {
    m: matplotlib.colors.to_hex(c)
    for m, c in zip(MODELS, palette[::-1])
}

plt.style.use("default")
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


# calculate norms for each score (mean, SD, low/high thresholds and proportion in categories)
def build_population_norms(scores_df, score_cols, save_path=None):
    rows = []
    for score in score_cols:
        s = scores_df[score].dropna().astype(float)
        n = int(s.shape[0])
        if n == 0:
            continue

        mean = float(s.mean())
        std = float(s.std(ddof=1))
        z = (s - mean) / std
        low_mask = z < -1
        low_n = int(low_mask.sum())

        rows.append({
            "score": score,
            "mean": mean,
            "std": std,
            "low_threshold": mean - std,
            "high_threshold": mean + std,
            "n": n,
            "low_n": low_n,
            "not_low_n": n - low_n,
            "low_prop": (low_n / n),
            "not_low_prop": 1 - (low_n / n),
        })

    norms = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        norms.to_csv(save_path, index=False)

    return norms

# if norms file doesn't exist yet, build & save it once
if not os.path.exists(NORMS_PATH):
    print(f"[classification] calculating population_norms, saved to: {NORMS_PATH}")
    build_population_norms(
        scores_df=scores,
        score_cols=SCORES,
        save_path=NORMS_PATH,
    )

# load pre-calculated norms for classification
def load_population_norms(path):
    norms_df = pd.read_csv(path)
    norms = {}
    for _, r in norms_df.iterrows():
        norms[str(r["score"])] = {
            "mean": float(r["mean"]),
            "std": float(r["std"]),
            "low_threshold": float(r["low_threshold"]),
            "high_threshold": float(r["high_threshold"]),
            "n": int(r["n"]),
        }
    return norms

population_norms = load_population_norms(NORMS_PATH)

# add z-scores for participants and binary label low if < -1 SD
def build_z_scores_and_labels(df, target_score):
    if target_score not in population_norms:
        raise ValueError(f"no population norm found for {target_score} in {NORMS_PATH}")
    mean = population_norms[target_score]["mean"]
    std = population_norms[target_score]["std"]

    z = (df[target_score] - mean) / std
    df[f"{target_score}_z"] = z
    df["is_low"] = (z < -1).astype(int)
    return df, float(mean), float(std)


# Random Forest classifier hyperparameter tuning
def classification_hyperparameter_tuning(
    task_name=TASK,
    target="SemanticFluencyScore",
    model_name="full",
    n_iter_random=100,
    refine_with_grid=True,
    test_half_label=1,
    random_state=RANDOM_STATE,
):
    outdir = os.path.join(CL_RESULTS_DIR, "grid_search")
    os.makedirs(outdir, exist_ok=True)

    if model_name != "full":
        raise NotImplementedError(
            "classification_hyperparameter_tuning currently only supports model_name='full'."
        )

    scores_all = pd.read_csv(SCORES_PATH)
    scores_sub = scores_all[[ID_COL, target]].dropna(subset=[target]).copy()
    folds = half_split

    features_path = os.path.join(
        GIT_DIRECTORY,
        "results", "features", "cleaned",
        f"{task_name}_cleaned.csv",
    )
    X_task = pd.read_csv(features_path)

    demo = demographics[[ID_COL] + demographic_variables].copy()

    for df in (scores_sub, X_task, demo, folds):
        df[ID_COL] = df[ID_COL].astype(str)

    df = (
        X_task.merge(demo, on=ID_COL, how="left")
        .merge(scores_sub, on=ID_COL, how="inner")
        .merge(folds[[ID_COL, "half_split"]], on=ID_COL, how="inner")
    )

    linguistic = get_linguistic_features()
    acoustic = get_acoustic_features()
    feature_columns = sorted(list(linguistic | acoustic)) + demographic_variables
    feature_columns = [c for c in feature_columns if c in df.columns]

    df = df.dropna(subset=[target] + feature_columns).reset_index(drop=True)

    train_df = df[df["half_split"] != test_half_label].reset_index(drop=True)
    test_df = df[df["half_split"] == test_half_label].reset_index(drop=True)

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = (((train_df[target] - train_df[target].mean()) / train_df[target].std()) < -1).astype(int).to_numpy()
    y_test = (((test_df[target] - train_df[target].mean()) / train_df[target].std()) < -1).astype(int).to_numpy()

    print(f"{task_name} | full | train N={len(X_train)}, test N={len(X_test)}, features={len(feature_columns)}")

    inner_cv = list(
        KFold(n_splits=5, shuffle=True, random_state=random_state)
        .split(np.arange(len(X_train)))
    )

    param_dist = {
        "n_estimators": randint(100, 1001),
        "max_depth": [None] + list(range(10, 31)),
        "max_features": ["sqrt", "log2"],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "bootstrap": [True, False],
        "class_weight": ["balanced"],
    }

    rs = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter_random,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        refit=True,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    best_cv = float(rs.best_score_)
    pd.DataFrame(rs.cv_results_).to_csv(
        os.path.join(outdir, "randomized_search_cv_results.csv"),
        index=False,
    )
    print("RandomizedSearch best params:", best_params)
    print("RandomizedSearch best inner-CV AUROC:", f"{best_cv:.3f}")

    final_params = best_params

    if refine_with_grid:
        bp = best_params

        n_estimators_step = 50
        grid = {
            "n_estimators": parameter_window(bp["n_estimators"], radius=200, low=100, high=1000, step=n_estimators_step),
            "max_depth": [None] if bp["max_depth"] is None else parameter_window(bp["max_depth"], radius=3, low=10, high=30, step=1),
            "max_features": ["sqrt", "log2"],
            "min_samples_split": parameter_window(bp["min_samples_split"], radius=1, low=2, high=10, step=1),
            "min_samples_leaf": parameter_window(bp["min_samples_leaf"], radius=1, low=1, high=5, step=1),
            "bootstrap": [True, False],
            "class_weight": ["balanced"],
        }

        gs = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
            param_grid=grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_train, y_train)
        best_params = gs.best_params_
        best_cv = float(gs.best_score_)
        pd.DataFrame(gs.cv_results_).to_csv(
            os.path.join(outdir, "grid_search_cv_results.csv"),
            index=False,
        )
        print("GridSearch best params:", best_params)
        print("GridSearch best inner-CV AUROC:", f"{best_cv:.3f}")
        final_params = best_params

    pd.DataFrame([final_params]).to_csv(
        os.path.join(outdir, "best_params_final.csv"),
        index=False,
    )
    return final_params

# load tuned hyperparameters for Random Forest classifier
def load_tuned_rf_params():
    tuned_path = os.path.join(CL_RESULTS_DIR, "grid_search", "best_params_final.csv")
    if os.path.exists(tuned_path):
        try:
            df = pd.read_csv(tuned_path)
            params = df.iloc[0].to_dict()
            params["class_weight"] = "balanced"
            params["random_state"] = RANDOM_STATE
            params["n_jobs"] = -1
            return params

        except Exception as e:
            print("Failed to read tuned params, using defaults. Error:", e)
    return dict(DEFAULT_RF_CL_PARAMS)


# calculate classification metrics: AUROC, AUPRC, sensitivity, specificity, accuracy, precision, F1, MCC
def classification_metrics(y_true, y_prob, threshold=BASELINE_THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # threshold-independent
    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    # threshold-dependent
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }

# get bootstrapped estimates of classification metrics (mean ± 95% CI)
def bootstrap_classification_metrics(oof_df, threshold=BASELINE_THRESHOLD,
                                     n_boot=N_BOOT, ci=CI, random_state=RANDOM_STATE):
    y = oof_df["y_true"].to_numpy()
    p = oof_df["y_prob"].to_numpy()
    n = len(y)

    idxs = bootstrap_index_generator(n, n_boot, random_state)
    rows = []

    for b in range(n_boot):
        idx = idxs[b]
        yb, pb = y[idx], p[idx]
        rows.append(classification_metrics(yb, pb, threshold=threshold))

    boot = pd.DataFrame(rows)
    out = {}
    for m in [
        "auroc", "auprc",
        "sensitivity", "specificity",
        "accuracy", "precision", "f1", "mcc"
    ]:
        lo, hi = confidence_intervals(boot[m].to_numpy(), ci=ci)
        out[f"{m}_mean"] = float(np.nanmean(boot[m]))
        out[f"{m}_ci_low"] = float(lo)
        out[f"{m}_ci_high"] = float(hi)

    out["threshold"] = threshold
    return boot, pd.DataFrame([out])

# load bootstrapped threshold-dependent metrics for plotting
def load_bootstrapped_threshold_metrics(outdir, task, target, model, threshold_type="max_f1"):

    prefix = f"{task}_{target}_{model}"
    path = os.path.join(
        outdir,
        target,
        model,
        f"{prefix}_bootstrapped_metrics.csv",
    )
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    row = df[df["threshold_type"] == threshold_type]
    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "threshold": np.nan,
        "precision": float(r.get("precision_mean", np.nan)),
        "recall": float(r.get("sensitivity_mean", np.nan)),
        "f1": float(r.get("f1_mean", np.nan)),
    }

# determine threshold that maximizes F1-score (balanced precision-recall)
def find_best_threshold_f1(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob, pos_label=1)
    t = np.append(t, 1.0)

    den = p + r
    f1 = np.zeros_like(den)
    valid = den > 0
    f1[valid] = 2 * p[valid] * r[valid] / den[valid]

    idx = int(np.nanargmax(f1))
    return {
        "best_threshold": float(t[idx]),
        "precision": float(p[idx]),
        "recall": float(r[idx]),
        "f1": float(f1[idx]),
    }

# determine lowest threshold where recall is ≥.80
def find_threshold_for_target_recall(y_true, y_prob, target_recall=0.80):
    p, r, t = precision_recall_curve(y_true, y_prob, pos_label=1)
    t = np.append(t, 1.0)
    idxs = np.where(r >= target_recall)[0]
    if len(idxs) == 0:
        idx = int(np.argmin(np.abs(r - target_recall)))
    else:
        idx = int(idxs[-1])
    return {
        "threshold": float(t[idx]),
        "precision": float(p[idx]),
        "recall": float(r[idx]),
    }

# determine classification thresholds on one half of the data and evaluate on other half
def evaluate_thresholds_half_split(oof_df, target_recall=0.80):
    if half_split is None:
        return []

    # make sure ID type matches on both sides
    df = oof_df.copy()
    df[ID_COL] = df[ID_COL].astype(str)

    folds = half_split.copy()
    folds[ID_COL] = folds[ID_COL].astype(str)

    df = df.merge(
        folds[[ID_COL, "half_split"]],
        on=ID_COL,
        how="inner",
    )

    two_folds = sorted(df["half_split"].dropna().unique())
    if len(two_folds) != 2:
        raise ValueError(
            f"Expected exactly 2 half-split labels, got {two_folds}"
        )

    train_label, test_label = two_folds[0], two_folds[1]

    train = df[df["half_split"] == train_label].copy()
    test = df[df["half_split"] == test_label].copy()

    y_train = train["y_true"].to_numpy().astype(int)
    p_train = train["y_prob"].to_numpy().astype(float)

    y_test = test["y_true"].to_numpy().astype(int)
    p_test = test["y_prob"].to_numpy().astype(float)

    # thresholds chosen on train half
    f1_best = find_best_threshold_f1(y_train, p_train)
    rec80 = find_threshold_for_target_recall(
        y_train, p_train, target_recall=target_recall
    )

    # metrics evaluated on test half
    f1_metrics = classification_metrics(
        y_test, p_test, threshold=f1_best["best_threshold"]
    )
    rec_metrics = classification_metrics(
        y_test, p_test, threshold=rec80["threshold"]
    )

    rows = []
    rows.append({
        "threshold_type": "max_f1",
        **f1_metrics,
        **f1_best,
    })
    rows.append({
        "threshold_type": "recall>=0.80",
        **rec_metrics,
        **rec80,
    })

    return rows

# bootstrap threshold-dependent classification metrics at determined thresholds
def bootstrap_half_split_threshold_metrics(
    oof_df,
    target_recall=0.80,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
):

    df = oof_df.copy().reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("No rows in oof_df for bootstrap.")

    # shared bootstrap indices
    idxs = bootstrap_index_generator(n, n_boot, random_state)

    rows = []
    for b in range(n_boot):
        ib = idxs[b]
        df_b = df.iloc[ib].reset_index(drop=True)

        try:
            thr_rows = evaluate_thresholds_half_split(
                df_b,
                target_recall=target_recall,
            )
        except Exception as e:
            continue

        for r in thr_rows:
            r_out = r.copy()
            r_out["boot"] = b
            rows.append(r_out)

    if not rows:
        raise ValueError("No valid bootstrap samples for half-split thresholds.")

    boot = pd.DataFrame(rows)

    # threshold-dependent metrics
    metric_cols = [
        "sensitivity",
        "specificity",
        "accuracy",
        "precision",
        "f1",
        "mcc",
    ]

    groups = []
    alpha = 1.0 - ci

    for thr_type, g in boot.groupby("threshold_type", dropna=False):
        rec = {"threshold_type": thr_type}
        for m in metric_cols:
            if m not in g.columns:
                continue
            vals = g[m].to_numpy().astype(float)
            rec[f"{m}_mean"] = float(np.nanmean(vals))
            rec[f"{m}_ci_low"] = float(np.nanquantile(vals, alpha / 2.0))
            rec[f"{m}_ci_high"] = float(np.nanquantile(vals, 1.0 - alpha / 2.0))
        groups.append(rec)

    boot_summ = pd.DataFrame(groups)

    return boot, boot_summ

# plot bootstrapped ROC-curve (for selected model)
def plot_roc_curve(oof_df, save_path, n_boot=N_BOOT, random_state=RANDOM_STATE):
    y = oof_df["y_true"].to_numpy()
    p = oof_df["y_prob"].to_numpy()
    n = len(y)
    fpr_grid = np.linspace(0, 1, 101)

    idxs = bootstrap_index_generator(n, n_boot, random_state)
    tprs = np.empty((n_boot, fpr_grid.size), dtype=float)
    aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = idxs[b]
        fpr_b, tpr_b, _ = roc_curve(y[idx], p[idx])
        aucs[b] = roc_auc_score(y[idx], p[idx]) if len(np.unique(y[idx])) == 2 else np.nan
        tprs[b] = np.interp(fpr_grid, fpr_b, tpr_b, left=0.0, right=1.0)

    tpr_mean = np.nanmean(tprs, axis=0)
    tpr_lo = np.nanquantile(tprs, 0.025, axis=0)
    tpr_hi = np.nanquantile(tprs, 0.975, axis=0)
    auc_mean = np.nanmean(aucs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_grid, tpr_mean, label="Mean ROC (bootstrapped)", color="C0")
    plt.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, label="95% CI", color="C0")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="orange", label="Chance")
    plt.plot([], [], " ", label=f"AUROC = {auc_mean:.3f}")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

# plot bootstrapped ROC curves for all models for specific target
def plot_all_roc_curves(
    task,
    target,
    models,
    outdir,
    n_boot=N_BOOT,
    random_state=RANDOM_STATE,
    save_name=None,
    show_ci="none",  # "none" / "all" / "one"
    ci_model="full",
    legend_metric="none",  # "none" / "mean" (AUROC)
):
    fpr_grid = np.linspace(0, 1, 101)

    file_prefix = "{task}_{target}_{model}_oof.csv"

    data = {}
    for m in models:
        model_dir = os.path.join(outdir, target, m)
        path = os.path.join(model_dir, file_prefix.format(task=task, target=target, model=m))
        df = pd.read_csv(path)

        df = df[[ID_COL, "y_true", "y_prob"]].dropna().copy()
        df[ID_COL] = df[ID_COL].astype(str)
        data[m] = df

    common = set.intersection(*(set(d[ID_COL]) for d in data.values()))
    for m in models:
        data[m] = (data[m][data[m][ID_COL].isin(common)]
                   .sort_values(ID_COL).reset_index(drop=True))
    N = len(next(iter(data.values())))
    if N == 0:
        raise ValueError("After alignment, there are 0 samples.")

    results = {}
    idxs = bootstrap_index_generator(N, n_boot, random_state)
    for m in models:
        model_df = data[m]
        y = model_df["y_true"].to_numpy().astype(int)
        p = model_df["y_prob"].to_numpy().astype(float)

        tprs = np.empty((n_boot, fpr_grid.size))
        aucs = np.empty(n_boot)
        for b in range(n_boot):
            ii = idxs[b]
            yb, pb = y[ii], p[ii]
            if np.unique(yb).size < 2:
                tprs[b, :] = np.nan
                aucs[b] = np.nan
                continue
            fpr_b, tpr_b, _ = roc_curve(yb, pb)
            aucs[b] = roc_auc_score(yb, pb)
            tprs[b, :] = np.interp(fpr_grid, fpr_b, tpr_b, left=0.0, right=1.0)

        results[m] = {
            "tpr_mean": np.nanmean(tprs, axis=0),
            "tpr_lo": np.nanquantile(tprs, 0.025, axis=0),
            "tpr_hi": np.nanquantile(tprs, 0.975, axis=0),
            "auc_mean": float(np.nanmean(aucs)),
        }

    plt.figure(figsize=(6.8, 5.6))
    for m in models:
        r = results[m]
        lw = 1.2
        z = 2 if m != "full" else 3
        display_name = label_map.get(m, m)

        if legend_metric == "mean":
            label = f"{display_name} | AUROC = {r['auc_mean']:.3f}"
        else:
            label = display_name

        if show_ci == "all" or (show_ci == "one" and m == ci_model):
            plt.fill_between(
                fpr_grid, r["tpr_lo"], r["tpr_hi"],
                alpha=0.15, color=colors.get(m, None), zorder=1,
            )

        plt.plot(
            fpr_grid, r["tpr_mean"],
            label=label, linewidth=lw, color=colors.get(m, None), zorder=z,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Chance")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(alpha=0.15, linestyle="--")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_name is None:
        save_name = f"{task}_{target}_all_ROC_curves.png"
    save_path = os.path.join(outdir, save_name)
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", save_path)
    return results

# plot precision-recall curve (for specific model)
def plot_pr_curve(oof_df, save_path):
    y_true = oof_df["y_true"].to_numpy().astype(int)
    y_prob = oof_df["y_prob"].to_numpy().astype(float)

    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    ap = average_precision_score(y_true, y_prob)
    prev = y_true.mean()

    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post", label="Precision-Recall curve")
    plt.hlines(prev, 0, 1, linestyles="--", linewidth=1,
               label=f"Baseline (prevalence={prev:.2f})")
    plt.plot([], [], " ", label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

# plot PR curves for all models for one target
def plot_all_pr_curves(
    task,
    target,
    models,
    outdir,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
    save_name=None,
    legend_metric="mean", # "none" / "mean" (AUPRC)
    show_f1_for="full", # "all" / model name / None
):

    file_prefix = "{task}_{target}_{model}_oof.csv"

    data = {}
    for m in models:
        model_dir = os.path.join(outdir, target, m)
        path = os.path.join(
            model_dir,
            file_prefix.format(task=task, target=target, model=m)
        )
        df = pd.read_csv(path)

        df = df[[ID_COL, "y_true", "y_prob"]].dropna().copy()
        df[ID_COL] = df[ID_COL].astype(str)
        data[m] = df

    common = set.intersection(*(set(d[ID_COL]) for d in data.values()))
    if not common:
        raise ValueError("After alignment, there are 0 common samples across models.")

    for m in models:
        data[m] = (
            data[m][data[m][ID_COL].isin(common)]
            .sort_values(ID_COL)
            .reset_index(drop=True)
        )

    N = len(next(iter(data.values())))
    if N == 0:
        raise ValueError("No samples after alignment.")

    idxs = bootstrap_index_generator(N, n_boot, random_state)
    alpha = 1.0 - ci

    results = {}
    for m in models:
        model_df = data[m]
        y = model_df["y_true"].to_numpy().astype(int)
        p = model_df["y_prob"].to_numpy().astype(float)

        pr_prec, pr_rec, _ = precision_recall_curve(y, p, pos_label=1)
        ap = average_precision_score(y, p)

        f1_info = find_best_threshold_f1(y, p)

        auprcs = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            ii = idxs[b]
            yb, pb = y[ii], p[ii]
            if np.unique(yb).size < 2:
                auprcs[b] = np.nan
            else:
                auprcs[b] = average_precision_score(yb, pb)

        ci_low = np.nanquantile(auprcs, alpha / 2.0)
        ci_high = np.nanquantile(auprcs, 1.0 - alpha / 2.0)

        results[m] = {
            "precision": pr_prec,
            "recall": pr_rec,
            "auprc": float(ap),
            "auprc_ci_low": float(ci_low),
            "auprc_ci_high": float(ci_high),
            "f1_info": f1_info,
        }

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    any_m = models[0]
    prevalence = data[any_m]["y_true"].mean()

    f1_value = None

    for m in models:
        r = results[m]
        display_name = label_map.get(m, m)
        col = colors.get(m, None)
        lw = 1.2
        z = 2 if m != "full" else 3

        if legend_metric == "mean":
            label = f"{display_name} | AUPRC = {r['auprc']:.3f}"
        else:
            label = display_name

        ax.step(
            r["recall"],
            r["precision"],
            where="post",
            linewidth=lw,
            color=col,
            label=label,
            zorder=z,
        )

        if show_f1_for in ("all", m):
            fi = r["f1_info"]
            ax.scatter(
                [fi["recall"]],
                [fi["precision"]],
                s=36,
                facecolors="white",
                edgecolors=col if col is not None else "black",
                zorder=z + 1,
            )
            if m == show_f1_for:
                f1_value = fi["f1"]

    baseline_label = f"Baseline (prevalence = {prevalence:.2f})"
    ax.hlines(
        prevalence, 0.0, 1.0,
        linestyles="--", linewidth=1,
        color="gray", label=baseline_label,
    )

    handles, labels = ax.get_legend_handles_labels()

    if show_f1_for in models and f1_value is not None:
        f1_marker = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=colors.get(show_f1_for, "black"),
            markersize=6,
        )
        f1_label = f"max F1 ({show_f1_for} model) = {f1_value:.3f}"
        handles.append(f1_marker)
        labels.append(f1_label)

    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.grid(alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_name is None:
        save_name = f"{task}_{target}_all_PR_curves.png"

    save_path = os.path.join(outdir, save_name)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", save_path)
    return results

# plot bootstrapped PR curves for all models for one target
def plot_all_pr_curves_bootstrapped(
    task,
    target,
    models,
    outdir,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
    save_name=None,
    legend_metric="mean", # "none" / "mean" (AUPRC)
    show_ci="one", # "none" / "all" / "one"
    ci_model="full", # if show_ci=="one"
    show_f1_for="full", # "all" / model name / None
):

    alpha = 1.0 - ci

    file_prefix = "{task}_{target}_{model}_oof.csv"

    # load combined bootstrapped metrics (so legend matches tables)
    metrics_df = None
    metrics_path = os.path.join(
        outdir,
        target,
        f"{task}_{target}_bootstrapped_metrics_all_models.csv",
    )
    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Warning: failed to read combined metrics from {metrics_path}: {e}")
            metrics_df = None

    #  load OOFs
    data = {}
    for m in models:
        model_dir = os.path.join(outdir, target, m)
        path = os.path.join(
            model_dir,
            file_prefix.format(task=task, target=target, model=m)
        )
        df = pd.read_csv(path)

        df = df[[ID_COL, "y_true", "y_prob"]].dropna().copy()
        df[ID_COL] = df[ID_COL].astype(str)
        data[m] = df

    # align on common subjects
    common = set.intersection(*(set(d[ID_COL]) for d in data.values()))
    if not common:
        raise ValueError("After alignment, there are 0 common samples across models.")

    for m in models:
        data[m] = (
            data[m][data[m][ID_COL].isin(common)]
            .sort_values(ID_COL)
            .reset_index(drop=True)
        )

    N = len(next(iter(data.values())))
    if N == 0:
        raise ValueError("No samples after alignment.")

    # shared bootstrap indices for consistency
    idxs = bootstrap_index_generator(N, n_boot, random_state)
    recall_grid = np.linspace(0.0, 1.0, 101)

    results = {}

    # PR + bootstraps for each model
    for m in models:
        model_df = data[m]
        y = model_df["y_true"].to_numpy().astype(int)
        p = model_df["y_prob"].to_numpy().astype(float)

        # full-data PR curve
        pr_prec, pr_rec, _ = precision_recall_curve(y, p, pos_label=1)
        ap_full = average_precision_score(y, p)

        # bootstrap AUPRC + PR curves
        auprcs = np.empty(n_boot, dtype=float)
        pr_boot = np.empty((n_boot, recall_grid.size), dtype=float)

        for b in range(n_boot):
            ii = idxs[b]
            yb, pb = y[ii], p[ii]

            if np.unique(yb).size < 2:
                auprcs[b] = np.nan
                pr_boot[b, :] = np.nan
                continue

            prec_b, rec_b, _ = precision_recall_curve(yb, pb, pos_label=1)

            order = np.argsort(rec_b)
            rec_b_sorted = rec_b[order]
            prec_b_sorted = prec_b[order]

            pr_boot[b, :] = np.interp(
                recall_grid,
                rec_b_sorted,
                prec_b_sorted,
                left=prec_b_sorted[0],
                right=0.0,
            )

            auprcs[b] = average_precision_score(yb, pb)

        ci_low = np.nanquantile(auprcs, alpha / 2.0)
        ci_high = np.nanquantile(auprcs, 1.0 - alpha / 2.0)
        auprc_mean = float(np.nanmean(auprcs))  # bootstrapped mean AUPRC

        prec_mean = np.nanmean(pr_boot, axis=0)
        prec_lo = np.nanquantile(pr_boot, 0.025, axis=0)
        prec_hi = np.nanquantile(pr_boot, 0.975, axis=0)

        # AUPRC from csv
        auprc_from_csv = None
        if metrics_df is not None:
            row_base = metrics_df[
                (metrics_df.get("model") == m)
                & (metrics_df.get("threshold_type") == "baseline_0.50")
            ]
            if not row_base.empty and "auprc_mean" in row_base.columns:
                try:
                    auprc_from_csv = float(row_base["auprc_mean"].iloc[0])
                except Exception:
                    auprc_from_csv = None

        results[m] = {
            "prec_full": pr_prec,
            "rec_full": pr_rec,
            "prec_mean": prec_mean,
            "prec_lo": prec_lo,
            "prec_hi": prec_hi,
            "auprc_full": float(ap_full),
            "auprc_mean": auprc_mean,
            "auprc_ci_low": float(ci_low),
            "auprc_ci_high": float(ci_high),
            "auprc_from_csv": auprc_from_csv,
        }

        # load bootstrapped half-split F1 metrics for this model (max_f1 criterion)
        thr_info = load_bootstrapped_threshold_metrics(
            outdir=outdir,
            task=task,
            target=target,
            model=m,
            threshold_type="max_f1",
        )
        results[m]["thr_info_max_f1"] = thr_info

    # plot
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    any_m = models[0]
    prevalence = data[any_m]["y_true"].mean()

    f1_value = None  # to show in legend for the chosen model

    for m in models:
        r = results[m]
        display_name = label_map.get(m, m)
        col = colors.get(m, None)
        lw = 1.2
        z = 2 if m != "full" else 3

        # choose AUPRC for legend
        if legend_metric == "mean":
            auprc_for_legend = (
                r["auprc_from_csv"]
                if r.get("auprc_from_csv") is not None and np.isfinite(r["auprc_from_csv"])
                else r["auprc_mean"]
            )
            label = f"{display_name} | AUPRC = {auprc_for_legend:.3f}"
        else:
            label = display_name

        # CI band
        if show_ci == "all" or (show_ci == "one" and m == ci_model):
            ax.fill_between(
                recall_grid,
                r["prec_lo"],
                r["prec_hi"],
                alpha=0.15,
                color=col,
                zorder=1,
            )

        # mean PR curve
        ax.plot(
            recall_grid,
            r["prec_mean"],
            linewidth=lw,
            color=col,
            label=label,
            zorder=z,
        )

        # marker for max-F1 operating point from bootstrapped half-split metrics
        thr_info = r.get("thr_info_max_f1", None)
        if show_f1_for in ("all", m) and thr_info is not None:
            rec_f1 = thr_info.get("recall", None)
            if rec_f1 is not None and np.isfinite(rec_f1):
                idx_f1 = int(np.argmin(np.abs(recall_grid - rec_f1)))
                rec_plot = recall_grid[idx_f1]
                prec_plot = r["prec_mean"][idx_f1]

                ax.scatter(
                    [rec_plot],
                    [prec_plot],
                    s=36,
                    facecolors="white",
                    edgecolors=col if col is not None else "black",
                    zorder=z + 1,
                )

                if m == show_f1_for:
                    f1_value = thr_info.get("f1", None)

    # baseline: prevalence
    baseline_label = f"Baseline (prevalence = {prevalence:.2f})"
    ax.hlines(
        prevalence,
        0.0,
        1.0,
        linestyles="--",
        linewidth=1,
        color="gray",
        label=baseline_label,
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.grid(alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()

    # legend entry for max F1 marker (for selected model)
    if show_f1_for in models and f1_value is not None and np.isfinite(f1_value):
        f1_marker = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=colors.get(show_f1_for, "black"),
            markersize=6,
        )
        f1_label = f"max F1 = {f1_value:.3f}"
        handles.append(f1_marker)
        labels.append(f1_label)

    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    fig.tight_layout()

    if save_name is None:
        save_name = f"{task}_{target}_all_PR_curves_bootstrapped.png"
    save_path = os.path.join(outdir, save_name)

    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved bootstrapped PR curves:", save_path)
    return results

# plot confusion matrix
def plot_confusion_matrix(oof_df, save_path):
    y_true = oof_df["y_true"].to_numpy().astype(int)
    y_pred = oof_df["y_pred"].to_numpy().astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Middle-High (0)", "Low (1)"])
    ax.set_yticklabels(["Middle-High (0)", "Low (1)"])
    ax.set_ylabel("True", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)

    vmax = cm.max() if cm.size else 0
    thresh = vmax / 2.0 if vmax > 0 else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color=color, fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

# plot confusion matrix for given classification threshold
def plot_confusion_matrix_at_threshold(oof_df, threshold, save_path, normalize=False):
    y_true = oof_df["y_true"].to_numpy().astype(int)
    y_pred = (oof_df["y_prob"].to_numpy().astype(float) >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count" if not normalize else "Proportion",
                       rotation=270, labelpad=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Middle-High (0)", "Low (1)"])
    ax.set_yticklabels(["Middle-High (0)", "Low (1)"])
    ax.set_ylabel("True", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)

    vmax = np.nanmax(cm) if cm.size else 0
    thresh = vmax / 2.0 if vmax > 0 else 0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color=("white" if val > thresh else "black"), fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()


# prepare to calculate SHAP values
def to_single_output_explanation(expl, class_index=1):
    vals = np.array(expl.values)
    base = np.array(expl.base_values)
    data = np.array(expl.data)

    if vals.ndim == 2:
        single = (vals.shape[0] == 1)
        return shap.Explanation(
            values=vals[0, :] if single else vals,
            base_values=float(np.ravel(base)[0]) if (single and base.size >= 1) else base,
            data=data[0, :] if (data.ndim >= 2 and data.shape[0] == 1) else data,
            feature_names=expl.feature_names,
        )

    if vals.ndim == 1:
        b = float(np.ravel(base)[0]) if base.size >= 1 else base
        d = data if data.ndim == 1 else data[0, :]
        return shap.Explanation(
            values=vals, base_values=b, data=d,
            feature_names=expl.feature_names
        )

    if vals.ndim == 3:
        k = class_index if vals.shape[2] > class_index else 0
        vals_sel = vals[:, :, k]
        base_sel = base[:, k] if base.ndim == 2 else (base[k] if base.ndim == 1 else base)
        single = (vals_sel.shape[0] == 1)
        data_sel = data
        return shap.Explanation(
            values=vals_sel[0, :] if single else vals_sel,
            base_values=float(np.ravel(base_sel)[0]) if (single and np.size(base_sel) >= 1) else base_sel,
            data=data_sel[0, :] if (data_sel.ndim >= 2 and data_sel.shape[0] == 1) else data_sel,
            feature_names=expl.feature_names,
        )

    raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

# create beeswarm plots and table with global feature importance
def shap_beeswarm_and_importance(
    df_use,
    feature_columns,
    fold_col,
    rf_params,
    out_prefix,
    demographic_variables=demographic_variables,
):
    linguistic = get_linguistic_features()
    acoustic = get_acoustic_features()

    def family_of(c):
        if c in linguistic:
            return "linguistic"
        if c in acoustic:
            return "acoustic"
        if c in demographic_variables:
            return "demographics"
        return "other"

    superscript_for = {
        "linguistic": "¹",
        "acoustic": "²",
        "demographics": "³",
        "other": "",
    }

    X_all = df_use[feature_columns].copy()
    y_all = df_use["is_low"].astype(int).to_numpy()
    folds = sorted(df_use[fold_col].unique())

    vals_list, data_list = [], []

    for f in folds:
        mask = df_use[fold_col] == f
        X_tr, y_tr = X_all.loc[~mask], y_all[~mask]
        X_te = X_all.loc[mask]

        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(clf)
        ex_all = explainer(X_te)
        ex = to_single_output_explanation(ex_all, class_index=1)

        vals_list.append(np.array(ex.values))
        data_list.append(np.array(ex.data))

    V = np.vstack(vals_list)
    D = np.vstack(data_list)

    names = list(feature_columns)
    labeled = [f"{n}{superscript_for[family_of(n)]}" for n in names]

    expl = shap.Explanation(
        values=V,
        base_values=np.mean(V, axis=0),
        data=D,
        feature_names=labeled,
    )

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 10))

    shap.plots.beeswarm(expl, max_display=20, show=False)

    ax = plt.gca()
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("Feature")

    plt.figtext(
        0.0,
        0.01,
        "¹ linguistic   ² acoustic   ³ demographics",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        out_prefix + "_shap_beeswarm.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    mean_abs = np.abs(V).mean(axis=0)
    feature_importances = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )
    feature_importances.to_csv(out_prefix + "_shap_importance.csv", index=False)
    return feature_importances



# feature importance waterfall plot for individual subject
def plot_local_waterfall(
    df_use,
    feature_columns,
    rf_params,
    oof_df,
    task,
    target_score,
    model_name,
    subject_id=None,
    outdir=CL_RESULTS_DIR,
    max_display=15,
):

    # pick subject
    if subject_id is None:
        lows = oof_df[oof_df["y_true"] == 1]
        if lows.empty:
            raise ValueError("No true low performers found in this subset.")
        subject_id = (
            lows.sort_values("y_prob", ascending=False)[ID_COL]
            .iloc[0]
        )

    row = df_use[df_use[ID_COL] == subject_id]
    if row.empty:
        raise ValueError(f"Subject {subject_id} not found in df_use.")
    if "fold" not in df_use.columns:
        raise ValueError("df_use must contain a 'fold' column for CV information.")

    s_fold = int(row["fold"].iloc[0])

    # train local RF on all other folds, test on that one subject
    train_df = df_use[df_use["fold"] != s_fold].copy()
    test_row = row.copy()

    X_train = train_df[feature_columns]
    y_train = train_df["is_low"].astype(int)
    X_test_one = test_row[feature_columns]
    y_test_one = int(test_row["is_low"].iloc[0])

    clf_local = RandomForestClassifier(**rf_params)
    clf_local.fit(X_train, y_train)

    # SHAP explanation for that subject
    explainer = shap.TreeExplainer(clf_local)
    ex_all = explainer(X_test_one)
    ex = to_single_output_explanation(ex_all, class_index=1)

    linguistic = get_linguistic_features()
    acoustic = get_acoustic_features()
    families = {
        c: (
            "linguistic" if c in linguistic else
            "acoustic" if c in acoustic else
            "demographics" if c in demographic_variables else
            "other"
        )
        for c in feature_columns
    }
    superscript_for = {
        "linguistic": "¹",
        "acoustic": "²",
        "demographics": "³",
        "other": "",
    }

    labeled = [
        f"{n}{superscript_for.get(families.get(n, 'other'), '')}"
        for n in ex.feature_names
    ]
    ex_labeled = shap.Explanation(
        values=ex.values,
        base_values=ex.base_values,
        data=ex.data,
        feature_names=labeled,
    )

    # save figure
    out_root = outdir
    outdir = os.path.join(out_root, target_score, model_name)
    os.makedirs(outdir, exist_ok=True)

    prefix = f"{task}_{target_score}_{model_name}"
    save_prefix = os.path.join(outdir, prefix)

    pred_prob = float(clf_local.predict_proba(X_test_one)[:, 1][0])
    title = (
        f"Local SHAP Waterfall\n"
        f"Subject {subject_id} | True label: {y_test_one} | "
        f"Predicted P(low)={pred_prob:.3f}"
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(ex_labeled, max_display=max_display, show=False)
    plt.title(title, fontsize=11)
    plt.figtext(
        0.00,
        0.02,
        "¹ linguistic   ² acoustic   ³ demographics",
        ha="left",
        va="top",
        fontsize=10,
    )
    plt.subplots_adjust(left=0.30, right=0.96, bottom=0.15, top=0.85)
    out_path = f"{save_prefix}_waterfall_subject-{subject_id}.png"
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()

    return {
        "Subject_ID": subject_id,
        "fold": s_fold,
        "pred_prob": pred_prob,
        "path": out_path,
    }


# load data for task and target
def load_data_for(task, target_score):
    demo_use = demographics[[ID_COL] + demographic_variables].copy()
    df = load_task_dataframe(task_name=task, target=target_score,
                             scores=scores, demographics=demo_use)
    return df

# run main classifier
def run_classifier(task, target_score, model_name,
                   outdir=CL_RESULTS_DIR,
                   baseline_threshold=BASELINE_THRESHOLD,
                   n_boot=N_BOOT,
                   random_state=RANDOM_STATE):
    rf_params = load_tuned_rf_params()
    df = load_data_for(task, target_score)
    df, mean, std = build_z_scores_and_labels(df, target_score)
    model_features = build_feature_sets(df.columns)
    feature_columns = model_features[model_name]

    subset = [target_score] + feature_columns
    df_use = df.dropna(subset=subset).copy()
    X = df_use[feature_columns].copy()
    y = df_use["is_low"].to_numpy()

    # Random Forest classifier in 5-fold cv
    clf = RandomForestClassifier(**rf_params)
    oof_rows = []
    for f in sorted(df_use["fold"].unique()):
        mask = (df_use["fold"] == f)
        X_train, y_train = X.loc[~mask], y[~mask]
        X_test, y_test = X.loc[mask], y[mask]
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test).astype(int)
        oof_rows.append(pd.DataFrame({
            ID_COL: df_use.loc[mask, ID_COL].values,
            "fold": f,
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }))
    oof = pd.concat(oof_rows, ignore_index=True)

    y_true = oof["y_true"].to_numpy()
    y_prob = oof["y_prob"].to_numpy()

    out_root = outdir
    outdir = os.path.join(out_root, target_score, model_name)
    os.makedirs(outdir, exist_ok=True)

    prefix = f"{task}_{target_score}_{model_name}"

    # classification metrics (point estimates) at different thresholds: baseline 0.50, max F1, recall ≥.80
    rows = []

    # baseline (RF predict with default threshold)
    base_metrics = classification_metrics(
        y_true, y_prob, threshold=baseline_threshold
    )
    rows.append({
        "task": task,
        "target": target_score,
        "model": model_name,
        "threshold_type": "baseline_0.50",
        **base_metrics,
    })

    # half-split thresholds (max F1 & recall≥0.80) evaluated on held-out half
    half_rows = evaluate_thresholds_half_split(oof)
    for r in half_rows:
        rows.append({
            "task": task,
            "target": target_score,
            "model": model_name,
            **r,
        })

    # threshold summary
    thr_records = []
    for r in half_rows:
        if r["threshold_type"] == "max_f1":
            thr_records.append({
                "task": task,
                "target": target_score,
                "model": model_name,
                "criterion": "max_f1",
                "threshold": r.get("threshold", np.nan),
                "precision": r.get("precision", np.nan),
                "recall": r.get("recall", np.nan),
                "f1": r.get("f1", np.nan),
            })
        elif r["threshold_type"] == "recall>=0.80":
            thr_records.append({
                "task": task,
                "target": target_score,
                "model": model_name,
                "criterion": "recall>=0.80",
                "threshold": r.get("threshold", np.nan),
                "precision": r.get("precision", np.nan),
                "recall": r.get("recall", np.nan),
                "f1": r.get("f1", np.nan),
            })

    if thr_records:
        thr_df = pd.DataFrame(thr_records)
        thr_df.to_csv(
            os.path.join(outdir, f"{prefix}_thresholds_summary.csv"),
            index=False,
        )

    # bootstrapped metrics at same classification thresholds

    # baseline threshold (0.50)
    boot_raw_baseline, boot_summ_baseline = bootstrap_classification_metrics(
        oof,
        threshold=baseline_threshold,
        n_boot=n_boot,
        ci=CI,
        random_state=random_state,
    )
    boot_summ_baseline = boot_summ_baseline.assign(
        threshold_type="baseline_0.50"
    )

    # half-split thresholds (max F1 & recall≥0.80)
    boot_raw_half, boot_summ_half = bootstrap_half_split_threshold_metrics(
        oof,
        target_recall=0.80,
        n_boot=n_boot,
        ci=CI,
        random_state=random_state,
    )

    # add meta info
    for df_ in (boot_summ_baseline, boot_summ_half):
        df_["task"] = task
        df_["target"] = target_score
        df_["model"] = model_name

    # combine all bootstrapped metrics for this model
    boot_all_combined = pd.concat(
        [boot_summ_baseline, boot_summ_half],
        ignore_index=True,
    )

    # order
    cols = list(boot_all_combined.columns)
    front = [c for c in ["target", "model", "threshold_type"] if c in cols]
    rest = [c for c in cols if c not in front]
    boot_all_combined = boot_all_combined[front + rest]

    # save per-model bootstrapped table
    boot_all_combined.to_csv(
        os.path.join(outdir, f"{prefix}_bootstrapped_metrics.csv"),
        index=False,
    )

    # save OOF predictions
    oof.to_csv(os.path.join(outdir, f"{prefix}_oof.csv"), index=False)

    # plot ROC curves
    plot_roc_curve(
        oof,
        save_path=os.path.join(outdir, f"{prefix}_roc_bootstrapped.png"),
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
    )

    # plot PR curves
    plot_pr_curve(
        oof,
        save_path=os.path.join(outdir, f"{prefix}_precision-recall.png"),
    )

    # plot confusion matrices at all thresholds

    # RF default .50 confusion matrix (from y_pred column)
    plot_confusion_matrix(
        oof,
        save_path=os.path.join(outdir, f"{prefix}_confusion_matrix.png"),
    )

    # confusion matrices at half-split thresholds (max F1 & recall≥0.80)
    thr_f1 = next(
        (r["threshold"] for r in half_rows if r["threshold_type"] == "max_f1"),
        None,
    )
    if thr_f1 is not None:
        plot_confusion_matrix_at_threshold(
            oof,
            thr_f1,
            save_path=os.path.join(outdir, f"{prefix}_confusion_matrix_maxf1.png"),
            normalize=False,
        )

    thr_rec80 = next(
        (r["threshold"] for r in half_rows
         if r["threshold_type"] == "recall>=0.80"),
        None,
    )
    if thr_rec80 is not None:
        plot_confusion_matrix_at_threshold(
            oof,
            thr_rec80,
            save_path=os.path.join(outdir, f"{prefix}_confusion_matrix_recall080.png"),
            normalize=False,
        )

    # SHAP feature importance (full model only): global -> beeswarm + table; local -> waterfall plot
    if model_name == "full":
        # global SHAP
        try:
            out_prefix = os.path.join(outdir, prefix)
            shap_beeswarm_and_importance(
                df_use=df_use,
                feature_columns=feature_columns,
                fold_col="fold",
                rf_params=rf_params,
                out_prefix=out_prefix,
                demographic_variables=demographic_variables,
            )
        except Exception as e:
            print("Global SHAP (beeswarm) skipped due to error:", e)

        # local SHAP waterfall
        try:
            plot_local_waterfall(
                df_use=df_use,
                feature_columns=feature_columns,
                rf_params=rf_params,
                oof_df=oof,
                task=task,
                target_score=target_score,
                model_name=model_name,
                subject_id=None,  # auto-pick low performer with highest P(low)
                outdir=out_root,
                max_display=15,
            )
        except Exception as e:
            print("Local SHAP waterfall skipped due to error:", e)

    return {
        "oof": oof,
        "boot_baseline_summary": boot_summ_baseline,
        "boot_all_summary": boot_all_combined,
    }


# run classifier for all models
def run_all_models(
    task,
    target,
    models=MODELS,
    outdir=CL_RESULTS_DIR,
):
    boot_list = []

    for m in models:
        print(f"{task} - {target} - {m}")
        out = run_classifier(task, target, m, outdir=outdir)
        boot_list.append(out["boot_all_summary"])

    # combine bootstrapped metrics across models + save
    if boot_list:
        boot_all = pd.concat(boot_list, ignore_index=True)

        # save per score (target) in the score directory
        target_dir = os.path.join(outdir, target)
        os.makedirs(target_dir, exist_ok=True)

        combined_path = os.path.join(
            target_dir,
            f"{task}_{target}_bootstrapped_metrics_all_models.csv",
        )
        boot_all.to_csv(combined_path, index=False)
        print("Saved combined bootstrapped metrics:", combined_path)

    return boot_all if boot_list else None
