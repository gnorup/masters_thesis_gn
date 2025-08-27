import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import r2_score
from pandas.api.types import CategoricalDtype
from itertools import combinations

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY

def format_title(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

def save_crossval_results(
        r2_list, rmse_list, mae_list,
        r2_mean, r2_std, r2_se, r2_ci_low, r2_ci_high,
        task_name, target, output_dir, n_folds=5,
        all_preds=None,
        model_type=None
):

    os.makedirs(output_dir, exist_ok=True)

    # save per-fold results
    cv_df = pd.DataFrame({
        "Fold": list(range(1, n_folds + 1)),
        "R2": r2_list,
        "RMSE": rmse_list,
        "MAE": mae_list,
    })
    cv_df_path = os.path.join(output_dir, f"cv_folds_{task_name}_{target}_{model_type.__name__}.csv")
    cv_df.to_csv(cv_df_path, index=False)

    # save summary
    summary_df = pd.DataFrame({
        "R2_Mean": [r2_mean],
        "R2_Std": [r2_std],
        "R2_SE": [r2_se],
        "R2_CI_Low": [r2_ci_low],
        "R2_CI_High": [r2_ci_high],
        "RMSE_Mean": [np.mean(rmse_list)],
        "MAE_Mean": [np.mean(mae_list)]
    })
    summary_path = os.path.join(output_dir, f"cv_summary_{task_name}_{target}_{model_type.__name__}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"saved per-fold cross-validation results to: {cv_df_path}")
    print(f"saved cross-validation summary to: {summary_path}")

    # optional: save scatterplot
    matplotlib.rcParams['font.family'] = 'Arial'
    # map model types for better readability in titles
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest Regression"
    }
    model_type_display_name = model_name_mapping.get(model_type.__name__, model_type.__name__)
    formatted_task_name = format_title(task_name)

    if all_preds is not None:
        plt.figure(figsize=(7, 6))
        for fold in range(1, n_folds + 1):
            fold_df = all_preds[all_preds['fold'] == fold]
            plt.scatter(fold_df["y_test"], fold_df["y_pred"], label=f"Fold {fold}", alpha=0.7)

        plt.plot(
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            [all_preds["y_test"].min(), all_preds["y_test"].max()],
            linestyle='--', color='gray', label="Perfect Prediction"
        )

        plt.xlabel("Actual Score", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted Score", fontsize=12, fontweight='bold')
        plt.title(f"{formatted_task_name}: Cross-Validated Predictions ({model_type_display_name})", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"cv_prediction_plot_{task_name}_{target}_{model_type.__name__}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"cross-validation prediction plot saved to {plot_path}")

def plot_per_fold_predictions(all_preds_df, task_name, target, model_type, output_dir, n_folds=5):

    # map model types for better readability in titles
    model_name_mapping = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "RandomForestRegressor": "Random Forest Regression"
    }
    model_type_display_name = model_name_mapping.get(model_type.__name__, model_type.__name__)
    format_task_name = format_title(task_name)
    format_target_name = format_title(target)

    matplotlib.rcParams['font.family'] = 'Arial'

    # create plot
    for fold in range(1, n_folds + 1):
        fold_df = all_preds_df[all_preds_df['fold'] == fold]
        y_test, y_pred = fold_df['y_test'], fold_df['y_pred']
        r2 = r2_score(y_test, y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, s=20, alpha=0.7, color='steelblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='gray')

        # legend and labels
        custom_legend = [plt.Line2D([0], [0], color='gray', linestyle='--', label='Perfect Prediction')]
        plt.legend(handles=custom_legend, loc="upper left", fontsize=10, frameon=True)
        plt.text(x=0.05, y=0.90, s=f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=10)

        plt.xlabel("Actual Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Predicted Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f"Fold {fold}: Predicted vs. Actual {format_target_name}\n({model_type_display_name}, {format_task_name})", fontsize=14, fontweight='bold', pad=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plot_path = os.path.join(output_dir, f"fold{fold}_actual_vs_predicted_{target}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"plot: actual vs predicted scores for fold {fold} saved to {plot_path}")


### model comparison helpers

def load_task_dataframe(task_name, target, scores, demographics):
    features_path = os.path.join(GIT_DIRECTORY, f"results/features/filtered/{task_name}_filtered.csv")
    folds_path = os.path.join(GIT_DIRECTORY, "data/stratified_folds.csv")
    features = pd.read_csv(features_path)
    folds = pd.read_csv(folds_path)
    df = pd.merge(features, scores[["Subject_ID", target]], on="Subject_ID")
    df = pd.merge(df, demographics, on="Subject_ID")
    df = pd.merge(df, folds[["Subject_ID", "fold"]], on="Subject_ID")
    return df

def get_model_feature_list(df_columns, selected_features, target_name):
    drop = {"Subject_ID", "fold", target_name}
    safe = []
    for f in selected_features:
        if f in drop:
            continue
        if f in df_columns:
            safe.append(f)
    return safe

def complete_subjects(df, feature_cols, target_name):
    need = [target_name] + (feature_cols if len(feature_cols) > 0 else [])
    return set(df.dropna(subset=need)["Subject_ID"]) # Subject_IDs with no missing in target & all features

def subjects_with_all_features_and_scores(full_subjects, scores_df, score_cols):
    if "Subject_ID" not in scores_df.columns:
        raise ValueError("scores_df must contain 'Subject_ID'")
    all_scores = set(scores_df.dropna(subset=list(score_cols))["Subject_ID"])
    return set(full_subjects) & all_scores

# bootstrap

def _normalize_oof_df(all_preds, target_col=None):
    if isinstance(all_preds, list):
        df = pd.concat(all_preds, ignore_index=True)
    else:
        df = all_preds.copy()

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"subject_id", "subject", "participant", "participant_id"}:
            rename_map[c] = "Subject_ID"
        elif cl in {"y_true", "y", "target", "true", "y_test", "y_actual"}:  # <-- add y_test here
            rename_map[c] = "y_true"
        elif cl in {"y_pred", "pred", "prediction", "yhat", "y_hat"}:
            rename_map[c] = "y_pred"
        elif cl == "fold":
            rename_map[c] = "fold"

    df = df.rename(columns=rename_map)

    # if still missing, try explicit target name
    if "y_true" not in df.columns and target_col is not None and target_col in df.columns:
        df = df.rename(columns={target_col: "y_true"})

    if "Subject_ID" in df.columns:
        df = df.drop_duplicates(subset=["Subject_ID"], keep="last")
    else:
        raise ValueError("OOF predictions must include a Subject_ID column (any typical variant).")

    need = {"Subject_ID", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"OOF predictions missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)}")

    return df[["Subject_ID", "y_true", "y_pred"]].reset_index(drop=True)



def bootstrap_summary_from_oof(
        oof_df: pd.DataFrame,
        group_cols=("model", "task"),
        n_boot=1000,
        ci=0.95,
        subject_set=None,
        random_state=42
):
    df = oof_df.copy()
    if subject_set is not None:
        df = df[df["Subject_ID"].isin(subject_set)].copy()

    required = {"Subject_ID", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OOF DF missing required columns: {missing}")

    rng = np.random.RandomState(random_state)
    alpha = 1 - ci

    boot_rows = []
    summ_rows = []

    for keys, g in df.groupby(list(group_cols), observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y = g["y_true"].to_numpy()
        yhat = g["y_pred"].to_numpy()
        n = len(y)

        if n < 2:
            r2_oof = np.nan
            boots = [np.nan] * n_boot
        else:
            r2_oof = r2_score(y, yhat)
            idxs = rng.randint(0, n, size=(n_boot, n))
            boots = [r2_score(y[idx], yhat[idx]) for idx in idxs]

        for b_idx, r2b in enumerate(boots):
            row = dict(zip(group_cols, keys))
            row.update({"bootstrap": b_idx, "r2": r2b})
            boot_rows.append(row)

        # add CI
        arr = np.array(boots, dtype=float)
        ci_low = np.nanquantile(arr, alpha / 2)
        ci_high = np.nanquantile(arr, 1 - alpha / 2)

        srow = dict(zip(group_cols, keys))
        srow.update({
            "r2_oof": r2_oof,
            "r2_ci_low": ci_low,
            "r2_ci_high": ci_high,
            "n_subjects": n,
            "n_boot": n_boot
        })
        summ_rows.append(srow)

    boot_df = pd.DataFrame(boot_rows)
    summ_df = pd.DataFrame(summ_rows)
    return boot_df, summ_df


def bootstrap_r2_diff_from_oof(oof_A, oof_B, n_boot=1000, random_state=42):
    # align on common subjects
    m = pd.merge(oof_A[["Subject_ID","y_true","y_pred"]],
                 oof_B[["Subject_ID","y_true","y_pred"]],
                 on="Subject_ID", suffixes=("_A","_B"))
    # y_true must match
    if not np.allclose(m["y_true_A"], m["y_true_B"], equal_nan=True):
        raise ValueError("y_true mismatch between models after aligning subjects.")
    y    = m["y_true_A"].to_numpy()
    yA   = m["y_pred_A"].to_numpy()
    yB   = m["y_pred_B"].to_numpy()
    n    = len(y)
    rng  = np.random.RandomState(random_state)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        r2A = r2_score(y[idx], yA[idx])
        r2B = r2_score(y[idx], yB[idx])
        diffs[b] = r2A - r2B
    # summary
    ci_low  = np.quantile(diffs, 0.025)
    ci_high = np.quantile(diffs, 0.975)
    mean_d  = diffs.mean()
    # two-sided bootstrap p-value (how often diff crosses 0)
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs, {"mean_diff": mean_d, "ci_low": ci_low, "ci_high": ci_high,
                   "p_boot": p_boot, "n_subjects": n, "n_boot": n_boot}

def compare_models_bootstrap(oof_preds, task, target, models=None, n_boot=500, random_state=42, preferred_order=None):
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
        diffs, summ = bootstrap_r2_diff_from_oof(oofA, oofB, n_boot=n_boot, random_state=random_state)
        rows.append({"task": task, "target": target, "model_a": a, "model_b": b, **summ})
    return pd.DataFrame(rows)

# plotting

def plot_bootstrap_violin(
    oof_preds: pd.DataFrame,
    target_list, # one or several scores; panels if >1
    x="model", # "model", "task", or "target"
    hue="task", # "task", "model", or "target"
    model=None, # optional filter: keep only full model
    task=None, # optional filter: keep only one task
    order_x=None,
    order_hue=None,
    n_boot=1000, ci=0.95,
    subject_set=None,
    save_path=None,
    filename_prefix="bootstrap_violin"
):

    if order_x is None:
        order_x = sorted(oof_preds[x].unique().tolist())
    if order_hue is None:
        order_hue = sorted(oof_preds[hue].unique().tolist())

    make_panels = len(target_list) > 1
    cols = len(target_list)
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 4), sharey=True)
    if cols == 1:
        axes = [axes]

    for ax, tgt in zip(axes, target_list):
        sub = oof_preds[oof_preds["target"] == tgt].copy()
        if model is not None:
            sub = sub[sub["model"] == model]
        if task is not None:
            sub = sub[sub["task"] == task]

        # compute bootstrap + OOF
        boot_df, summ_df = bootstrap_summary_from_oof(
            sub, group_cols=(x, hue), n_boot=n_boot, ci=ci, subject_set=subject_set
        )

        # order categories
        boot_df[x] = boot_df[x].astype(CategoricalDtype(categories=order_x, ordered=True))
        boot_df[hue] = boot_df[hue].astype(CategoricalDtype(categories=order_hue, ordered=True))
        summ_df[x] = summ_df[x].astype(CategoricalDtype(categories=order_x, ordered=True))
        summ_df[hue] = summ_df[hue].astype(CategoricalDtype(categories=order_hue, ordered=True))

        palette = dict(zip(order_hue, sns.color_palette("muted", n_colors=len(order_hue))))

        ax = plt.gca()
        vp = sns.violinplot(
            data=boot_df, x=x, y="r2", hue=hue,
            order=order_x, hue_order=order_hue,
            inner=None, cut=0, dodge=True, ax=ax, palette=palette,
            linewidth=0.8, width=0.9, saturation=1.0
        )

        for coll in vp.collections:
            try:
                coll.set_alpha(0.25)  # translucent so dots show inside
                coll.set_zorder(1)  # behind dots/errorbars
            except Exception:
                pass

        # bootstrap dots on top
        sns.stripplot(
            data=boot_df, x=x, y="r2", hue=hue,
            order=order_x, hue_order=order_hue,
            dodge=True, jitter=0.25, size=2.5, alpha=0.55,
            edgecolor=None, ax=ax, palette=palette, zorder=3
        )

        # remove duplicate legend
        if ax.legend_:
            ax.legend_.remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(order_hue)], labels[:len(order_hue)], title=hue.title(), loc="upper left")

        # zero line
        ax.axhline(0, color="#777", lw=0.8, ls="--", zorder=10)

        # place mean OOF R^2 + bootstrap CI
        offsets = np.linspace(-0.25, 0.25, len(order_hue))
        for _, row in summ_df.iterrows():
            xi = order_x.index(row[x])
            hi = order_hue.index(row[hue])
            xpos = xi + offsets[hi]
            # CI around OOF center
            ax.errorbar(x=xpos, y=row["r2_oof"],
                        yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                              [row["r2_ci_high"] - row["r2_oof"]]],
                        fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=6)
            ax.plot([xpos], [row["r2_oof"]], marker="o", markersize=4, color="black", zorder=7)

        # labels
        filt = []
        if model is not None: filt.append(f"model={model}")
        if task is not None:  filt.append(f"task={task}")
        filt_str = f"  ({', '.join(filt)})" if filt else ""
        ax.set_title(f"{tgt}{filt_str}")
        ax.set_xlabel(x.title())
        ax.set_ylabel("Bootstrapped R²")

        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path:
        tag = f"_{x}_by_{hue}"
        if len(target_list) == 1:
            tag += f"_{target_list[0]}"
        if model is not None:
            tag += f"_model-{model}"
        if task is not None:
            tag += f"_task-{task}"
        out = os.path.join(save_path, f"{filename_prefix}{tag}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


