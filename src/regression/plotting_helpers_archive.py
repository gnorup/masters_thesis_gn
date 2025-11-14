# ** TO DO delete unused imports in this file **
import sys
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms as mtransforms
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
from itertools import combinations

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from regression.model_evaluation_helpers import bootstrap_summary_from_oof, bootstrap_by_model, pairwise_score_tests_for_task, pairwise_model_tests_for_task_target, pairwise_group_tests_for_task, _pairwise_group_level_tests_single_score

matplotlib.rcParams['font.family'] = 'Arial'

def format_title(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

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
        custom_legend = [Line2D([0], [0], color='gray', linestyle='--', label='Perfect Prediction')]
        plt.legend(handles=custom_legend, loc="upper left", fontsize=10, frameon=True)
        plt.text(x=0.05, y=0.90, s=f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=10)

        plt.xlabel("Actual Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Predicted Score", fontsize=12, fontweight='bold', labelpad=10)
        plt.title(f"Fold {fold}: Predicted vs. Actual {format_target_name}\n({model_type_display_name}, {format_task_name})", fontsize=14, fontweight='bold', pad=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plot_path = os.path.join(output_dir, f"fold{fold}_actual_vs_predicted_{target}.png")
        plt.savefig(plot_path, dpi=600)
        plt.close()

        print(f"plot: actual vs predicted scores for fold {fold} saved to {plot_path}")

### model comparison plots

# define color palette for tasks
def make_task_palette(order_tasks, base="crest", lo=0.0, hi=0.8):
    # only use segment of colormap (avoids very dark tones)
    cmap = sns.color_palette(base, as_cmap=True)
    vals = np.linspace(lo, hi, len(order_tasks))
    colors = [cmap(v)[:3] for v in vals]
    return dict(zip(order_tasks, colors))

def make_score_palette(order_scores, base="crest", lo=0.0, hi=0.8):
    cmap = sns.color_palette(base, as_cmap=True)
    vals = np.linspace(lo, hi, len(order_scores))
    colors = [cmap(v)[:3] for v in vals]
    return dict(zip(order_scores, colors))

# define color palette for models
def make_model_palette(order_models, base="crest", lo=0.15, hi=0.85):
    cmap = sns.color_palette(base, as_cmap=True)
    vals = np.linspace(lo, hi, len(order_models))
    return dict(zip(order_models, [cmap(v)[:3] for v in vals]))

# helper to leave space between boxplots (so that they're not touching)
def shrink_seaborn_boxes(ax, factor=0.82):
    # shrink box patches
    boxes = [c for c in ax.get_children() if isinstance(c, PathPatch)]
    for patch in boxes:
        path = patch.get_path()
        verts = path.vertices
        xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
        cx = 0.5 * (xmin + xmax)
        verts[:, 0] = (verts[:, 0] - cx) * factor + cx
        path.vertices = verts

    # shrink horizontal lines (medians & caps), keep whiskers vertical
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        # horizontal if y is constant and x varies
        if len(x) == 2 and np.isclose(y[0], y[1]) and not np.isclose(x[0], x[1]):
            cx = 0.5 * (x[0] + x[1])
            line.set_xdata((x - cx) * factor + cx)

# violin plot for score comparison with mean+CI (one model, all tasks)
def plot_bootstrap_score_comparison(
    oof_preds,
    model, # fixed model
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set=None,
    order_scores=None, # ["PictureNamingScore","SemanticFluencyScore","PhonemicFluencyScore"]
    order_tasks=None, # tasks as hue
    save_path = None
):

    # filter to model
    sub = oof_preds[oof_preds["model"] == model].copy()

    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    # bootstrap summaries grouped by score & task
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("target", "task"), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )

    boot_df = boot_df[boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)]
    summ_df = summ_df[summ_df["target"].isin(order_scores) & summ_df["task"].isin(order_tasks)]

    # categorical ordering
    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    for df in (boot_df, summ_df):
        df["target"] = df["target"].astype(cat_score)
        df["task"]   = df["task"].astype(cat_task)

    # palette
    palette = make_task_palette(order_tasks)

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # violins
    violin_width = 0.60
    sns.violinplot(
        data=boot_df, x="target", y="r2", hue="task",
        order=order_scores, hue_order=order_tasks,
        inner=None, cut=0, dodge=True, palette=palette,
        density_norm="width", bw_adjust=1.2,
        width=violin_width, linewidth=0.7, saturation=1.0, ax=ax
    )

    # put mean dot + bootstrap CI at the exact dodge centers
    n_hue = len(order_tasks)
    def center_x(x_index: int, hue_index: int) -> float:
        return x_index + violin_width * (hue_index - (n_hue - 1) / 2) / n_hue

    for _, row in summ_df.iterrows():
        xi = order_scores.index(row["target"])
        hi = order_tasks.index(row["task"])
        xpos = center_x(xi, hi)
        ax.errorbar(x=xpos, y=row["r2_oof"],
                    yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                          [row["r2_ci_high"] - row["r2_oof"]]],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2, zorder=7)

        ax.plot([xpos], [row["r2_oof"]],
                marker="o", markersize=3,
                markerfacecolor="black", markeredgecolor="black",
                markeredgewidth=0.8, zorder=8)

    # legend and style
    ax.axhline(0, color="#777", lw=0.8, alpha=0.7, ls="--", zorder=-10)
    ax.set_title("Score Comparison")
    ax.set_xlabel("Score", fontsize=10)
    ticks = np.arange(len(order_scores))
    ax.set_xticks(ticks, [format_title(s) for s in order_scores])
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:len(order_tasks)], labels[:len(order_tasks)], title="Task", loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.margins(x=0.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_violin.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# violin plot for model comparison with mean+CI (all tasks, one score)
def plot_bootstrap_model_violin(
        oof_preds,
        target,
        order_models,
        order_tasks,
        n_boot = 1000,
        ci = 0.95,
        random_state = 42,
        subject_set = None,
        drop_models = None,
        save_path = None
):

    sub = oof_preds[oof_preds["target"] == target].copy()

    # bootstrap
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("model", "task"), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )

    # filter
    boot_df = boot_df[boot_df["model"].isin(order_models)]
    summ_df = summ_df[summ_df["model"].isin(order_models)]

    # option to drop a model (like baseline)
    if drop_models:
        boot_df = boot_df[~boot_df["model"].isin(drop_models)]
        summ_df = summ_df[~summ_df["model"].isin(drop_models)]
        order_models = [m for m in order_models if m not in set(drop_models)]

    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    for df in (boot_df, summ_df):
        df["model"] = df["model"].astype(cat_model)
        df["task"]  = df["task"].astype(cat_task)

    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # violins
    violin_width = 0.60
    sns.violinplot(
        data=boot_df, x="model", y="r2", hue="task",
        order=order_models, hue_order=order_tasks,
        inner=None, cut=0, dodge=True, palette=palette,
        density_norm="width", bw_adjust=1.2,
        width=violin_width, linewidth=0.8, saturation=1.0, ax=ax
    )

    # compute exact dodge centers, equally spaced
    n_hue = len(order_tasks)
    def center_x(x_index, hue_index):
        return x_index + violin_width * (hue_index - (n_hue - 1) / 2) / n_hue

    # add mean dot + bootstrap CI at the centers
    for _, row in summ_df.iterrows():
        xi = order_models.index(row["model"])
        hi = order_tasks.index(row["task"])
        xpos = center_x(xi, hi)

        ax.errorbar(x=xpos, y=row["r2_oof"],
                    yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                          [row["r2_ci_high"] - row["r2_oof"]]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=7)

        ax.plot([xpos], [row["r2_oof"]],
                marker="o", markersize=3,
                markerfacecolor="black", markeredgecolor="black",
                markeredgewidth=0.8, zorder=8)

    # legend and style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(order_tasks)], labels[:len(order_tasks)],
              title="Task", loc="upper left")

    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(target)
    ax.set_xlabel("Model")
    ax.set_ylabel("R² (bootstrapped)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.margins(x=0.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{target}_violin.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# boxplot for model comparison (all tasks, one score)
def plot_bootstrap_model_box(
    oof_preds,
    target,
    order_models,
    order_tasks,
    n_boot = 1000,
    ci = 0.95,
    subject_set = None,
    save_path = None
):

    sub = oof_preds[oof_preds["target"] == target].copy()

    # compute bootstrap
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("model", "task"), n_boot=n_boot, ci=ci, subject_set=subject_set
    )

    # filter
    boot_df = boot_df[boot_df["model"].isin(order_models)]
    summ_df = summ_df[summ_df["model"].isin(order_models)]

    # order
    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    boot_df["model"] = boot_df["model"].astype(cat_model)
    boot_df["task"]  = boot_df["task"].astype(cat_task)

    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    linew = 0.8

    ax = sns.boxplot(
        data=boot_df, x="model", y="r2", hue="task",
        order=order_models, hue_order=order_tasks,
        palette=palette, dodge=True, width=0.6,
        showcaps=True, showfliers=False, boxprops={"alpha":1},
        whis=(2.5, 97.5),
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=linew,
        ax=ax
    )
    shrink_seaborn_boxes(ax, factor=0.82)  # for space between boxes
    ax.margins(x=0.04)

    # style
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:len(order_tasks)], labels[:len(order_tasks)],
        title="Task", loc="upper left",
        fontsize=9, title_fontsize=10,
        frameon=True, borderaxespad=0.2, labelspacing=0.3,
        handlelength=1.0, handletextpad=0.4
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(target)
    ax.set_xlabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#9AA0A6", alpha=0.55)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_visible(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{target}_boxplot.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# violin plot (with standard boxes inside) for model comparison (all tasks, one score)
def plot_bootstrap_model_violin_standard(
    oof_preds,
    target,
    order_models,
    order_tasks,
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set = None,
    save_path = None
):

    sub = oof_preds[oof_preds["target"] == target].copy()

    # compute bootstrap
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("model", "task"), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )

    # filter
    boot_df = boot_df[boot_df["model"].isin(order_models)]
    summ_df = summ_df[summ_df["model"].isin(order_models)]

    # ordering
    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    boot_df["model"] = boot_df["model"].astype(cat_model)
    boot_df["task"]  = boot_df["task"].astype(cat_task)

    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.violinplot(
        data=boot_df, x="model", y="r2", hue="task",
        order=order_models, hue_order=order_tasks,
        inner="box", cut=0, dodge=True,
        palette=palette, density_norm="width",
        bw_adjust=1.2, width=0.6, linewidth=0.6, ax=ax
    )

    # style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(order_tasks)], labels[:len(order_tasks)],
              title="Task", loc="upper left")

    ax.axhline(0, color="#777", lw=0.8, alpha=0.7, ls="--", zorder=-10)
    ax.set_title(target)
    ax.set_xlabel("Model")
    ax.set_ylabel("R² (bootstrapped)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{target}_violin_standard.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# boxplot for score comparison (one model, all tasks)
def plot_bootstrap_score_boxplot(
    oof_preds,
    model,
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set = None,
    order_scores = None,
    order_tasks = None,
    save_path = None
):
    # filter to model
    sub = oof_preds[oof_preds["model"] == model].copy()

    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    # bootstrap
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=("target", "task"), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )
    boot_df = boot_df[boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)]

    # ordering
    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=boot_df, x="target", y="r2", hue="task",
        order=order_scores, hue_order=order_tasks,
        palette=palette, dodge=True, width=0.6,
        showcaps=True, showfliers=False, boxprops={"alpha": 1}, ax=ax
    )

    shrink_seaborn_boxes(ax, factor=0.82)  # for space between boxes
    ax.margins(x=0.04)

    # style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(order_tasks)], labels[:len(order_tasks)],
              title="Task", loc="upper right")

    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_title("Score Comparison")
    ax.set_xlabel("Score", fontsize=12)
    ticks = np.arange(len(order_scores))
    ax.set_xticks(ticks, [format_title(s) for s in order_scores])
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_boxplot.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# violin plot (with standard boxes inside) for score comparison (all tasks, one model)
def plot_bootstrap_score_violin_standard(
    oof_preds,
    model,
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set = None,
    order_scores = None,
    order_tasks = None,
    save_path = None
):
    # filter to model
    sub = oof_preds[oof_preds["model"] == model].copy()

    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    # bootstrap
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=("target", "task"), n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )
    boot_df = boot_df[boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)]

    # ordering
    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=boot_df, x="target", y="r2", hue="task",
        order=order_scores, hue_order=order_tasks,
        inner="box", cut=0, dodge=True,
        palette=palette, density_norm="width",
        bw_adjust=1.2, width=0.6, linewidth=0.6, ax=ax
    )

    # style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(order_tasks)], labels[:len(order_tasks)],
              title="Task", loc="upper right")

    ax.axhline(0, color="#777", lw=0.8, alpha=0.7, ls="--", zorder=-10)
    ax.set_title("Score Comparison")
    ax.set_xlabel("Score", fontsize=12)
    ticks = np.arange(len(order_scores))
    ax.set_xticks(ticks, [format_title(s) for s in order_scores])
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_violin_std.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# plot mean bootstrapped R² with CIs for model comparison (one score, all tasks)
def plot_bootstrap_mean_ci_models(
    oof_preds,
    target,
    order_models,
    order_tasks,
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set = None,
    drop_models = None,
    save_path = None
):
    # subset to target
    sub = oof_preds[oof_preds["target"] == target].copy()

    # bootstrap summaries per (model & task)
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub, group_cols=("model", "task"),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )

    # optionally drop models (like baseline)
    if drop_models:
        boot_df = boot_df[~boot_df["model"].isin(drop_models)]
        summ_df = summ_df[~summ_df["model"].isin(drop_models)]
        order_models = [m for m in order_models if m not in set(drop_models)]

    # categorical ordering
    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    for df in (boot_df, summ_df):
        df["model"] = df["model"].astype(cat_model)
        df["task"]  = df["task"].astype(cat_task)

    # palette
    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # centers hue positions inside each x-bin
    n_hue = len(order_tasks)
    def center_x(x_index: int, hue_index: int) -> float:
        return x_index + 0.60 * (hue_index - (n_hue - 1) / 2) / n_hue

    # draw one errorbar+dot per row of summary
    for _, row in summ_df.iterrows():
        if np.isnan(row["r2_oof"]):
            continue
        xi = order_models.index(row["model"])
        hi = order_tasks.index(row["task"])
        xpos = center_x(xi, hi)
        color = palette[row["task"]]

        # CI whisker in the same color
        ax.errorbar(
            x=xpos, y=row["r2_oof"],
            yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                  [row["r2_ci_high"] - row["r2_oof"]]],
            fmt="none", ecolor=color, elinewidth=1.0, capsize=3, zorder=6
        )
        # mean dot in the same color
        ax.plot([xpos], [row["r2_oof"]],
                marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor="none", zorder=7)

        # mean label on the right of the dot
        ax.annotate(
            f"{row['r2_oof']:.2f}",
            xy=(xpos, row["r2_oof"]),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", color="black", fontsize=8, zorder=8
        )

    # style
    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title((format_title(target)))
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean Bootstrapped R² (95% CI)", fontsize=11)

    ticks = np.arange(len(order_models))
    ax.set_xticks(ticks)
    ax.set_xticklabels(order_models)
    ax.tick_params(axis="x", labelsize=9, pad=2)

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")
    ax.margins(x=0.04)

    handles = [Line2D([0], [0], marker='o', color='none',
                      markerfacecolor=palette[t], markeredgecolor='none',
                      markersize=5, label=t)
               for t in order_tasks]
    ax.legend(handles=handles, title="Task", loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = f"{target}_mean_ci.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()

# plot mean bootstrapped R² with CIs for score comparison (one model, all tasks)
def plot_bootstrap_mean_ci_scores(
    oof_preds,
    model,
    order_scores,
    order_tasks,
    n_boot = 1000,
    ci = 0.95,
    random_state = 42,
    subject_set = None,
    save_path = None
):

    # subset to this model
    sub = oof_preds[oof_preds["model"] == model].copy()

    # bootstrapping per score & task
    boot_df, summ_df = bootstrap_summary_from_oof(
        sub,
        group_cols=("target", "task"),
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=subject_set
    )
    boot_df = boot_df[boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)]
    summ_df = summ_df[summ_df["target"].isin(order_scores) & summ_df["task"].isin(order_tasks)]

    # categorical ordering
    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)
    for df in (boot_df, summ_df):
        df["target"] = df["target"].astype(cat_score)
        df["task"]   = df["task"].astype(cat_task)

    # palette
    palette = make_task_palette(order_tasks)

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))

    n_hue = len(order_tasks)
    def center_x(xi, hi):
        return xi + 0.6 * (hi - (n_hue - 1) / 2) / n_hue

    # points + CIs
    for _, row in summ_df.iterrows():
        if np.isnan(row["r2_oof"]):
            continue
        xi = order_scores.index(row["target"])
        hi = order_tasks.index(row["task"])
        xpos = center_x(xi, hi)
        color = palette[row["task"]]

        ax.errorbar(
            x=xpos, y=row["r2_oof"],
            yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                  [row["r2_ci_high"] - row["r2_oof"]]],
            fmt="none", ecolor=color, elinewidth=1.0, capsize=3, zorder=6
        )
        ax.plot([xpos], [row["r2_oof"]],
                marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor="none", zorder=7)

        ax.annotate(
            f"{row['r2_oof']:.2f}",
            xy=(xpos, row["r2_oof"]),
            xytext=(6,0), textcoords="offset points",
            ha="left", va="center", color="#333", fontsize=8, zorder=8
        )

    # axes & labels
    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title((format_title(model) if 'format_title' in globals() else model))
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Mean Bootstrapped R² (95% CI)", fontsize=11)

    ticks = np.arange(len(order_scores))
    ax.set_xticks(ticks)
    ax.set_xticklabels(order_scores)
    ax.tick_params(axis="x", labelsize=9, pad=2)

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(False, axis="x")

    # make labels fit
    last_center = center_x(len(order_scores) - 1, n_hue - 1)
    ax.set_xlim(-0.5 + 0.10, last_center + 0.30)

    handles = [Line2D([0], [0], marker='o', color='none',
                      markerfacecolor=palette[t], markeredgecolor='none',
                      markersize=5, label=t)
               for t in order_tasks]
    ax.legend(handles=handles, title="Task", loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = "score_comparison_mean_ci.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()


# model comparison for one task, one score only

# violin plot with mean+CIs for model comparison (one task, one score)
def plot_models_violin_single(
    oof_preds, target, task, order_models,
    drop_models=None, save_path=None, subject_set=None
):
    # bootstrap per model for this task & score
    boot_df, summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models, subject_set=subject_set
    )

    # colors per model
    pal_models = make_model_palette(order_models)

    # figure sizing & tighter category spacing
    n = len(order_models)
    fig_w = min(7.5, 1.15 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    ax.set_xlim(-0.25, n - 0.75)
    ax.margins(x=0.01)

    # violin (bootstrap distribution shape for each model)
    sns.violinplot(
        data=boot_df, x="model", y="r2",
        hue="model", dodge=False, legend=False,
        inner=None, cut=0, density_norm="width",
        width=0.60, bw_adjust=1.2, gap=0.02,
        palette=pal_models, linewidth=0.7, ax=ax
    )
    # slightly transparent for readable CI overlays
    for coll in ax.collections:
        coll.set_alpha(0.75)

    # overlay point estimate (mean OOF R²) + bootstrap CI whiskers
    for _, row in summ_df.iterrows():
        xi = order_models.index(row["model"])
        ax.errorbar(
            x=xi, y=row["r2_oof"],
            yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                  [row["r2_ci_high"] - row["r2_oof"]]],
            fmt="none", ecolor=pal_models[row["model"]],
            elinewidth=0.9, capsize=2, zorder=7
        )
        ax.plot([xi], [row["r2_oof"]],
                marker="o", markersize=4.5,
                markerfacecolor=pal_models[row["model"]],
                markeredgecolor=pal_models[row["model"]],
                zorder=8)

    # style
    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(f"{format_title(task)} — {format_title(target)}")
    ticks = np.arange(len(order_models))
    ax.set_xticks(ticks); ax.set_xticklabels(order_models)
    ax.set_xlabel("Model"); ax.set_ylabel("Bootstrapped R²")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35); ax.grid(False, axis="x")

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = f"{task}_{target}_violin.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()

# violin plot with standard boxes inside (one task, one score)
def plot_models_violin_standard_single(
    oof_preds, target, task, order_models,
    drop_models=None, save_path=None, subject_set=None
):

    # bootstrap per model for this task & score
    boot_df, _summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models,
        subject_set=subject_set
    )

    # consistent colors per model
    pal_models = make_model_palette(order_models)

    # compact sizing and tighter category spacing
    n = len(order_models)
    fig_w = min(7.5, 1.15 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    ax.set_xlim(-0.25, n - 0.75)
    ax.margins(x=0.01)

    # violin with built-in box (median/IQR) drawn from bootstrap distribution
    sns.violinplot(
        data=boot_df, x="model", y="r2",
        hue="model", dodge=False, legend=False,
        inner="box", cut=0, density_norm="width",
        width=0.6, bw_adjust=1.2, gap=0.02,
        palette=pal_models, linewidth=0.6, ax=ax
    )
    ax.margins(x=0.03)

    # style
    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(f"{format_title(task)} — {format_title(target)}")
    ax.set_xlabel("Model"); ax.set_ylabel("Bootstrapped R²")
    ticks = np.arange(len(order_models))
    ax.set_xticks(ticks); ax.set_xticklabels(order_models)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    plt.tight_layout()
    if save_path:
        out = f"{task}_{target}_standard_violin.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()


# boxplot for model comparison (one task, one score)
def plot_models_box_single(
    oof_preds, target, task, order_models,
    drop_models=None, save_path=None, subject_set=None
):

    # bootstrap per model for this task & score
    boot_df, _summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models,
       subject_set=subject_set
    )

    # consistent colors per model
    pal_models = make_model_palette(order_models)

    # compact sizing and tighter category spacing
    n = len(order_models)
    fig_w = min(7.5, 1.15 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.25, n - 0.75)
    ax.margins(x=0.01)

    # boxplot over the bootstrap distribution
    sns.boxplot(
        data=boot_df, x="model", y="r2",
        hue="model", dodge=False, legend=False,
        palette=pal_models,
        width=0.3, gap=0.02,
        showcaps=True, showfliers=False,
        boxprops={"alpha": 1},
        whis=(2.5, 97.5),
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=0.6, linestyle="-"),
        medianprops=dict(color="#none", linewidth=0),
        linewidth=0.6, ax=ax
    )
    ax.margins(x=0.03)

    # style
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(f"{format_title(task)} — {format_title(target)}")
    ax.set_xlabel("Model"); ax.set_ylabel("Bootstrapped R²")
    ticks = np.arange(len(order_models))
    ax.set_xticks(ticks); ax.set_xticklabels(order_models)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_visible(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    if save_path:
        out = f"{task}_{target}_boxplot.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()

# mean & CI for model comparison (one task, one score)
def plot_models_mean_ci_single(
    oof_preds, target, task, order_models,
    drop_models=None, save_path=None, subject_set=None
):
    # bootstrap per model (distribution + summary)
    boot_df, summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models,
        subject_set=subject_set
    )

    # colors per model
    pal_models = make_model_palette(order_models)

    # figure sizing & tighter spacing
    n = len(order_models)
    fig_w = min(7.0, 1.0 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    ax.set_xlim(-0.25, n - 0.75)
    ax.margins(x=0.01)

    # one CI whisker + mean dot per model
    for _, row in summ_df.iterrows():
        m = row["model"]; color = pal_models[m]; xi = order_models.index(m)

        ax.errorbar(
            x=xi, y=row["r2_oof"],
            yerr=[[row["r2_oof"] - row["r2_ci_low"]],
                  [row["r2_ci_high"] - row["r2_oof"]]],
            fmt="none", ecolor=color, elinewidth=1.0, capsize=3, zorder=6
        )
        ax.plot([xi], [row["r2_oof"]],
                marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor="none", zorder=7)

        # add label next to the dot (last label nudged inward)
        dx = 4 if xi < len(order_models) - 1 else -4
        ha = "left" if dx > 0 else "right"
        ax.annotate(f"{row['r2_oof']:.2f}",
                    xy=(xi, row["r2_oof"]), xytext=(dx, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=9, color=color, zorder=8)

    # style
    ax.axhline(0, color="#777", lw=0.8, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(f"{format_title(task)} — {format_title(target)}")
    ticks = np.arange(len(order_models))
    ax.set_xticks(ticks); ax.set_xticklabels(order_models)
    ax.set_xlabel("Model"); ax.set_ylabel("Mean Bootstrapped R² (95% CI)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35); ax.grid(False, axis="x")
    ax.margins(x=0.05)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = f"{task}_{target}_mean_ci.png"
        plt.savefig(os.path.join(save_path, out), dpi=600, bbox_inches="tight")
    plt.show()


### with p-values

def format_p_label(p, stars=True, decimals=3,
                         drop_leading_zero=True, use_lt_for_small=True,
                         n_boot=None):
    if p is None or not np.isfinite(p):
        core = "p = n/a"
    else:
        thresh = 10 ** (-decimals)
        if use_lt_for_small and (p == 0 or p < thresh):
            val = f"< {thresh:.{decimals}f}"
        else:
            val = f"= {p:.{decimals}f}"
        if drop_leading_zero:
            val = val.replace("0.", ".").replace("< 0.", "< .")
        core = f"p {val}"
    prefix = _p_to_stars(p) if stars else ""
    return f"{prefix} ({core})" if prefix else core

# p-value text/stars
def _p_to_stars(p):
    if p is None or np.isnan(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# boxplot score comparison (full model, all tasks; p-values for picnicScene)
def plot_bootstrap_score_boxplot_pvals(
    oof_preds,
    model="full",
    task_to_annotate="picnicScene",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    subject_set=None,
    order_scores=None,
    order_tasks=None,
    adjust="holm",
    save_path=None
):
    # subset & order
    sub = oof_preds[oof_preds["model"] == model].copy()
    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)

    n_scores  = len(order_scores)
    n_tasks   = len(order_tasks)

    # bootstrap summary
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=("target", "task"),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )
    boot_df = boot_df[
        boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)
    ].copy()
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    # palette
    palette = make_score_palette(order_scores)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    linew = 0.8

    # boxplots
    sns.boxplot(
        data=boot_df, x="task", y="r2", hue="target",
        order=order_tasks, hue_order=order_scores,
        palette=palette, dodge=True, width=0.6,
        showcaps=True, showfliers=False, boxprops={"alpha": 1},
        whis=(2.5, 97.5),  # 95% bootstrap CI as whiskers
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=linew,
        ax=ax
    )
    shrink_seaborn_boxes(ax, factor=0.82)
    ax.margins(x=0.04)

    # legend and labels
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:n_scores], [format_title(l) for l in labels[:n_scores]],
        title="Score", loc="upper right", fontsize=9, title_fontsize=10,
        frameon=True, borderaxespad=0.2, labelspacing=0.3,
        handlelength=1.0, handletextpad=0.4
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(np.arange(n_tasks), [format_title(t) for t in order_tasks])
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    # pairwise score tests within one task (picnicScene)
    test_df = pairwise_score_tests_for_task(
        oof_preds, task=task_to_annotate, model=model,
        targets=tuple(order_scores), n_boot=n_boot, adjust=adjust, random_state=random_state
    )

    # find the boxes for the chosen task's hues (scores)
    t_idx  = order_tasks.index(task_to_annotate)
    boxes  = ax.artists  # one per (task, score) combo

    if len(boxes) != n_tasks * n_scores:
        x_centers = (
            t_idx +
            (np.arange(n_scores) - (n_scores - 1) / 2) * (0.6 / n_scores)
        )
    else:
        x_centers = np.array(
            [
                boxes[t_idx * n_scores + s_idx].get_x() +
                boxes[t_idx * n_scores + s_idx].get_width() / 2.0
                for s_idx in range(n_scores)
            ],
            dtype=float
        )

    # y baselines for that task, per score
    y_max_by_score = (
        boot_df[boot_df["task"] == task_to_annotate]
        .groupby("target", observed=True)["r2"].max()
        .reindex(order_scores).to_numpy()
    )

    # bracket placement
    axis_top = 0.39
    y_pad    = 0.006 * axis_top
    height   = 0.008 * axis_top
    rung_gap = 1.8  * height

    idx = {s: i for i, s in enumerate(order_scores)}
    rows = list(test_df.itertuples(index=False))

    anchor_for_pair = {}
    if len(order_scores) == 3:
        try:
            pn, sf, pf = order_scores
            y_top   = 0.38    # PictureNaming - PhonemicFluency
            y_short = 0.36   # PictureNaming - SemanticFluency
            anchor_for_pair = {
                frozenset({pn, pf}): y_top   - height,
                frozenset({pn, sf}): y_short - height,
            }
        except Exception:
            anchor_for_pair = {}

    # draw anchored pairs first, then others by proximity
    rows.sort(key=lambda r: (frozenset({r.A, r.B}) in anchor_for_pair, abs(idx[r.A] - idx[r.B])))

    for r in rows:
        i, j = idx[r.A], idx[r.B]
        pair = frozenset({r.A, r.B})

        if pair in anchor_for_pair:
            y_base = anchor_for_pair[pair]
        else:
            y_base = max(y_max_by_score[i], y_max_by_score[j]) + y_pad

        # bracket
        ax.plot(
            [x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
            [y_base, y_base + height, y_base + height, y_base],
            lw=linew, color="#333"
        )

        # label
        ax.text(
            (x_centers[i] + x_centers[j]) / 2, y_base + height,
            format_p_label(
                r.p_adj, stars=True, decimals=3,
                drop_leading_zero=True, use_lt_for_small=True, n_boot=n_boot
            ),
            ha="center", va="bottom", fontsize=8
        )

        if pair not in anchor_for_pair:
            y_max_by_score[i] = y_base + rung_gap
            y_max_by_score[j] = y_base + rung_gap

    # style
    ax.set_ylim(0.0, axis_top)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0.0, axis_top + 1e-9, 0.1))
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#9AA0A6", alpha=0.55)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_boxplot_with_pvals_flipped.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# score comparison with p-values (flipped: tasks as hues, scores in x-axis)
def plot_bootstrap_score_boxplot_with_pvals(
    oof_preds,
    model="full",
    task_to_annotate="picnicScene",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    subject_set=None,
    order_scores=None,
    order_tasks=None,
    adjust="holm",
    save_path=None
):
    # subset & order
    sub = oof_preds[oof_preds["model"] == model].copy()
    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)

    n_scores  = len(order_scores)
    n_tasks   = len(order_tasks)

    # bootstrap summary
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=("target", "task"),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=subject_set
    )
    boot_df = boot_df[
        boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)
    ]
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    # set up figure and palette
    palette = make_task_palette(order_tasks)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    linew = 0.8 # line width

    # boxplot
    sns.boxplot(
        data=boot_df, x="target", y="r2", hue="task",
        order=order_scores, hue_order=order_tasks,
        palette=palette, dodge=True, width=0.6,
        showcaps=True, showfliers=False, boxprops={"alpha": 1},
        whis=(2.5, 97.5), # whiskers = 95% bootstrap CI
        showmeans=True, meanline=True, # show mean
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0), # hide median
        linewidth=linew,
        ax=ax
    )
    shrink_seaborn_boxes(ax, factor=0.82)
    ax.margins(x=0.04)

    # legend and labels
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:n_tasks], labels[:n_tasks],
        title="Task", loc="upper right", fontsize=9, title_fontsize=10,
        frameon=True, borderaxespad=0.2, labelspacing=0.3,
        handlelength=1.0, handletextpad=0.4
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.set_xlabel("Score", fontsize=12)
    ax.set_xticks(np.arange(n_scores), [format_title(s) for s in order_scores])
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    # pairwise tests for the chosen task & bracket x-positions
    test_df = pairwise_score_tests_for_task(
        oof_preds, task=task_to_annotate, model=model,
        targets=tuple(order_scores), n_boot=n_boot, adjust=adjust, random_state=random_state
    )

    hue_idx   = order_tasks.index(task_to_annotate)
    boxes     = ax.artists

    if len(boxes) != n_scores * n_tasks:
        x_centers = np.arange(n_scores) + (hue_idx - (n_tasks - 1) / 2) * (0.6 / n_tasks)
    else:
        x_centers = np.array(
            [
                boxes[i * n_tasks + hue_idx].get_x() + boxes[i * n_tasks + hue_idx].get_width() / 2.0
                for i in range(n_scores)
            ],
            dtype=float
        )

    idx = {s: i for i, s in enumerate(order_scores)}
    y_max_by_score = (
        boot_df[boot_df["task"] == task_to_annotate]
        .groupby("target", observed=True)["r2"].max()
        .reindex(order_scores).to_numpy()
    )

    # bracket placement
    axis_top = 0.39
    y_pad = 0.006 * axis_top
    height = 0.008 * axis_top
    rung_gap = 1.8 * height

    y_top = 0.38     # PictureNaming - PhonemicFluency
    y_short = 0.365  # PictureNaming - SemanticFluency

    pn, sf, pf = order_scores
    anchor_for_pair = {
        frozenset({pn, pf}): y_top - height,
        frozenset({pn, sf}): y_short - height,
    }

    # draw brackets in controlled order
    rows = list(test_df.itertuples(index=False))
    rows.sort(
        key=lambda r: (frozenset({r.A, r.B}) in anchor_for_pair, abs(idx[r.A] - idx[r.B]))
    )

    for r in rows:
        i, j = idx[r.A], idx[r.B]
        pair = frozenset({r.A, r.B})

        # anchored pairs use fixed y; others use data-driven baseline + padding
        if pair in anchor_for_pair:
            y_base = anchor_for_pair[pair]
        else:
            y_base = max(y_max_by_score[i], y_max_by_score[j]) + y_pad

        # bracket
        ax.plot(
            [x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
            [y_base, y_base + height, y_base + height, y_base],
            lw=linew, color="#333"
        )

        # label
        ax.text(
            (x_centers[i] + x_centers[j]) / 2, y_base + height,
            format_p_label(
                r.p_adj, stars=True, decimals=3,
                drop_leading_zero=True, use_lt_for_small=True, n_boot=n_boot
            ),
            ha="center", va="bottom", fontsize=8
        )

        # only bump spacing for unanchored pairs
        if pair not in anchor_for_pair:
            y_max_by_score[i] = y_base + rung_gap
            y_max_by_score[j] = y_base + rung_gap

    # final axis range, grid, spines, zero line
    ax.set_ylim(0.0, axis_top)

    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0.0, axis_top + 1e-9, 0.1))
    ax.grid(
        True, axis="y", which="major",
        linestyle=":", linewidth=0.8, color="#9AA0A6", alpha=0.55
    )

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

   # layout & saving
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_boxplot_with_pvals.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

# model comparison for one task-score combination (with p-values)
def plot_models_box_with_pvals(
    oof_preds,
    target="SemanticFluencyScore",
    task="picnicScene",
    order_models=None,
    n_boot=1000,
    ci=0.95,
    random_state=42,
    subject_set=None,
    adjust="holm",
    save_path=None,
    pairs_to_show=(), # e.g. [("demographics","acoustic"), ("demographics","linguistic")]
    bracket_gap=0.05, # vertical gap between rows
    tests_df=None, # precomputed pairwise results
    pval_models=None # if no tests_df, use this full model list for Holm family
):

    # subset & order for boxes
    sub = oof_preds[(oof_preds["task"] == task) & (oof_preds["target"] == target)].copy()
    if subject_set is not None:
        sub = sub[sub["Subject_ID"].isin(subject_set)].copy()
    if order_models is None:
        order_models = sorted(sub["model"].unique().tolist())

    # bootstrap for boxes
    boot_df, _summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models=None,
        subject_set=subject_set, n_boot=n_boot, ci=ci, random_state=random_state
    )

    # plot
    pal_models = make_model_palette(order_models)
    n = len(order_models)
    fig_w = min(7.5, 1.15 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.25, n - 0.75)
    ax.margins(x=0.01)
    linew = 0.8

    sns.boxplot(
        data=boot_df, x="model", y="r2",
        hue="model", dodge=False, legend=False,
        palette=pal_models,
        width=0.34, gap=0.02,
        showcaps=True, showfliers=False, boxprops={"alpha": 1},
        whis=(2.5, 97.5),
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=linew, ax=ax
    )
    shrink_seaborn_boxes(ax, factor=0.82)
    ax.margins(x=0.03)

    # p-values: use precomputed table if provided; otherwise compute using a fixed full family
    if tests_df is not None:
        test_df = tests_df.copy()
        # keep only rows for this task+target
        if "task" in test_df.columns:
            test_df = test_df[test_df["task"] == task]
        if "target" in test_df.columns:
            test_df = test_df[test_df["target"] == target]
    else:
        if pval_models is None:
            pval_models = sorted(oof_preds.loc[
                (oof_preds["task"] == task) & (oof_preds["target"] == target), "model"
            ].unique().tolist())
        test_df = pairwise_model_tests_for_task_target(
            oof_preds, task=task, target=target,
            models=tuple(pval_models), drop_models=None,
            n_boot=n_boot, adjust=adjust, random_state=random_state
        )

    # keep only selected pairs
    if not pairs_to_show:
        df_anno = test_df.iloc[0:0].copy()
    else:
        want = {tuple(sorted(p)) for p in pairs_to_show}
        df_anno = test_df[test_df.apply(lambda r: tuple(sorted([r.A, r.B])) in want, axis=1)].copy()
        df_anno = df_anno[df_anno["A"].isin(order_models) & df_anno["B"].isin(order_models)]

    # bracket placement
    boxes = ax.artists
    if len(boxes) != n:
        x_centers = np.arange(n, dtype=float)
    else:
        x_centers = np.array([b.get_x() + b.get_width()/2.0 for b in boxes], dtype=float)
    idx = {m: i for i, m in enumerate(order_models)}

    rows = list(df_anno.itertuples(index=False))
    rows.sort(key=lambda r: -(abs(idx[r.A] - idx[r.B])))

    lanes, placed = [], []
    for r in rows:
        i, j = idx[r.A], idx[r.B]
        L, R = (min(i, j), max(i, j))
        for lane_i, lane in enumerate(lanes):
            if all(R <= L2 or L >= R2 for (L2, R2) in lane):
                lane.append((L, R)); placed.append((r, lane_i)); break
        else:
            lanes.append([(L, R)]); placed.append((r, len(lanes)-1))

    bracket_y_start = 0.99
    bracket_height  = 0.022
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for r, lane_i in placed:
        i, j = idx[r.A], idx[r.B]
        xi, xj = x_centers[i], x_centers[j]
        y_top = bracket_y_start - lane_i * bracket_gap
        y0 = y_top - bracket_height
        ax.plot([xi, xi, xj, xj], [y0, y_top, y_top, y0],
                lw=linew, color="#333", transform=trans, clip_on=False)
        ax.text((xi + xj) / 2.0, y_top,
                format_p_label(r.p_adj, stars=True, decimals=3,
                               drop_leading_zero=True, use_lt_for_small=True, n_boot=n_boot),
                ha="center", va="bottom", fontsize=8, transform=trans, clip_on=False)

    # style
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_xlabel("Model", fontsize=12); ax.set_ylabel("R² (bootstrapped)", fontsize=12)
    ax.set_xticks(np.arange(len(order_models))); ax.set_xticklabels(order_models)
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#9AA0A6", alpha=0.55)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc"); ax.spines[side].set_linewidth(0.5); ax.spines[side].set_visible(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{task}_{target}_model_boxplot_with_pvals.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()


### bias analyses

# put near your other small helpers
def format_bias_labels(group_col, levels):
    """Return display labels for x-ticks without changing the data order."""
    mapping = {
        "Gender_label": {"f": "Female", "m": "Male"},
        "Country_label": {"uk": "UK", "usa": "US"},
        # add more if you like, e.g.:
        # "AgeGroup": {"<65": "<65", "65–75": "65–75", ">75": ">75"},
    }
    m = mapping.get(group_col, {})
    return [m.get(l, str(l)) for l in levels]


def bias_palette(group_col, levels):
    """Return a dict level -> color for consistent, readable palettes."""
    # Your brand pair
    BLUE = "#6AA2FF"   # use for male + USA
    RED  = "#E07C7C"   # use for female + UK

    # Normalize to list
    levels = list(levels)

    if group_col == "Gender_label":
        # You said: male (blue), female (red)
        mapping = {"m": BLUE, "f": RED}
        return {lvl: mapping.get(lvl, BLUE) for lvl in levels}

    if group_col == "Country_label":
        # UK blue, USA red (parallel to gender for consistency)
        mapping = {"uk": RED, "usa": BLUE}
        return {lvl: mapping.get(lvl, BLUE) for lvl in levels}

    if group_col == "AgeGroup":
        # Prefer Blues (sequential). Darker = older.
        # If you later want the *lightest* on the right, use "Blues_r" instead.
        seq = sns.color_palette("Blues", n_colors=len(levels))
        return {lvl: seq[i] for i, lvl in enumerate(levels)}

    # Fallback: use your default tuple
    default = ("#6AA2FF","#E07C7C","#8EC07C","#E0A96D")
    return {lvl: (default[i % len(default)]) for i, lvl in enumerate(levels)}

# --- keep your existing imports at the top of the file ---
# uses: bootstrap_summary_from_oof, format_p_label, format_title,
#       _pairwise_group_level_tests_single_score

def plot_group_box_with_pvals(
    oof_preds,
    target,
    group_col,
    levels,
    task="picnicScene",
    model="full",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    adjust="holm",
    save_dir=None,
    pairs_to_show=None,
    box_width=0.24,             # tighter boxes
    bracket_gap=0.05,
    ax=None,
    figsize=(4.6, 2.9),
    show_legend=False,          # <- default OFF
    ymin=None, ymax=None,
    clip_quantiles=(0.02, 0.98),
    bracket_scale=1.0,          # bump for more bracket spacing
    ceil_pad=0.10               # NEW: extra headroom above highest bracket (axis fraction)
):
    levels = list(levels) if isinstance(levels, (list, tuple)) else [levels]
    assert len(levels) >= 2, "Provide at least two group levels."

    sub = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target) &
        (oof_preds[group_col].isin(levels))
    ].copy()
    if sub.empty:
        print("No data for the selected filters.")
        return None

    sub[group_col] = sub[group_col].astype(CategoricalDtype(categories=levels, ordered=True))

    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=(group_col,),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=None
    )
    boot_df = boot_df[boot_df[group_col].isin(levels)].copy()
    boot_df[group_col] = boot_df[group_col].astype(CategoricalDtype(categories=levels, ordered=True))

    pal = bias_palette(group_col, levels)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    sns.boxplot(
        data=boot_df, x=group_col, y="r2",
        order=levels, hue=group_col, hue_order=levels, legend=show_legend,
        palette=pal, dodge=False, width=box_width,
        showcaps=True, showfliers=False, boxprops={"alpha": 1, "edgecolor": "black"},
        whiskerprops={"alpha": 1, "color": "black"}, capprops={"alpha": 1, "color": "black"},
        whis=(2.5, 97.5), showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=0.9, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=0.8, ax=ax
    )

    # tighter between boxes, more space at edges
    ax.margins(x=0.12)

    ticks = np.arange(len(levels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(format_bias_labels(group_col, levels))
    ax.set_xlabel(format_title(group_col.replace("_label","")), fontsize=12)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#E0E0E0", alpha=0.4)
    for side in ("left","bottom"):
        ax.spines[side].set_color("#cccccc"); ax.spines[side].set_linewidth(0.5)
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    tests_rows = _pairwise_group_level_tests_single_score(
        oof_preds, group_col, levels, task, target, model=model,
        n_boot=n_boot, adjust=adjust, random_state=random_state, pairs_to_test=pairs_to_show
    )

    boxes = ax.artists
    x_centers = (np.arange(len(levels), dtype=float) if len(boxes) != len(levels)
                 else np.array([b.get_x() + b.get_width()/2.0 for b in boxes], dtype=float))
    idx = {lvl: i for i, lvl in enumerate(levels)}

    # y-lims
    yvals = boot_df["r2"].to_numpy()
    if ymin is None or ymax is None:
        q_lo = np.nanquantile(yvals, clip_quantiles[0])
        q_hi = np.nanquantile(yvals, clip_quantiles[1])
        yr   = max(1e-6, q_hi - q_lo)
        auto_bottom = min(-0.02, q_lo - 0.04*yr)
        auto_top    = max(0.39, q_hi + 0.06*yr)
        axis_bottom = ymin if ymin is not None else auto_bottom
        axis_top    = ymax if ymax is not None else auto_top
    else:
        axis_bottom, axis_top = ymin, ymax

    ax.set_ylim(axis_bottom, axis_top)
    axis_range = axis_top - axis_bottom
    height  = 0.012 * axis_range
    y_pad   = (0.008 * bracket_scale) * axis_range
    rung_gap = (1.8  * bracket_scale) * height

    ymax_by_level = (
        boot_df.groupby(group_col, observed=True)["r2"]
               .max().reindex(levels).to_numpy()
    )

    for r in sorted(tests_rows, key=lambda r: abs(idx[r["A"]] - idx[r["B"]])):
        A, B, p_adj = r["A"], r["B"], r["p_adj"]
        i, j = idx[A], idx[B]
        y_base = max(ymax_by_level[i], ymax_by_level[j]) + y_pad
        ax.plot([x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
                [y_base, y_base + height, y_base + height, y_base],
                lw=0.8, color="#333")
        ax.text((x_centers[i] + x_centers[j]) / 2.0, y_base + height,
                format_p_label(p_adj, stars=True, decimals=3,
                               drop_leading_zero=True, use_lt_for_small=True, n_boot=n_boot),
                ha="center", va="bottom", fontsize=8)
        ymax_by_level[i] = y_base + rung_gap
        ymax_by_level[j] = y_base + rung_gap

    # lift the ceiling a touch so the top bracket breathes
    cur_bot, cur_top = ax.get_ylim()
    peak = float(np.nanmax(ymax_by_level)) if len(ymax_by_level) else cur_top
    maybe_top = max(cur_top, peak + ceil_pad * (cur_top - cur_bot))
    if maybe_top > cur_top:
        ax.set_ylim(cur_bot, maybe_top)

    plt.tight_layout()
    if save_dir and created_fig:
        os.makedirs(save_dir, exist_ok=True)
        safe_levels = "_".join(str(l) for l in levels)
        out = os.path.join(save_dir, f"group_box_single__{task}__{model}__{target}__{safe_levels}.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")

    if created_fig:
        plt.show()
    return ax



def plot_bias_panels_ABC(
    oof_preds,
    target="SemanticFluencyScore",
    task="picnicScene",
    model="full",
    save_path=None,
    c_ylims=(-0.15, 0.35),   # fixed axis for C (Age)
    bracket_scale_c=1.9
):
    """A: Gender and B: Country share a common y-axis; C: Age has its own."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=False)

    # --- compute a robust shared y-range from both panels A & B ---
    def _shared_ab_ylims():
        def _get(coln, lvls):
            sub = oof_preds[
                (oof_preds["task"]==task) & (oof_preds["model"]==model) &
                (oof_preds["target"]==target) & (oof_preds[coln].isin(lvls))
            ].copy()
            if sub.empty:
                return (0.0, 0.39)
            boot, _ = bootstrap_summary_from_oof(
                sub, group_cols=(coln,),
                n_boot=500, ci=0.95, random_state=42, subject_set=None
            )
            y = boot["r2"].to_numpy()
            ql, qh = np.nanquantile(y, 0.02), np.nanquantile(y, 0.98)
            rng = max(1e-6, qh-ql)
            return (min(-0.02, ql - 0.04*rng), max(0.39, qh + 0.06*rng))

        yA = _get("Gender_label", ("f","m"))
        yB = _get("Country_label", ("uk","usa"))
        return (min(yA[0], yB[0]), max(yA[1], yB[1]))

    ymin_AB, ymax_AB = _shared_ab_ylims()

    # --- A: Gender (share y via explicit limits) ---
    plot_group_box_with_pvals(
        oof_preds, target, "Gender_label", ("f","m"),
        task=task, model=model, ax=axes[0], show_legend=False,
        figsize=(4.6, 2.9),
        box_width=0.24, ymin=ymin_AB, ymax=ymax_AB,
        bracket_scale=1.2, ceil_pad=0.10
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # --- B: Country (apply the exact same y-lims) ---
    plot_group_box_with_pvals(
        oof_preds, target, "Country_label", ("uk","usa"),
        task=task, model=model, ax=axes[1], show_legend=False,
        figsize=(4.6, 2.9),
        box_width=0.24, ymin=ymin_AB, ymax=ymax_AB,
        bracket_scale=1.2, ceil_pad=0.10
    )
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # --- lock A & B to the exact same y-axis AFTER plotting ---
    yA_lo, yA_hi = axes[0].get_ylim()
    yB_lo, yB_hi = axes[1].get_ylim()
    shared_lo = min(yA_lo, yB_lo)
    shared_hi = max(yA_hi, yB_hi)
    axes[0].set_ylim(shared_lo, shared_hi)
    axes[1].set_ylim(shared_lo, shared_hi)

    # --- C: Age group (keep its own axis) ---
    plot_group_box_with_pvals(
        oof_preds, target, "AgeGroup", ("<65","65–75",">75"),
        task=task, model=model, ax=axes[2], show_legend=False,
        figsize=(4.6, 2.9),
        pairs_to_show=[("<65","65–75"), ("65–75",">75"), ("<65",">75")],
        box_width=0.24, ymin=c_ylims[0], ymax=c_ylims[1],
        bracket_scale=bracket_scale_c, ceil_pad=0.12
    )
    axes[2].text(0.02, 0.98, "C", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return fig, axes


