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
from pandas.api.types import CategoricalDtype
from itertools import combinations

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from regression.model_evaluation_helpers import bootstrap_summary_from_oof, bootstrap_by_model, pairwise_score_tests_for_task, pairwise_model_tests_for_task_target, pairwise_task_tests_for_scores

matplotlib.rcParams['font.family'] = 'Arial'

# turn variable-names into "normal writing" for labels
def format_title(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()

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

# p-value text/stars
def p_to_stars(p):
    if p is None or np.isnan(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# prepare formatting of p-values for plots
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
    prefix = p_to_stars(p) if stars else ""
    return f"{prefix} ({core})" if prefix else core


### model/score comparison plots

# define color palettes for tasks and models
def color_palette(keys, base="crest", lo=0.0, hi=0.8):
    cmap = sns.color_palette(base, as_cmap=True)
    vals = np.linspace(lo, hi, len(keys))
    return dict(zip(keys, [cmap(v)[:3] for v in vals]))

# score comparison (full model, all tasks; p-values for picnicScene; scores as hues, tasks in x-axis)
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
    palette = color_palette(order_scores, base="crest", lo=0.0, hi=0.8)

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
        out = os.path.join(save_path, "score_comparison_boxplot_with_pvals.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()


def plot_bootstrap_score_boxplot_pvals_all_tasks(
    oof_preds,
    model="full",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    subject_set=None,
    order_scores=None,
    order_tasks=None,
    adjust="holm",
    alpha=0.05,
    save_path=None,
):
    """
    Score comparison boxplot (R² bootstrapped) with p-value brackets for pairwise score comparisons within each task.
    """

    # subset & order
    sub = oof_preds[oof_preds["model"] == model].copy()
    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)

    n_scores = len(order_scores)
    n_tasks  = len(order_tasks)

    # bootstrap summary
    boot_df, _ = bootstrap_summary_from_oof(
        sub,
        group_cols=("target", "task"),
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=subject_set,
    )
    boot_df = boot_df[
        boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)
    ].copy()
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    # palette & figure
    palette = color_palette(order_scores, base="crest", lo=0.0, hi=0.8)

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
        whis=(2.5, 97.5),
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=linew,
        ax=ax,
    )
    shrink_seaborn_boxes(ax, factor=0.82)
    ax.margins(x=0.04)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:n_scores],
        [format_title(l) for l in labels[:n_scores]],
        title="Score",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        borderaxespad=0.2,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels([format_title(t) for t in order_tasks], fontsize=10)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    # collect box centers
    boxes = ax.artists
    centers = {}
    if len(boxes) == n_tasks * n_scores:
        for ti, task in enumerate(order_tasks):
            for si, score in enumerate(order_scores):
                idx = ti * n_scores + si
                box = boxes[idx]
                centers[(task, score)] = box.get_x() + box.get_width() / 2.0
    else:
        # fallback if artists layout changes
        for ti, task in enumerate(order_tasks):
            base_x = ti
            for si, score in enumerate(order_scores):
                offset = (si - (n_scores - 1) / 2) * (0.6 / n_scores)
                centers[(task, score)] = base_x + offset

    # compute whisker tops (97.5th percentile) per (task, score)
    whisker_top = {}
    for (t, s), grp in boot_df.groupby(["task", "target"], observed=True):
        if len(grp) > 0:
            whisker_top[(t, s)] = grp["r2"].quantile(0.975)
        else:
            whisker_top[(t, s)] = np.nan

    # axis + bracket params
    axis_top = 0.37
    ax.set_ylim(0.0, axis_top)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0.0, axis_top + 1e-9, 0.1))
    ax.grid(True, axis="y", which="major",
            linestyle=":", linewidth=0.8,
            color="#9AA0A6", alpha=0.55)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    # small absolute offsets so brackets hug whiskers
    pad      = 0.01   # above whisker
    gap      = 0.02   # between stacked brackets
    height   = 0.004   # bracket height
    top_pad  = 0.003   # space below top axis
    y_max_allowed = axis_top - top_pad - height

    idx_score = {s: i for i, s in enumerate(order_scores)}

    # draw brackets per task between scores
    for task in order_tasks:
        test_df = pairwise_score_tests_for_task(
            oof_preds,
            task=task,
            model=model,
            targets=tuple(order_scores),
            n_boot=n_boot,
            adjust=adjust,
            random_state=random_state,
        )
        if test_df is None or len(test_df) == 0:
            continue

        rows = [r for r in test_df.itertuples(index=False) if r.p_adj < alpha]
        if not rows:
            continue

        # track current "used" height per score within this task
        levels = {}
        for s in order_scores:
            wt = whisker_top.get((task, s), np.nan)
            levels[s] = (wt + pad) if not np.isnan(wt) else 0.0

        # shorter brackets (closer scores) first
        rows.sort(key=lambda r: abs(idx_score[r.A] - idx_score[r.B]))

        for r in rows:
            s1, s2 = r.A, r.B
            if s1 not in order_scores or s2 not in order_scores:
                continue

            x1 = centers[(task, s1)]
            x2 = centers[(task, s2)]

            # base just above the higher of the two current levels
            y_base = max(levels[s1], levels[s2])

            # cap if space too small
            if y_base > y_max_allowed:
                y_base = y_max_allowed

            # draw bracket
            ax.plot(
                [x1, x1, x2, x2],
                [y_base, y_base + height, y_base + height, y_base],
                lw=linew, color="#333", clip_on=True,
            )
            # label
            ax.text(
                0.5 * (x1 + x2),
                y_base + height,
                format_p_label(
                    r.p_adj,
                    stars=True,
                    decimals=3,
                    drop_leading_zero=True,
                    use_lt_for_small=True,
                    n_boot=n_boot,
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )

            # update levels so next brackets stack above
            new_level = y_base + gap
            levels[s1] = max(levels[s1], new_level)
            levels[s2] = max(levels[s2], new_level)

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_boxplot_pvals_all_tasks.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()

def plot_bootstrap_task_boxplot_pvals_across_scores(
    oof_preds,
    model="full",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    subject_set=None,
    order_scores=None,
    order_tasks=None,
    adjust="holm",
    alpha=0.05,
    score_for_brackets=None,  # e.g. "PictureNamingScore" to show only that score
    save_path=None,
):
    """
    Task comparison boxplot (R² bootstrapped) with p-value brackets for pairwise task comparisons within each score.
    """

    # subset & order
    sub = oof_preds[oof_preds["model"] == model].copy()
    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task  = CategoricalDtype(categories=order_tasks,  ordered=True)

    n_scores = len(order_scores)
    n_tasks  = len(order_tasks)

    # bootstrap summary
    boot_df, _ = bootstrap_summary_from_oof(
        sub,
        group_cols=("target", "task"),
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=subject_set,
    )
    boot_df = boot_df[
        boot_df["target"].isin(order_scores) & boot_df["task"].isin(order_tasks)
    ].copy()
    boot_df["target"] = boot_df["target"].astype(cat_score)
    boot_df["task"]   = boot_df["task"].astype(cat_task)

    # palette & figure
    palette = color_palette(order_scores, base="crest", lo=0.0, hi=0.8)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    linew = 0.8

    sns.boxplot(
        data=boot_df, x="task", y="r2", hue="target",
        order=order_tasks, hue_order=order_scores,
        palette=palette, dodge=True, width=0.6,
        showcaps=True, showfliers=False, boxprops={"alpha": 1},
        whis=(2.5, 97.5),
        showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=linew, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=linew,
        ax=ax,
    )
    shrink_seaborn_boxes(ax, factor=0.82)
    ax.margins(x=0.04)

    # legend inside
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:n_scores],
        [format_title(l) for l in labels[:n_scores]],
        title="Score",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        borderaxespad=0.2,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels([format_title(t) for t in order_tasks], fontsize=10)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    # collect box centers
    boxes = ax.artists
    centers = {}
    if len(boxes) == n_tasks * n_scores:
        for ti, task in enumerate(order_tasks):
            for si, score in enumerate(order_scores):
                idx = ti * n_scores + si
                box = boxes[idx]
                centers[(task, score)] = box.get_x() + box.get_width() / 2.0
    else:
        # fallback: approximate centers if artists layout changes
        for ti, task in enumerate(order_tasks):
            base_x = ti
            for si, score in enumerate(order_scores):
                offset = (si - (n_scores - 1) / 2) * (0.6 / n_scores)
                centers[(task, score)] = base_x + offset

    # compute whisker tops (97.5th percentile) for each (task, score)
    whisker_top = {}
    for (t, s), grp in boot_df.groupby(["task", "target"], observed=True):
        if len(grp) > 0:
            whisker_top[(t, s)] = grp["r2"].quantile(0.975)
        else:
            whisker_top[(t, s)] = np.nan

    # axis (fixed)
    axis_top = 0.37
    ax.set_ylim(0.0, axis_top)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0.0, axis_top + 1e-9, 0.1))
    ax.grid(True, axis="y", which="major",
            linestyle=":", linewidth=0.8,
            color="#9AA0A6", alpha=0.55)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    # bracket layout: directly above whiskers
    pad     = 0.000005   # tiny distance above top whisker
    gap     = 0.016   # gap between stacked brackets
    height  = 0.004   # bracket height
    top_pad = 0.003   # min distance from axis top
    y_max_allowed = axis_top - top_pad - height

    # collect tests
    tests_df = pairwise_task_tests_for_scores(
        oof_preds,
        scores=tuple(order_scores),
        tasks=tuple(order_tasks),
        model=model,
        n_boot=n_boot,
        adjust=adjust,
        random_state=random_state,
    )
    if tests_df is None or len(tests_df) == 0:
        plt.tight_layout(rect=[0, 0, 0.86, 1])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out = os.path.join(save_path, "task_comparison_boxplot_with_pvals.png")
            plt.savefig(out, dpi=600, bbox_inches="tight")
        plt.show()
        return

    idx_task = {t: i for i, t in enumerate(order_tasks)}

    # draw brackets
    for score in order_scores:
        if score_for_brackets is not None and score != score_for_brackets:
            continue

        sub_tests = tests_df[tests_df["target"] == score]
        if len(sub_tests) == 0:
            continue

        rows = [r for r in sub_tests.itertuples(index=False) if r.p_adj < alpha]
        if not rows:
            continue

        # sort by task distance for cleaner layout (shorter brackets first)
        rows.sort(key=lambda r: abs(idx_task[r.A] - idx_task[r.B]))
        n = len(rows)

        # top whisker among all tasks for this score
        tops = [
            whisker_top.get((t, score), np.nan)
            for t in order_tasks
        ]
        if all(np.isnan(t) for t in tops):
            continue
        max_for_score = np.nanmax(tops)

        # start baseline just above the whiskers
        baseline = max_for_score + pad

        # if stacked brackets would exceed top limit, move stack down
        needed_top = baseline + (n - 1) * gap
        if needed_top > y_max_allowed:
            baseline = y_max_allowed - (n - 1) * gap
            # still never below whisker+pad
            if baseline < max_for_score + pad:
                baseline = max_for_score + pad

        for k, r in enumerate(rows):
            t1, t2 = r.A, r.B
            if t1 not in order_tasks or t2 not in order_tasks:
                continue

            x1 = centers[(t1, score)]
            x2 = centers[(t2, score)]
            y_base = baseline + k * gap

            ax.plot(
                [x1, x1, x2, x2],
                [y_base, y_base + height, y_base + height, y_base],
                lw=linew, color="#333", clip_on=True,
            )
            ax.text(
                0.5 * (x1 + x2),
                y_base + height,
                format_p_label(
                    r.p_adj,
                    stars=True,
                    decimals=3,
                    drop_leading_zero=True,
                    use_lt_for_small=True,
                    n_boot=n_boot,
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "task_comparison_boxplot_with_pvals.png")
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
    pal_models = color_palette(order_models, base="crest", lo=0.15, hi=0.85)
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

    palette = color_palette(order_tasks, base="crest", lo=0.0, hi=0.8)

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

