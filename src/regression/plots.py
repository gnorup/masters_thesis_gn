import re
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator
import matplotlib.transforms as mtransforms
from matplotlib.patches import PathPatch
from pandas.api.types import CategoricalDtype

from config.constants import RANDOM_STATE, N_BOOT, CI, ALPHA
from data_preparation.data_handling import subset_in_order, make_age_binary_groups

from regression.model_evaluation import (
    bootstrap_r2_summary, bootstrap_by_model,
    compare_scores, compare_models,
    compare_tasks, compare_tasks_for_one_score,
    metric_label, bootstrap_metric_unpaired, compare_subgroups
)

matplotlib.rcParams['font.family'] = 'Arial'


### formatting

def format_title(name):
    """
    Turn variable-names into "normal writing" for labels
    """
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)  # insert space before capitals
    return name.title().strip()


def format_model_label(name):
    """
    Labels for model names
    """
    label_map = {
        "full": "full",
        "linguistic+acoustic": "linguistic + acoustic",
        "linguistic": "linguistic",
        "acoustic": "acoustic",
        "demographics": "demographic",
    }
    return label_map.get(name, name)


def p_to_stars(p):
    """
    p-value text/stars
    """
    if p is None or np.isnan(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def format_p_label(p, stars=True, decimals=3):
    """
    Format labels for p-values in plots
    """
    if p is None or not np.isfinite(p):
        core = "p = n/a"
    else:
        # threshold of decimals shown
        thresh = 10 ** (-decimals)
        # format number to given threshold
        if (p == 0 or p < thresh):
            val = f"< {thresh:.{decimals}f}"
        else:
            val = f"= {p:.{decimals}f}"
        # format label
        val = val.replace("0.", ".").replace("< 0.", "< .")
        core = f"p {val}"
    prefix = p_to_stars(p) if stars else ""
    return f"{prefix} ({core})" if prefix else core


### color palettes

def color_palette(keys, base="crest", lo=0.0, hi=0.8):
    """
    Define color palette for main regression analysis
    """
    cmap = sns.color_palette(base, as_cmap=True)
    vals = np.linspace(lo, hi, len(keys))
    return dict(zip(keys, [cmap(v)[:3] for v in vals]))


def bias_palette(group_col, levels):
    """
    Define color palette for demographic subgroup-comparisons
    """
    # for country and gender
    blue = "#629ccc"  # for male + US
    red = "#E07C7C"  # for female + UK

    levels = list(levels)

    if group_col == "Gender_label":
        mapping = {"m": blue, "f": red}
        return {lvl: mapping.get(lvl, blue) for lvl in levels}

    if group_col == "Country_label":
        mapping = {"uk": red, "usa": blue}
        return {lvl: mapping.get(lvl, blue) for lvl in levels}

    # different blues for Age (3 or 2 groups) (darker with older age)
    if group_col in ("AgeGroup", "AgeGroup2"):
        age_palette = [
                          "#cbdeee",  # pastel blue (youngest)
                          "#306694",  # cobalt (oldest)
                      ][:len(levels)]
        return {lvl: age_palette[i] for i, lvl in enumerate(levels)}

    if group_col == "CountryGender_label":
        mapping = {
            "uk_f": "#E07C7C",  # UK Female (red)
            "uk_m": "#C95F5F",  # UK Male (darker red)
            "usa_f": "#8DB8FF",  # US Female (lighter blue)
            "usa_m": "#6AA2FF",  # US Male (blue)
        }
        return {lvl: mapping.get(lvl, "#6AA2FF") for lvl in levels}

    # fallback
    seq = sns.color_palette("Blues", n_colors=len(levels))
    return {lvl: seq[i] for i, lvl in enumerate(levels)}


### plotting helpers

def shrink_seaborn_boxes(ax, factor=0.82):
    """
    Leave space between boxplots so that they are not touching
    """
    # shrink box patches
    boxes = [c for c in ax.get_children() if isinstance(c, PathPatch)]
    for patch in boxes:
        path = patch.get_path()
        verts = path.vertices
        xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
        center_x = 0.5 * (xmin + xmax)
        verts[:, 0] = (verts[:, 0] - center_x) * factor + center_x
        path.vertices = verts

    # shrink horizontal lines (medians & caps), keep whiskers vertical
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        # horizontal if y is constant and x varies
        if len(x) == 2 and np.isclose(y[0], y[1]) and not np.isclose(x[0], x[1]):
            center_x = 0.5 * (x[0] + x[1])
            line.set_xdata((x - center_x) * factor + center_x)


def prepare_bootstrapped_r2_by_task_score(
        oof_preds,
        model,
        order_scores=None,
        order_tasks=None,
        n_boot=N_BOOT,
        ci=CI,
        random_state=RANDOM_STATE,
        subject_set=None,
):
    """
    Subset for selected model, determine score/task order, and bootstrap R² per score & task
    """
    sub = oof_preds[oof_preds["model"] == model].copy()
    if order_scores is None:
        order_scores = sorted(sub["target"].unique().tolist())
    if order_tasks is None:
        order_tasks = sorted(sub["task"].unique().tolist())

    cat_score = CategoricalDtype(categories=order_scores, ordered=True)
    cat_task = CategoricalDtype(categories=order_tasks, ordered=True)

    boot_df, _ = bootstrap_r2_summary(
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
    boot_df["task"] = boot_df["task"].astype(cat_task)

    return boot_df, order_scores, order_tasks


def compute_box_centers(ax, order_tasks, order_scores, width=0.6):
    """
    Compute x-centers of boxes for task, score combinations -> for p-value brackets
    """
    boxes = ax.artists
    centers = {}
    n_tasks = len(order_tasks)
    n_scores = len(order_scores)

    if len(boxes) == n_tasks * n_scores:
        for task_index, task in enumerate(order_tasks):
            for score_index, score in enumerate(order_scores):
                artist_index = task_index * n_scores + score_index
                box = boxes[artist_index]
                centers[(task, score)] = box.get_x() + box.get_width() / 2.0
    else:
        # fallback: approximate centers if artists layout changes
        for task_index, task in enumerate(order_tasks):
            base_x = task_index
            for score_index, score in enumerate(order_scores):
                offset = (score_index - (n_scores - 1) / 2) * (width / n_scores)
                centers[(task, score)] = base_x + offset
    return centers


def compute_whisker_tops(boot_df, value_col="r2", group_cols=("task", "target"), q=0.975):
    """
    Compute upper whisker (e.g., 97.5th percentile) per group -> for placement of brackets
    """
    whisker_top = {}
    for group_key, grp in boot_df.groupby(list(group_cols), observed=True):
        if len(grp) > 0:
            whisker_top[group_key] = grp[value_col].quantile(q)
        else:
            whisker_top[group_key] = np.nan
    return whisker_top


def style_r2_axis(ax, axis_top=None, tick_step=0.1, show_zero_line=True):
    """
    Consistent styling for R² axes (grid, spines, ticks, zero-line).
    """
    if axis_top is not None:
        ax.set_ylim(0.0, axis_top)
        ax.set_yticks(np.arange(0.0, axis_top + 1e-9, tick_step))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major",
            linestyle=":", linewidth=0.8,
            color="#9AA0A6", alpha=0.55)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if show_zero_line:
        ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)


### main comparison plots

def plot_score_comparison_boxplots_bootstrapped(
        oof_preds,
        model="full",
        n_boot=N_BOOT,
        ci=CI,
        random_state=RANDOM_STATE,
        subject_set=None,
        order_scores=None,
        order_tasks=None,
        adjust="holm",
        alpha=ALPHA,
        save_path=None,
):
    """
    Score comparison boxplots (R² bootstrapped) with p-values for pairwise score comparisons within each task
    """

    # subset data for selected model and set score/task order
    boot_df, order_scores, order_tasks = prepare_bootstrapped_r2_by_task_score(
        oof_preds=oof_preds,
        model=model,
        order_scores=order_scores,
        order_tasks=order_tasks,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=subject_set,
    )
    n_scores = len(order_scores)
    n_tasks = len(order_tasks)

    # figure style
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
    legend = ax.legend(
        handles[:n_scores],
        [format_title(l) for l in labels[:n_scores]],
        title="Score",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        borderaxespad=0.2,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    # labels
    ax.set_xlabel("Task", fontsize=14)
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels([format_title(t) for t in order_tasks], fontsize=12)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # compute box centers (for placement of p-value brackets)
    centers = compute_box_centers(ax, order_tasks, order_scores, width=0.6)

    # compute whisker tops (97.5th percentile) per task & score
    whisker_top = compute_whisker_tops(
        boot_df, value_col="r2", group_cols=("task", "target"), q=0.975
    )

    # plot style: axis + grid
    axis_top = 0.395
    style_r2_axis(ax, axis_top=axis_top, tick_step=0.1, show_zero_line=True)

    # placement of p-value brackets above whiskers
    pad = 0.01  # above whisker
    gap = 0.02  # between stacked brackets
    height = 0.004  # bracket height
    top_pad = 0.003  # space below top axis
    y_max_allowed = axis_top - top_pad - height

    score_index = {s: i for i, s in enumerate(order_scores)}

    max_bracket_y = None

    # draw brackets per task between scores
    for task in order_tasks:
        test_df = compare_scores(
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
        # only show significant pairs
        significant_pairs = [r for r in test_df.itertuples(index=False) if r.p_adj < alpha]
        if not significant_pairs:
            continue

        bracket_levels = {}
        for s in order_scores:
            whisker_value = whisker_top.get((task, s), np.nan)
            bracket_levels[s] = (whisker_value + pad) if not np.isnan(whisker_value) else 0.0
        # shorter brackets (closer scores) first
        significant_pairs.sort(key=lambda r: abs(score_index[r.A] - score_index[r.B]))

        for r in significant_pairs:
            s1, s2 = r.A, r.B
            if s1 not in order_scores or s2 not in order_scores:
                continue
            # get position on x-axis
            x1 = centers[(task, s1)]
            x2 = centers[(task, s2)]
            # base just above the higher of the two current levels
            y_base = max(bracket_levels[s1], bracket_levels[s2])

            if y_base > y_max_allowed:
                y_base = y_max_allowed
            # draw bracket
            ax.plot(
                [x1, x1, x2, x2],
                [y_base, y_base + height, y_base + height, y_base],
                lw=linew, color="#333", clip_on=True,
            )
            # add p-label above bracket
            text_y = y_base + height
            ax.text(
                0.5 * (x1 + x2),
                y_base + height,
                format_p_label(
                    r.p_adj,
                    stars=True,
                    decimals=3,
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
            if max_bracket_y is None or text_y > max_bracket_y:
                max_bracket_y = text_y
            # update levels so next brackets stack above
            new_level = y_base + gap
            bracket_levels[s1] = max(bracket_levels[s1], new_level)
            bracket_levels[s2] = max(bracket_levels[s2], new_level)

    if max_bracket_y is not None:
        cur_lo, cur_hi = ax.get_ylim()
        span = cur_hi - cur_lo if cur_hi > cur_lo else 1.0
        headroom = 0.03 * span  # small padding above highest label
        needed_top = max_bracket_y + headroom
        if needed_top > cur_hi:
            ax.set_ylim(cur_lo, needed_top)

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    # save plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, "score_comparison_boxplot_pvals_all_tasks.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")


def plot_task_comparison_boxplots_bootstrapped(
        oof_preds,
        model="full",
        n_boot=N_BOOT,
        ci=CI,
        random_state=RANDOM_STATE,
        subject_set=None,
        order_scores=None,
        order_tasks=None,
        adjust="holm",
        alpha=ALPHA,
        score_for_brackets=None,  # e.g. "PictureNamingScore" to show only that score
        save_path=None,
):
    """
    Task comparison boxplots (R² bootstrapped) with p-values for pairwise task comparisons within each score
    """

    # subset data for selected model and set score/task order
    boot_df, order_scores, order_tasks = prepare_bootstrapped_r2_by_task_score(
        oof_preds=oof_preds,
        model=model,
        order_scores=order_scores,
        order_tasks=order_tasks,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=subject_set,
    )
    n_scores = len(order_scores)
    n_tasks = len(order_tasks)

    # figure style
    palette = color_palette(order_scores, base="crest", lo=0.0, hi=0.8)

    fig, ax = plt.subplots(figsize=(11, 5))
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
    legend = ax.legend(
        handles[:n_scores],
        [format_title(l) for l in labels[:n_scores]],
        title="Score",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        borderaxespad=0.2,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    # labels
    ax.set_xlabel("Task", fontsize=14)
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels([format_title(t) for t in order_tasks], fontsize=12)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("R² (bootstrapped)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # compute box centers (for placement of p-value brackets)
    centers = compute_box_centers(ax, order_tasks, order_scores, width=0.6)

    # compute whisker tops (97.5th percentile) per task & score
    whisker_top = compute_whisker_tops(
        boot_df, value_col="r2", group_cols=("task", "target"), q=0.975
    )

    # axis (initial)
    axis_top = 0.37
    style_r2_axis(ax, axis_top=axis_top, tick_step=0.1, show_zero_line=True)

    # placement of p-value brackets above whiskers
    pad = 0.000005  # tiny distance above top whisker
    gap = 0.016  # gap between stacked brackets
    height = 0.004  # bracket height
    top_pad = 0.003  # min distance from axis top
    y_max_allowed = axis_top - top_pad - height

    # collect tests
    tests_df = compare_tasks(
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
        return

    task_index = {t: i for i, t in enumerate(order_tasks)}

    max_bracket_y = None

    # draw brackets
    for score in order_scores:
        if score_for_brackets is not None and score != score_for_brackets:
            continue

        sub_tests = tests_df[tests_df["target"] == score]
        if len(sub_tests) == 0:
            continue
        # only show significant pairs
        significant_pairs = [r for r in sub_tests.itertuples(index=False) if r.p_adj < alpha]
        if not significant_pairs:
            continue

        # sort by task distance (shorter brackets first)
        significant_pairs.sort(key=lambda r: abs(task_index[r.A] - task_index[r.B]))
        n_pairs = len(significant_pairs)

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
        required_top = baseline + (n_pairs - 1) * gap
        if required_top > y_max_allowed:
            baseline = y_max_allowed - (n_pairs - 1) * gap
            if baseline < max_for_score + pad:
                baseline = max_for_score + pad

        for pair_index, r in enumerate(significant_pairs):
            t1, t2 = r.A, r.B
            if t1 not in order_tasks or t2 not in order_tasks:
                continue
            # position on x-axis
            x1 = centers[(t1, score)]
            x2 = centers[(t2, score)]
            y_base = baseline + pair_index * gap
            # draw bracket
            ax.plot(
                [x1, x1, x2, x2],
                [y_base, y_base + height, y_base + height, y_base],
                lw=linew, color="#333", clip_on=True,
            )
            # add p-label above bracket
            text_y = y_base + height
            ax.text(
                0.5 * (x1 + x2),
                y_base + height,
                format_p_label(
                    r.p_adj,
                    stars=True,
                    decimals=3,
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
            if max_bracket_y is None or text_y > max_bracket_y:
                max_bracket_y = text_y

    if max_bracket_y is not None:
        cur_lo, cur_hi = ax.get_ylim()
        span = cur_hi - cur_lo if cur_hi > cur_lo else 1.0
        headroom = 0.03 * span
        needed_top = max_bracket_y + headroom
        if needed_top > cur_hi:
            ax.set_ylim(cur_lo, needed_top)

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    # save plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        if score_for_brackets is None:
            score_suffix = "all_scores"
        else:
            score_suffix = str(score_for_brackets)

        score_suffix = re.sub(r"\W+", "_", score_suffix)

        out = os.path.join(
            save_path,
            f"task_comparison_boxplot_with_pvals_{score_suffix}.png",
        )
        plt.savefig(out, dpi=600, bbox_inches="tight")


def plot_model_comparison_boxplots_bootstrapped(
        oof_preds,
        target="SemanticFluencyScore",
        task="picnicScene",
        order_models=None,
        n_boot=N_BOOT,
        ci=CI,
        random_state=RANDOM_STATE,
        subject_set=None,
        adjust="holm",
        save_path=None,
        pairs_to_show=(),  # e.g. [("demographics","acoustic"), ("demographics","linguistic")]
        tests_df=None,
        pval_models=None
):
    """
    Model comparison boxplots (R² bootstrapped) for one score and task with p-values
    """

    # subset & order for boxes
    sub = oof_preds[(oof_preds["task"] == task) & (oof_preds["target"] == target)].copy()
    if subject_set is not None:
        sub = sub[sub["Subject_ID"].isin(subject_set)].copy()
    if order_models is None:
        order_models = sorted(sub["model"].unique().tolist())

    # bootstrap summary
    boot_df, _summ_df, order_models = bootstrap_by_model(
        oof_preds, target, task, order_models, drop_models=None,
        subject_set=subject_set, n_boot=n_boot, ci=ci, random_state=random_state
    )

    # plot
    pal_models = color_palette(order_models, base="crest", lo=0.15, hi=0.85)
    n_models = len(order_models)
    fig_w = min(7.5, 1.15 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.25, n_models - 0.75)
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
        test_df = compare_models(
            oof_preds, task=task, target=target,
            models=tuple(pval_models), drop_models=None,
            n_boot=n_boot, adjust=adjust, random_state=random_state
        )

    # keep only selected pairs
    if not pairs_to_show:
        df_anno = test_df.iloc[0:0].copy()
    else:
        desired_pairs = {tuple(sorted(p)) for p in pairs_to_show}
        df_anno = test_df[test_df.apply(lambda r: tuple(sorted([r.A, r.B])) in desired_pairs, axis=1)].copy()
        df_anno = df_anno[df_anno["A"].isin(order_models) & df_anno["B"].isin(order_models)]

    # bracket placement
    box_artists = ax.artists
    if len(box_artists) != n_models:
        x_centers = np.arange(n_models, dtype=float)
    else:
        x_centers = np.array([b.get_x() + b.get_width() / 2.0 for b in box_artists], dtype=float)
    model_index = {m: i for i, m in enumerate(order_models)}

    pair_rows = list(df_anno.itertuples(index=False))
    pair_rows.sort(key=lambda r: -(abs(model_index[r.A] - model_index[r.B])))

    bracket_lanes, placed_pairs = [], []
    for r in pair_rows:
        i, j = model_index[r.A], model_index[r.B]
        left_index, right_index = (min(i, j), max(i, j))
        for lane_i, bracket_lane in enumerate(bracket_lanes):
            if all(right_index <= lane_left or left_index >= lane_right for (lane_left, lane_right) in bracket_lane):
                bracket_lane.append((left_index, right_index))
                placed_pairs.append((r, lane_i))
                break
        else:
            bracket_lanes.append([(left_index, right_index)])
            placed_pairs.append((r, len(bracket_lanes) - 1))

    bracket_y_start = 0.99
    bracket_height = 0.022
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for r, lane_i in placed_pairs:
        i, j = model_index[r.A], model_index[r.B]
        xi, xj = x_centers[i], x_centers[j]
        y_top = bracket_y_start - lane_i * 0.05
        y0 = y_top - bracket_height
        ax.plot([xi, xi, xj, xj], [y0, y_top, y_top, y0],
                lw=linew, color="#333", transform=trans, clip_on=False)
        ax.text((xi + xj) / 2.0, y_top,
                format_p_label(r.p_adj, stars=True, decimals=3),
                ha="center", va="bottom", fontsize=8, transform=trans, clip_on=False)

    # style
    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_xlabel("Model", fontsize=12);
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)
    ax.set_xticks(np.arange(len(order_models)))
    ax.set_xticklabels([format_model_label(m) for m in order_models])
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#9AA0A6", alpha=0.55)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc");
        ax.spines[side].set_linewidth(0.5);
        ax.spines[side].set_visible(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{task}_{target}_model_boxplot_with_pvals.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")


def plot_bootstrap_models_tasks(
        oof_preds,
        target,
        order_models,
        order_tasks,
        n_boot=N_BOOT,
        ci=CI,
        subject_set=None,
        save_path=None
):
    """
    Model comparison boxplots for all tasks for one score (without statistical comparisons)
    """
    sub = oof_preds[oof_preds["target"] == target].copy()

    # bootstrap summary
    boot_df, summ_df = bootstrap_r2_summary(
        sub, group_cols=("model", "task"), n_boot=n_boot, ci=ci, subject_set=subject_set
    )

    # filter & order
    boot_df = boot_df[boot_df["model"].isin(order_models)]
    summ_df = summ_df[summ_df["model"].isin(order_models)]

    cat_model = CategoricalDtype(categories=order_models, ordered=True)
    cat_task = CategoricalDtype(categories=order_tasks, ordered=True)
    boot_df["model"] = boot_df["model"].astype(cat_model)
    boot_df["task"] = boot_df["task"].astype(cat_task)

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
        showcaps=True, showfliers=False, boxprops={"alpha": 1},
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
    legend = ax.legend(
        handles[:len(order_tasks)], labels[:len(order_tasks)],
        title="Task", loc="upper left",
        fontsize=9, title_fontsize=10,
        frameon=True, borderaxespad=0.2, labelspacing=0.3,
        handlelength=1.0, handletextpad=0.4
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.5)
    frame.set_alpha(1.0)

    ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=-10)
    ax.set_title(target)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticklabels([format_model_label(m) for m in order_models])
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


### picture description analyses

def plot_pd_duration_box_with_pvals(
        oof_preds,
        target="SemanticFluencyScore",
        model="full",
        order_tasks=("picture_description_1min", "picture_description_2min", "picture_description"),
        n_boot=N_BOOT,
        ci=CI,
        random_state=RANDOM_STATE,
        subject_set=None,
        adjust="holm",
        save_path=None,
        pairs_to_show=None,
        display_labels=None,
):
    """
    Task comparison boxplots (R² bootstrapped) for picture description duration variants (1min / 2min / ≤5min)
    """

    # subset & order
    sub = oof_preds[(oof_preds["target"] == target) & (oof_preds["model"] == model)].copy()
    if subject_set is not None:
        sub = sub[sub["Subject_ID"].isin(subject_set)].copy()

    cat_task = CategoricalDtype(categories=list(order_tasks), ordered=True)
    sub["task"] = sub["task"].astype(cat_task)

    # bootstrap distributions for boxes
    boot_df, _ = bootstrap_r2_summary(
        sub,
        group_cols=("task",),
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        subject_set=None,
    )
    boot_df["task"] = boot_df["task"].astype(cat_task)

    # pairwise tests
    test_df = compare_tasks_for_one_score(
        oof_preds,
        target=target,
        model=model,
        tasks=tuple(order_tasks),
        n_boot=n_boot,
        adjust=adjust,
        random_state=random_state,
        subject_set=subject_set,
    )
    if pairs_to_show is not None:
        desired_pairs = {tuple(sorted(p)) for p in pairs_to_show}
        test_df = test_df[
            test_df.apply(
                lambda r: tuple(sorted([r["task_A"], r["task_B"]])) in desired_pairs,
                axis=1,
            )
        ].copy()

    # plot
    fig, ax = plt.subplots(figsize=(max(6, 1 * len(order_tasks)), 3.6))
    pal = sns.color_palette("crest", n_colors=len(order_tasks))
    sns.boxplot(
        data=boot_df,
        x="task",
        y="r2",
        order=order_tasks,
        palette=pal,
        width=0.3,
        showcaps=True,
        showfliers=False,
        whis=(2.5, 97.5),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="#111", linewidth=0.8, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=0.8,
        ax=ax,
    )

    ax.margins(x=0.12)
    fig.subplots_adjust(bottom=0.18)

    # dynamic pads
    y_lo = float(boot_df["r2"].min())
    y_hi = float(boot_df["r2"].max())
    y_span = max(1e-9, y_hi - y_lo)

    box_top_padding = 0.06 * y_span
    bracket_vertical_gap = 0.08 * y_span
    bracket_height = 0.02 * y_span
    bottom_padding = 0.03 * y_span
    label_extra_padding = 0.010 * y_span

    # avoid overlapping brackets
    task_index = {t: i for i, t in enumerate(order_tasks)}
    bracket_candidates = []
    for r in test_df.itertuples(index=False):
        i, j = task_index[r.task_A], task_index[r.task_B]
        bracket_candidates.append((abs(i - j), min(i, j), max(i, j), r))
    bracket_candidates.sort(reverse=True)

    bracket_lanes, placed_brackets = [], []
    for _, left_index, right_index, row in bracket_candidates:
        for lane_index, bracket_lane in enumerate(bracket_lanes):
            if all(right_index <= lane_left or left_index >= lane_right for (lane_left, lane_right) in bracket_lane):
                bracket_lane.append((left_index, right_index))
                placed_brackets.append((row, lane_index))
                break
        else:
            bracket_lanes.append([(left_index, right_index)])
            placed_brackets.append((row, len(bracket_lanes) - 1))

    # x positions for bracket ends
    x_centers = np.arange(len(order_tasks), dtype=float)

    # y levels for brackets
    first_bracket_y_position = y_hi + box_top_padding
    top_bracket_y_position = first_bracket_y_position + max(0, len(bracket_lanes) - 1) * (
                bracket_height + bracket_vertical_gap) + label_extra_padding

    for row, lane_index in placed_brackets:
        i, j = task_index[row.task_A], task_index[row.task_B]
        xi, xj = x_centers[i], x_centers[j]
        y_top = first_bracket_y_position + lane_index * (bracket_height + bracket_vertical_gap)
        y0 = y_top - bracket_height
        ax.plot(
            [xi, xi, xj, xj],
            [y0, y_top, y_top, y0],
            lw=0.9,
            color="#333",
            clip_on=False,
        )
        ax.text(
            (xi + xj) / 2,
            y_top,
            format_p_label(row.p_adj, stars=True, decimals=3),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )

    # y-limits
    y_min = y_lo - bottom_padding
    y_max = top_bracket_y_position
    ax.set_ylim(y_min, y_max)

    step = 0.05
    lo = math.floor(ax.get_ylim()[0] / step) * step
    hi = math.ceil(ax.get_ylim()[1] / step) * step
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    ax.set_xlabel("Feature Set Variant", fontsize=12)
    # labels: use mapping if provided
    if display_labels is not None:
        labels = [display_labels.get(t, t).replace("_", " ") for t in order_tasks]
    else:
        labels = [t.replace("_", " ") for t in order_tasks]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylabel("R² (bootstrapped)", fontsize=12)

    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle=":",
        linewidth=0.8,
        color="#9AA0A6",
        alpha=0.55,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{target}_tasks_boxplot_with_pvals.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")

    return test_df


### bias analyses

def format_bias_labels(group_col, levels):
    """
    Format demographic subgroup labels on x-axis for bias plots
    """
    mapping = {
        "Gender_label": {"f": "Female", "m": "Male"},
        "Country_label": {"uk": "UK", "usa": "US"},
        "CountryGender_label": {
            "uk_f": "UK Female",
            "uk_m": "UK Male",
            "usa_f": "US Female",
            "usa_m": "US Male",
        },
    }
    m = mapping.get(group_col, {})
    return [m.get(l, str(l)) for l in levels]


def plot_bias_subgroup_comparison(
        oof_preds,
        target,
        group_col,
        levels,
        task,
        model="full",
        metric="R2",
        save_path=None,
        pairs_to_show=None,
        ax=None,
        x_label=None,
        show_ylabel=True,
):
    """
    Boxplots for comparisons of demographic subgroups (bias analyses)
    """

    # fix parameters for plots
    box_width = 0.24
    whiskers = (2.5, 97.5)
    bracket_scale = 3.0
    ceil_pad = 0.06

    metric = metric.upper()

    # subset & order
    sub, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    # bootstrap
    boot_rows = []
    for i, lvl in enumerate(levels):
        d = sub[sub[group_col] == lvl]
        draws = bootstrap_metric_unpaired(d, metric, n_boot=N_BOOT, random_state=RANDOM_STATE + 1000 * i)
        boot_rows.extend([{group_col: lvl, "value": v} for v in draws])
    boot_df = pd.DataFrame(boot_rows)
    boot_df[group_col] = boot_df[group_col].astype(CategoricalDtype(categories=levels, ordered=True))

    pal = bias_palette(group_col, levels)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        created_fig = True
    else:
        fig = ax.figure

    # plot
    sns.boxplot(
        data=boot_df, x=group_col, y="value",
        order=levels, hue=group_col, hue_order=levels, legend=False,
        palette=pal, dodge=False, width=box_width,
        showcaps=True, showfliers=False,
        boxprops={"alpha": 1, "edgecolor": "black"},
        whiskerprops={"alpha": 1, "color": "black"},
        capprops={"alpha": 1, "color": "black"},
        whis=whiskers, showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=0.9, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=0.8, ax=ax
    )

    # labels
    ax.margins(x=0.12)
    ticks = np.arange(len(levels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(format_bias_labels(group_col, levels))
    label = x_label if x_label is not None else group_col.replace("_label", "").capitalize()
    ax.set_xlabel(label, fontsize=17)
    if show_ylabel:
        ax.set_ylabel(metric_label(metric), fontsize=17)
    else:
        ax.set_ylabel(None)

    ax.tick_params(axis="x", labelsize=14, pad=2)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_axisbelow(True)
    if metric == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    else:
        ax.yaxis.set_major_locator(AutoLocator())
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#E0E0E0", alpha=0.6)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc");
        ax.spines[side].set_linewidth(0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if metric == "R2":
        ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    # tests (Holm-Bonferroni corrected)
    tests_all = compare_subgroups(
        oof_preds=oof_preds, group_col=group_col, levels=levels,
        task=task, target=target, model=model,
        metric=metric, n_boot=N_BOOT, ci=CI, random_state=RANDOM_STATE
    )
    if pairs_to_show is None:
        tests_df = tests_all.copy()
    else:
        desired_pairs = {tuple(sorted(p)) for p in pairs_to_show}
        tests_df = tests_all[
            tests_all.apply(lambda r: tuple(sorted((r["A"], r["B"]))) in desired_pairs, axis=1)
        ].copy()

    # y-limits
    lo, hi = ax.get_ylim()
    axis_range = hi - lo
    bracket_height = 0.012 * axis_range
    bracket_y_padding = (0.010 * bracket_scale) * axis_range
    bracket_min_gap = (2.2 * bracket_scale) * bracket_height

    x_centers = ax.get_xticks()
    level_index = {lvl: i for i, lvl in enumerate(levels)}
    ymax_by_level = (boot_df.groupby(group_col, observed=True)["value"]
                     .quantile(0.98).reindex(levels).to_numpy())

    placed_brackets = []
    for _, r in tests_df.sort_values("p_adj").iterrows():
        A, B, p_adj = r["A"], r["B"], r["p_adj"]
        i, j = level_index[A], level_index[B]
        if i > j:
            i, j = j, i
        y_base = max(ymax_by_level[i], ymax_by_level[j]) + bracket_y_padding
        y_top = y_base + bracket_height

        bumped = True
        while bumped:
            bumped = False
            for (prev_i, prev_j, prev_y_top) in placed_brackets:
                if not (j < prev_i or prev_j < i):
                    if y_base < (prev_y_top + bracket_min_gap):
                        y_base = prev_y_top + bracket_min_gap
                        y_top = y_base + bracket_height
                        bumped = True

        # only draw bracket + label if significant
        if p_adj < 0.05:
            ax.plot(
                [x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
                [y_base, y_top, y_top, y_base],
                lw=0.8,
                color="#333",
            )

            stars_txt = format_p_label(
                p_adj,
                stars=True,
                decimals=3,
            )

            ax.text(
                (x_centers[i] + x_centers[j]) / 2.0,
                y_top,
                stars_txt,
                ha="center",
                va="bottom",
                fontsize=12,
            )

            placed_brackets.append((i, j, y_top))
            ymax_by_level[i] = max(ymax_by_level[i], y_top + 0.2 * bracket_min_gap)
            ymax_by_level[j] = max(ymax_by_level[j], y_top + 0.2 * bracket_min_gap)

    # headroom
    cur_bot, cur_top = ax.get_ylim()
    peak_top = max([py for (_, _, py) in placed_brackets], default=cur_top)
    maybe_top = max(cur_top, peak_top + ceil_pad * (cur_top - cur_bot))
    if maybe_top > cur_top:
        ax.set_ylim(cur_bot, maybe_top)

    plt.tight_layout()

    # save plot
    if save_path and created_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return ax


def plot_bias_panels(
        oof_preds,
        target,
        task,
        model="full",
        metric="R2",
        save_path=None
):
    """
    Create panel plots ABC (Country, Gender, Age) for bias analyses
    """
    metric = metric.upper()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=False)

    # A: Country
    plot_bias_subgroup_comparison(
        oof_preds, target, "Country_label", ("uk", "usa"),
        task=task, model=model, metric=metric, ax=axes[0], show_ylabel=True
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=16, fontweight="bold")

    # B: Gender
    plot_bias_subgroup_comparison(
        oof_preds, target, "Gender_label", ("f", "m"),
        task=task, model=model, metric=metric, ax=axes[1], show_ylabel=False
    )
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=16, fontweight="bold")

    # C: Age
    current_subset = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target)
        ].copy()
    current_subset = current_subset.drop(columns=["AgeGroup", "AgeGroup2"], errors="ignore")

    age_df, age_levels = make_age_binary_groups(
        current_subset, age_col="Age", label_col="AgeGroup2"
    )

    plot_bias_subgroup_comparison(
        age_df, target, "AgeGroup2", age_levels,
        task=task, model=model, metric=metric,
        ax=axes[2], x_label="Age", show_ylabel=False
    )
    axes[2].text(0.02, 0.98, "C", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=16, fontweight="bold")

    # unify y across panels
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)

    fig.align_ylabels()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    return fig, axes
