import os
import numpy as np
import pandas as pd

from itertools import combinations

from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator
import seaborn as sns

from regression.model_evaluation_helpers import (
    adjust_pvals, confidence_intervals, p_value_bootstrap
)
from regression.plotting_helpers import format_p_label

# style
plt.rcParams["font.family"] = "Arial"
sns.set_theme(context="paper", style="white")

# parameters
random_state = 42
ci = 0.95
n_boot = 1000

# calculate metrics
def metric_value(y_true, y_pred, metric):
    m = metric.upper()
    if m == "R2":
        return float(r2_score(y_true, y_pred))
    err = y_pred - y_true
    if m == "RMSE":
        return float(np.sqrt(np.mean(err**2)))
    if m == "MAE":
        return float(np.mean(np.abs(err)))

# bootstrap metric -> unpaired because subjects from different subgroups are compared
def bootstrap_metric_unpaired(df, metric, n_boot = n_boot, random_state = random_state):

    rng = np.random.default_rng(random_state) # generate random numbers
    idx_by_sid = df.groupby("Subject_ID").indices
    subj = np.array(list(idx_by_sid.keys())) # identify subjects in the group
    n = len(subj) # -> resample same size as original group

    ytrue = df["y_true"].to_numpy()
    ypred = df["y_pred"].to_numpy()

    # bootstrapping
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot_ids = rng.choice(subj, size=n, replace=True)
        idxs = np.concatenate([idx_by_sid[s] for s in boot_ids])
        draws[b] = metric_value(ytrue[idxs], ypred[idxs], metric)
    return draws

def metric_label(metric):
    return "R\u00B2" if metric.upper() == 'R2' else metric

# create subset of data
def subset_in_order(oof_preds, task, model, target, group_col, levels):
    levels = list(levels)
    sub = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target) &
        (oof_preds[group_col].isin(levels))
    ].copy()
    sub[group_col] = sub[group_col].astype(CategoricalDtype(categories=levels, ordered=True))
    return sub, levels

# prepare age groups: 2 age bins (<65 & ≥65) (three were unstable; third too small)
def make_age_binary_groups(df, age_col="Age", label_col="AgeGroup2"):

    ages = pd.to_numeric(df[age_col], errors="coerce")

    bins = [float("-inf"), 65, float("inf")]
    right = True
    labels = ["<65", "≥65"]

    df2 = df.copy()
    df2[label_col] = pd.cut(ages, bins=bins, include_lowest=True, right=right)
    # rename categories for labels
    df2[label_col] = df2[label_col].cat.rename_categories(labels)

    return df2, labels

# create summary per subgroup with all metrics: R^2, MAE, RMSE (with CIs); variance (y_pred), mean error (y_true - y_pred)
def summarize_groups_all_metrics(
    oof_preds,
    task,
    model,
    target,
    group_col,
    levels,
    n_boot = n_boot,
    ci = ci,
    random_state = random_state,
):

    sub, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    rows = []
    for i, lvl in enumerate(levels):
        d = sub[sub[group_col] == lvl]
        if d.empty:
            rows.append({
                group_col: lvl, "n_subjects": 0,
                "R2_mean": np.nan, "R2_ci_low": np.nan, "R2_ci_high": np.nan,
                "MAE_mean": np.nan, "MAE_ci_low": np.nan, "MAE_ci_high": np.nan,
                "RMSE_mean": np.nan, "RMSE_ci_low": np.nan, "RMSE_ci_high": np.nan,
                "var_pred": np.nan, "mean_error": np.nan
            })
            continue

        # bootstrap draws for each metric
        out = {group_col: lvl, "n_subjects": int(d["Subject_ID"].nunique())}
        for j, metric in enumerate(("R2","MAE","RMSE")):
            draws = bootstrap_metric_unpaired(d, metric, n_boot=n_boot, random_state=random_state + 1000*i)
            mean = float(np.mean(draws))
            lo, hi = confidence_intervals(draws, ci=ci)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = float(lo)
            out[f"{metric}_ci_high"] = float(hi)

        out["var_pred"]   = float(np.nanvar(d["y_pred"].to_numpy(), ddof=1)) if len(d) > 1 else 0.0
        out["mean_error"] = float(np.nanmean(d["y_true"].to_numpy() - d["y_pred"].to_numpy()))
        rows.append(out)

    df = pd.DataFrame(rows)

    # column order
    fixed = [group_col, "n_subjects",
             "R2_mean","R2_ci_low","R2_ci_high",
             "MAE_mean","MAE_ci_low","MAE_ci_high",
             "RMSE_mean","RMSE_ci_low","RMSE_ci_high",
             "var_pred","mean_error"]

    return df[fixed]
# pairwise comparisons for one metric at a time (R^2, RMSE & MAE) -> Δ(A-B), CI, p-value
def compare_groups_metric(
    oof_preds,
    group_col, # select variable to split on (Gender, Country, Age)
    levels, # allowed levels and order
    task,
    target,
    model="full",
    metric="R2",
    n_boot=n_boot,
    ci=ci,
    random_state=random_state,
    pairs_to_test=None # select what levels to test against each other (if None -> all)
):
    metric = metric.upper()
    sub, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    # index for stable per-level seeds
    idx = {lvl: i for i, lvl in enumerate(levels)}

    pairs = list(combinations(levels, 2)) if (pairs_to_test is None) else [tuple(p) for p in pairs_to_test]
    rows = []
    for (A, B) in pairs:
        dA = sub[sub[group_col] == A]
        dB = sub[sub[group_col] == B]

        if dA.empty or dB.empty:
            rows.append({"A": A, "B": B, "mean_diff": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan,
                         "p_raw": np.nan, "p_adj": np.nan})
            continue

        # observed difference
        obs = metric_value(dA["y_true"].to_numpy(), dA["y_pred"].to_numpy(), metric) - \
              metric_value(dB["y_true"].to_numpy(), dB["y_pred"].to_numpy(), metric)

        # use the SAME per-level seeds as summaries/plots
        seed_A = random_state + 1000 * idx[A]
        seed_B = random_state + 1000 * idx[B]

        # bootstrap draws
        mA = bootstrap_metric_unpaired(dA, metric, n_boot=n_boot, random_state=seed_A)
        mB = bootstrap_metric_unpaired(dB, metric, n_boot=n_boot, random_state=seed_B)
        diff = mA - mB

        mean_d = float(np.nanmean(diff))
        lo, hi = confidence_intervals(diff, ci=ci)
        p = float(p_value_bootstrap(diff, obs))

        rows.append({"A": A, "B": B, "mean_diff": mean_d,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "p_raw": p, "p_adj": None})

    out = pd.DataFrame(rows)
    # correct p-values for multiple testing
    out["p_adj"] = adjust_pvals(out["p_raw"].to_numpy(), method="holm")
    out.insert(0, "group_col", group_col)
    cols = ["group_col","A","B","mean_diff","ci_low","ci_high","p_raw","p_adj"]
    return out[cols].sort_values(["group_col","p_adj","A","B"])

# format labels for plots (x-axes)
def format_bias_labels(group_col, levels):
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

# define color palette for group-comparisons
def bias_palette(group_col, levels):
    # for country and gender
    blue = "#6AA2FF"   # for male + US
    red  = "#E07C7C"   # for female + UK

    levels = list(levels)

    if group_col == "Gender_label":
        mapping = {"m": blue, "f": red}
        return {lvl: mapping.get(lvl, blue) for lvl in levels}

    if group_col == "Country_label":
        mapping = {"uk": red, "usa": blue}
        return {lvl: mapping.get(lvl, blue) for lvl in levels}

    # different blues for Age (3 or 2 groups) (darker with older age)
    if group_col in ("AgeGroup", "AgeGroup2"):
        seq = sns.color_palette("Blues", n_colors=len(levels))
        return {lvl: seq[i] for i, lvl in enumerate(levels)}

    if group_col == "CountryGender_label":
        mapping = {
            "uk_f": "#E07C7C",  # UK Female  (red)
            "uk_m": "#C95F5F",  # UK Male    (darker red)
            "usa_f": "#8DB8FF",  # US Female  (lighter blue)
            "usa_m": "#6AA2FF",  # US Male    (blue)
        }
        return {lvl: mapping.get(lvl, "#6AA2FF") for lvl in levels}

    # fallback
    seq = sns.color_palette("Blues", n_colors=len(levels))
    return {lvl: seq[i] for i, lvl in enumerate(levels)}

# boxplots with p-values above for comparisons
def plot_group_box_with_pvals(
    oof_preds,
    target,
    group_col,
    levels,
    task,
    model = "full",
    metric = "R2",
    save_path = None,
    pairs_to_show = None,
    ax=None,
    x_label = None
):

    # fix parameters for plots
    box_width = 0.24
    whiskers = (2.5, 97.5)
    bracket_scale = 3.0
    ceil_pad = 0.06

    metric = metric.upper()

    # subset + order
    sub, levels = subset_in_order(oof_preds, task, model, target, group_col, levels)

    # bootstrap draws per level
    boot_rows = []
    for i, lvl in enumerate(levels):
        d = sub[sub[group_col] == lvl]
        draws = bootstrap_metric_unpaired(d, metric, n_boot=n_boot, random_state=random_state + 1000*i)
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

    ax.margins(x=0.12)
    ticks = np.arange(len(levels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(format_bias_labels(group_col, levels))
    label = x_label if x_label is not None else group_col.replace("_label","").capitalize()
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel(metric_label(metric), fontsize=12)

    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.set_axisbelow(True)
    if metric == "R2":
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    else:
        ax.yaxis.set_major_locator(AutoLocator())
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8, color="#E0E0E0", alpha=0.4)
    for side in ("left","bottom"):
        ax.spines[side].set_color("#cccccc"); ax.spines[side].set_linewidth(0.5)
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    if metric == "R2":
        ax.axhline(0, color="#777", lw=0.6, ls="--", alpha=0.7, zorder=0.5)

    # tests (Holm-Bonferroni corrected)
    tests_all = compare_groups_metric(
        oof_preds=oof_preds, group_col=group_col, levels=levels,
        task=task, target=target, model=model,
        metric=metric, n_boot=n_boot, ci=ci, random_state=random_state
    )
    if pairs_to_show is None:
        tests_df = tests_all.copy()
    else:
        want = {tuple(sorted(p)) for p in pairs_to_show}
        tests_df = tests_all[
            tests_all.apply(lambda r: tuple(sorted((r["A"], r["B"]))) in want, axis=1)
        ].copy()

    # y-limits
    lo, hi = ax.get_ylim()
    axis_range = hi - lo
    height   = 0.012 * axis_range
    y_pad    = (0.010 * bracket_scale) * axis_range
    rung_gap = (2.2  * bracket_scale) * height

    x_centers = ax.get_xticks()
    idx = {lvl: i for i, lvl in enumerate(levels)}
    ymax_by_level = (boot_df.groupby(group_col, observed=True)["value"]
                     .quantile(0.98).reindex(levels).to_numpy())

    placed = []
    for _, r in tests_df.sort_values("p_adj").iterrows():
        A, B, p_adj = r["A"], r["B"], r["p_adj"]
        i, j = idx[A], idx[B]
        if i > j: i, j = j, i
        y_base = max(ymax_by_level[i], ymax_by_level[j]) + y_pad
        y_top  = y_base + height

        # bump up if colliding
        bumped = True
        while bumped:
            bumped = False
            for (pi, pj, py_top) in placed:
                if not (j < pi or pj < i):
                    if y_base < (py_top + rung_gap):
                        y_base = py_top + rung_gap
                        y_top  = y_base + height
                        bumped = True

        # draw
        ax.plot([x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
                [y_base, y_top, y_top, y_base], lw=0.8, color="#333")

        # label -> stars + numeric p-value
        stars_txt = format_p_label(
            p_adj,
            stars=True,
            decimals=3,
            drop_leading_zero=True,
            use_lt_for_small=True,
            n_boot=n_boot,
        )

        label_txt = f"{stars_txt}"

        ax.text((x_centers[i] + x_centers[j]) / 2.0, y_top,
                label_txt, ha="center", va="bottom", fontsize=8)

        placed.append((i, j, y_top))
        ymax_by_level[i] = max(ymax_by_level[i], y_top + 0.2*rung_gap)
        ymax_by_level[j] = max(ymax_by_level[j], y_top + 0.2*rung_gap)

    # headroom
    cur_bot, cur_top = ax.get_ylim()
    peak = max([py for (_,_,py) in placed], default=cur_top)
    maybe_top = max(cur_top, peak + ceil_pad * (cur_top - cur_bot))
    if maybe_top > cur_top:
        ax.set_ylim(cur_bot, maybe_top)

    plt.tight_layout()
    if save_path and created_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    if created_fig:
        plt.show()
    return ax

# create panels ABC (Country, Gender, Age)
def plot_bias_panels(
    oof_preds,
    target,
    task,
    model = "full",
    metric = "R2",
    save_path = None
):

    metric = metric.upper()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=False)

    # A: Country
    plot_group_box_with_pvals(
        oof_preds, target, "Country_label", ("uk","usa"),
        task=task, model=model, metric=metric, ax=axes[0]
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # B: Gender
    plot_group_box_with_pvals(
        oof_preds, target, "Gender_label", ("f", "m"),
        task=task, model=model, metric=metric, ax=axes[1]
    )
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # C: Age
    cur = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target)
    ].copy()
    cur = cur.drop(columns=["AgeGroup", "AgeGroup2"], errors="ignore")

    age_df, age_levels = make_age_binary_groups(
        cur, age_col="Age", label_col="AgeGroup2"
    )

    plot_group_box_with_pvals(
        age_df, target, "AgeGroup2", age_levels,
        task=task, model=model, metric=metric,
        ax=axes[2], x_label="Age"
    )
    axes[2].text(0.02, 0.98, "C", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # unify y across panels
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return fig, axes

# run everything for one score
def run_bias_for_score(
    oof_preds,
    outdir,
    target,
    task = "picnicScene",
    model = "full",
    n_boot = n_boot,
    ci = ci,
    random_state = random_state,
):
    os.makedirs(outdir, exist_ok=True)

    cur = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target)
    ].copy()

    # summaries
    g_summary = summarize_groups_all_metrics(
        oof_preds, task, model, target, "Gender_label", ("f","m"),
        n_boot=n_boot, ci=ci, random_state=random_state
    )
    c_summary = summarize_groups_all_metrics(
        oof_preds, task, model, target, "Country_label", ("uk","usa"),
        n_boot=n_boot, ci=ci, random_state=random_state+10
    )
    # Age: <65 vs ≥65
    cur = cur.drop(columns=["AgeGroup", "AgeGroup2"], errors="ignore")
    age_df, age_levels = make_age_binary_groups(
        cur, age_col="Age", label_col="AgeGroup2"
    )
    a_summary = summarize_groups_all_metrics(
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
    for metric in ("R2", "MAE", "RMSE"):
        tests = []
        tests.append(compare_groups_metric(oof_preds, "Gender_label", ("f","m"),
                                           task, target, model, metric, n_boot, ci, random_state))
        tests.append(compare_groups_metric(oof_preds, "Country_label", ("uk","usa"),
                                           task, target, model, metric, n_boot, ci, random_state))
        tests.append(compare_groups_metric(age_df, "AgeGroup2", age_levels,
                                           task, target, model, metric, n_boot, ci, random_state))

        tests_all = pd.concat(tests, ignore_index=True)
        out = os.path.join(outdir, f"bias_pairwise_tests_{metric}_{target}.csv")
        tests_all.to_csv(out, index=False)
        paths[f"pairwise_tests_{metric}_csv"] = out

    # panel plots per metric
    for metric in ("R2", "MAE", "RMSE"):
        img = os.path.join(outdir, f"bias_panels_{metric}_{target}.png")
        plot_bias_panels(
            oof_preds, target, task, model, metric=metric,
            save_path=img
        )
        paths[f"panels_{metric}_png"] = img

    return paths
