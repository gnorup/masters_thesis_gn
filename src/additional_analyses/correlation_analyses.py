# Calculate feature-score, score-score and demographics-score correlations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter
import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
sns.set(style="whitegrid", rc={"font.family": "Arial"})

from config.constants import GIT_DIRECTORY, ID_COL, TASKS, SCORES
from data_preparation.data_handling import load_demographics
from regression.plots import format_title

# config
SCORES_PATH = os.path.join(GIT_DIRECTORY, "data", "language_scores_all_subjects.csv")
FEATURES_DIR = os.path.join(GIT_DIRECTORY, "results", "features", "cleaned")

CORR_OUTDIR = os.path.join(GIT_DIRECTORY, "results", "correlations")
FEATURE_CORR_DIR = os.path.join(CORR_OUTDIR, "features")
SCORE_CORR_DIR   = os.path.join(CORR_OUTDIR, "scores")
DEMO_CORR_DIR    = os.path.join(CORR_OUTDIR, "demographics")

for p in (FEATURE_CORR_DIR, SCORE_CORR_DIR, DEMO_CORR_DIR):
    os.makedirs(p, exist_ok=True)


def compute_correlations_for_task(
    task_name,
    scores=SCORES,
    outdir = FEATURE_CORR_DIR,
):
    """
    Calculate feature-score correlations for given task (with r, r², Holm-corrected p)
    """
    # load data
    scores_df = pd.read_csv(SCORES_PATH)[[ID_COL, *scores]]
    features_path = os.path.join(FEATURES_DIR, f"{task_name}_cleaned.csv")
    features_df = pd.read_csv(features_path)

    for df in (scores_df, features_df):
        df[ID_COL] = df[ID_COL].astype(str)

    df = pd.merge(features_df, scores_df, on=ID_COL, how="inner").dropna()

    feature_cols = [c for c in features_df.columns if c != ID_COL]
    corr_results = pd.DataFrame({"feature": feature_cols})

    for target in scores:
        r_values, p_values = [], []

        for f in feature_cols:
            r, p = pearsonr(df[f], df[target])
            r_values.append(r)
            p_values.append(p)

        # Holm–Bonferroni correction within each target
        _, p_corr, _, _ = multipletests(p_values, method="holm")

        r_values = np.asarray(r_values, dtype=float)
        r2_values = r_values ** 2

        corr_results[f"r_{target}"] = r_values
        corr_results[f"r2_{target}"] = r2_values
        corr_results[f"p_{target}"] = p_corr

    # save CSV
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{task_name}_feature_score_correlations.csv")
    corr_results.to_csv(out_path, index=False)
    print(f"[compute_correlations_for_task] saved correlation results to: {out_path}")

    return corr_results


def compute_all_correlations(
    tasks=TASKS,
    scores=SCORES,
    outdir = FEATURE_CORR_DIR,
):
    """
    Calculate correlations across tasks and scores (R, R², corrected p) and plot heatmap
    """

    os.makedirs(outdir, exist_ok=True)

    # load scores
    scores_df = pd.read_csv(SCORES_PATH)[[ID_COL, *scores]]
    scores_df[ID_COL] = scores_df[ID_COL].astype(str)

    # correlations and p-values
    all_r = {}
    all_p = {}
    all_features = set()

    # loop over tasks and scores
    for task in tasks:
        features_path = os.path.join(FEATURES_DIR, f"{task}_cleaned.csv")

        features = pd.read_csv(features_path)
        features[ID_COL] = features[ID_COL].astype(str)

        df = pd.merge(features, scores_df, on=ID_COL, how="inner").dropna()

        feat_cols = [c for c in features.columns if c != ID_COL]
        all_features.update(feat_cols)

        for score in scores:
            r_vals, p_vals = [], []
            for f in feat_cols:
                r, p = pearsonr(df[f], df[score])
                r_vals.append(r)
                p_vals.append(p)

            _, p_corr, _, _ = multipletests(p_vals, method="holm")

            all_r[(score, task)] = pd.Series(r_vals, index=feat_cols, dtype=float)
            all_p[(score, task)] = pd.Series(p_corr, index=feat_cols, dtype=float)

    all_features = sorted(all_features)

    # fixed order for tasks × scores
    ordered_cols = [
        (scores[0], tasks[0]), (scores[0], tasks[1]), (scores[0], tasks[2]),
        (scores[1], tasks[0]), (scores[1], tasks[1]), (scores[1], tasks[2]),
        (scores[2], tasks[0]), (scores[2], tasks[1]), (scores[2], tasks[2]),
    ]
    col_tuples = [c for c in ordered_cols if c in all_r]

    R = pd.DataFrame(
        index=all_features,
        columns=pd.MultiIndex.from_tuples(col_tuples, names=["score", "task"]),
        dtype=float,
    )
    P = pd.DataFrame(
        index=all_features,
        columns=pd.MultiIndex.from_tuples(col_tuples, names=["score", "task"]),
        dtype=float,
    )

    for col in col_tuples:
        r_series = all_r.get(col)
        p_series = all_p.get(col)
        if r_series is not None:
            R.loc[r_series.index, col] = r_series.values
        if p_series is not None:
            P.loc[p_series.index, col] = p_series.values

    # sort features by mean |r|
    mean_abs = R.abs().mean(axis=1, skipna=True)
    R_sorted = R.loc[mean_abs.sort_values(ascending=False).index]
    P_sorted = P.loc[R_sorted.index]

    # R² matrix
    R2_sorted = R_sorted ** 2

    # save CSVs
    R_out = os.path.join(outdir, "feature_score_correlations_r.csv")
    R2_out = os.path.join(outdir, "feature_score_correlations_r2.csv")
    P_out = os.path.join(outdir, "feature_score_correlations_p_corr.csv")

    R_sorted.to_csv(R_out)
    R2_sorted.to_csv(R2_out)
    P_sorted.to_csv(P_out)

    print(f"[compute_all_correlations] saved r to: {R_out}")
    print(f"[compute_all_correlations] saved r² to: {R2_out}")
    print(f"[compute_all_correlations] saved p (Holm-corrected) to: {P_out}")

    # build heatmap data (r) + significance labels
    plot_data = R_sorted.copy()

    def stars(p):
        if pd.isna(p):
            return ""
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    annot = P_sorted.map(stars)

    cmap = plt.get_cmap("coolwarm").copy()
    center = 0
    vmin, vmax = -1, 1

    cmap.set_bad("white")

    plt.figure(figsize=(12, max(6, 0.22 * len(plot_data))))
    mask = plot_data.isna()

    ax = sns.heatmap(
        plot_data,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        annot=annot.astype(str),
        fmt="",
        annot_kws={"fontsize": 7, "ha": "center", "va": "center"},
        square=False,
    )

    xticklabels = [f"{score}\n{task}" for score, task in plot_data.columns]
    ax.set_xticklabels(xticklabels, rotation=90, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.subplots_adjust(top=0.90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel("Features")
    ax.set_xlabel("")

    plt.tight_layout()
    fig_path = os.path.join(
        outdir,
        f"correlations_heatmap.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[compute_all_correlations] saved heatmap to: {fig_path}")

    return R_sorted, R2_sorted, P_sorted


def plot_feature_vs_scores(
    task_name,
    feature_name,
    scores=SCORES,
    save_dir = CORR_OUTDIR,
):
    """
    Plot correlation of one feature for all three scores as panel
    """

    os.makedirs(save_dir, exist_ok=True)

    # load data
    features_path = os.path.join(FEATURES_DIR, f"{task_name}_cleaned.csv")
    features = pd.read_csv(features_path)
    scores_df = pd.read_csv(SCORES_PATH)

    for df in (features, scores_df):
        df[ID_COL] = df[ID_COL].astype(str)

    fig, axes = plt.subplots(1, len(scores), figsize=(20, 6), sharex=False, sharey=False)

    panel_labels = ["A", "B", "C"][: len(scores)]
    N_summary = []

    if len(scores) == 1:
        axes = [axes]

    for ax, target, label in zip(axes, scores, panel_labels):
        df = pd.merge(features, scores_df[[ID_COL, target]], on=ID_COL, how="inner")
        df = df[[feature_name, target]].dropna()

        # correlation
        x = df[feature_name].to_numpy()
        y = df[target].to_numpy()
        r, p = pearsonr(x, y)
        p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
        corr_text = rf"r = {r:.2f}, p {p_text}"

        # add jitter
        rng = np.random.default_rng(42)
        x_plot = x + rng.normal(loc=0.0, scale=0.2, size=len(x))
        y_plot = y

        # scatter + regression line
        sns.scatterplot(
            x=x_plot,
            y=y_plot,
            color="cadetblue",
            alpha=0.7,
            ax=ax,
        )
        sns.regplot(
            x=x,
            y=y,
            scatter=False,
            color="teal",
            ci=95,
            line_kws={"linewidth": 1.5},
            ax=ax,
        )

        ax.set_xlabel(feature_name, fontsize=25)
        ax.set_ylabel(format_title(target), fontsize=25)
        ax.tick_params(axis="both", labelsize=21)
        sns.despine(ax=ax)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_locator(MultipleLocator(5))

        ax.text(
            0.02,
            0.98,
            corr_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=23,
            style="italic",
            color="teal",
            fontfamily="Arial",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.3",
            ),
            zorder=10,
        )

        N_summary.append([label, target, len(df)])

    for ax, label in zip(axes, panel_labels):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=23,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    plt.tight_layout()
    fname = f"{feature_name}_scores_correlation_{task_name}.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()

    N_table = pd.DataFrame(N_summary, columns=["Panel", "Score", "N"])
    print(f"[plot_feature_vs_scores] saved plot to: {out_path}")
    print(N_table.to_string(index=False))

    return out_path, N_table


def plot_top_k_features_by_mean_abs_r(
    task_name,
    scores=SCORES,
    k = 1,
    corr_outdir = FEATURE_CORR_DIR,
    save_dir = FEATURE_CORR_DIR,
):
    """
    Automatically select and plot top-k features (by mean |r|)
    """

    os.makedirs(corr_outdir, exist_ok=True)

    corr_path = os.path.join(corr_outdir, f"{task_name}_feature_score_correlations.csv")
    if os.path.exists(corr_path):
        corr_results = pd.read_csv(corr_path)
        print(f"[plot_top_k_features_by_mean_abs_r] loaded existing correlations from: {corr_path}")
    else:
        corr_results = compute_correlations_for_task(
            task_name, scores=scores, outdir=corr_outdir
        )

    # compute mean |r| across the chosen scores
    r_cols = [f"r_{s}" for s in scores]
    r_matrix = corr_results[r_cols].values
    mean_abs_r = np.nanmean(np.abs(r_matrix), axis=1)

    corr_results["mean_abs_r"] = mean_abs_r
    corr_results_sorted = corr_results.sort_values("mean_abs_r", ascending=False)

    top_features = corr_results_sorted["feature"].head(k).tolist()
    print(f"[plot_top_k_features_by_mean_abs_r] top {k} features for {task_name}: {top_features}")

    results = []
    for feat in top_features:
        plot_path, N_table = plot_feature_vs_scores(
            task_name=task_name,
            feature_name=feat,
            scores=scores,
            save_dir=save_dir,
        )
        results.append({"feature": feat, "plot_path": plot_path, "N_table": N_table})

    return results


def compute_score_correlations(
    scores=SCORES,
    outdir = SCORE_CORR_DIR,
):
    """
    Calculate correlations between language test scores
    """

    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(SCORES_PATH)[[ID_COL, *scores]].dropna()
    df[ID_COL] = df[ID_COL].astype(str)
    n = df.shape[0]

    # pairwise correlations
    pairs = []
    p_values = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            s1, s2 = scores[i], scores[j]
            r, p = pearsonr(df[s1], df[s2])
            pairs.append({"score_x": s1, "score_y": s2, "r": r, "p_raw": p})
            p_values.append(p)

    # Holm correction across the 3 pairs
    if p_values:
        _, p_corr, _, _ = multipletests(p_values, method="holm")
    else:
        p_corr = []

    for pair, p_c in zip(pairs, p_corr):
        pair["p_corr"] = p_c
        pair["r2"] = pair["r"] ** 2
        pair["n"] = n

    score_correlations = pd.DataFrame(pairs)
    save_path = os.path.join(outdir, "score_correlations.csv")
    score_correlations.to_csv(save_path, index=False)
    print(f"[compute_score_correlations] saved score correlations to: {save_path}")

    # matrix for heatmap
    R = df[scores].corr(method="pearson")
    matrix_path = os.path.join(outdir, "score_correlations_matrix.csv")
    R.to_csv(matrix_path)
    print(f"[compute_score_correlations] saved correlation matrix to: {matrix_path}")

    # heatmap with stars
    def stars(p):
        if pd.isna(p):
            return ""
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # build p-matrix for annotation
    p_mat = pd.DataFrame(np.ones_like(R.values), index=R.index, columns=R.columns, dtype=float)
    for _, row in score_correlations.iterrows():
        s1, s2 = row["score_x"], row["score_y"]
        p_c = row["p_corr"]
        p_mat.loc[s1, s2] = p_c
        p_mat.loc[s2, s1] = p_c

    annot = p_mat.map(stars)

    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    ax = sns.heatmap(
        R,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annot.astype(str),
        fmt="",
        annot_kws={"fontsize": 10, "ha": "center", "va": "center"},
        cbar_kws={"label": "Pearson r"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    fig_path = os.path.join(outdir, "score_correlations_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[compute_score_correlations] saved score correlation heatmap to: {fig_path}")

    # pairwise scatter plots with regression lines
    n_pairs = len(score_correlations)
    if n_pairs > 0:
        fig, axes = plt.subplots(1, n_pairs, figsize=(20, 6), sharex=False, sharey=False)
        if n_pairs == 1:
            axes = [axes]

        panel_labels = ["A", "B", "C"][:n_pairs]

        for ax, (_, row), label in zip(axes, score_correlations.iterrows(), panel_labels):
            s1, s2, r, p_c = row["score_x"], row["score_y"], row["r"], row["p_corr"]

            tmp = df[[s1, s2]].dropna()
            X = tmp[s1].to_numpy()
            Y = tmp[s2].to_numpy()

            # add jitter
            rng = np.random.default_rng(42)
            x_plot = X + rng.normal(loc=0.0, scale=0.2, size=len(X))
            y_plot = Y

            sns.scatterplot(
                x=x_plot,
                y=y_plot,
                alpha=0.7,
                ax=ax,
                color="cadetblue",
            )
            sns.regplot(
                x=X,
                y=Y,
                ax=ax,
                scatter=False,
                color="teal",
                ci=95,
                line_kws={"linewidth": 1.5},
            )

            ax.set_xlabel(format_title(s1), fontsize=25)
            ax.set_ylabel(format_title(s2), fontsize=25)
            ax.tick_params(axis="both", labelsize=21)
            sns.despine(ax=ax)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

            p_text = "< .001" if p_c < 0.001 else f"= {p_c:.3f}"
            text = rf"r = {r:.2f}, p {p_text}"
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=23,
                style="italic",
                color="teal",
                fontfamily="Arial",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
            )

            ax.text(
                -0.1,
                1.05,
                label,
                transform=ax.transAxes,
                fontsize=23,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        plt.tight_layout()
        scatter_path = os.path.join(outdir, "score_correlations_scatterplots.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[compute_score_correlations] saved score correlation scatter plots to: {scatter_path}")

    return score_correlations, R


def compute_demographic_score_correlations(
    scores=SCORES,
    outdir = DEMO_CORR_DIR,
    demographics = None,
):
    """
    Correlations between demographics and language test scores
    """
    os.makedirs(outdir, exist_ok=True)

    if demographics is None:
        demographics = load_demographics()

    scores_df = pd.read_csv(SCORES_PATH)[[ID_COL, *scores]]

    for df in (demographics, scores_df):
        df[ID_COL] = df[ID_COL].astype(str)
    df = pd.merge(demographics, scores_df, on=ID_COL, how="inner").dropna()

    demo_cols = [c for c in demographics.columns if c != ID_COL]

    rows = []
    p_values = []
    for demo in demo_cols:
        for score in scores:
            r, p = pearsonr(df[demo], df[score])
            rows.append(
                {
                    "demographic": demo,
                    "score": score,
                    "r": r,
                    "p_raw": p,
                }
            )
            p_values.append(p)

    if p_values:
        _, p_corr, _, _ = multipletests(p_values, method="holm")
    else:
        p_corr = []

    for row, p_c in zip(rows, p_corr):
        row["p_corr"] = p_c
        row["r2"] = row["r"] ** 2
        row["n"] = len(df)

    demographic_correlations = pd.DataFrame(rows)
    out_path = os.path.join(outdir, "demographics_score_correlations.csv")
    demographic_correlations.to_csv(out_path, index=False)
    print(f"[compute_demographic_score_correlations] saved results to: {out_path}")

    return demographic_correlations


# run all correlation analyses
def run_correlation_analyses(
    tasks=TASKS,
    scores=SCORES,
):
    print("\n[correlations] calculating per-task correlations for all tasks")
    for task in tasks:
        print(f"\n[correlations] task: {task}")
        compute_correlations_for_task(task, scores=scores)

    print("\n[correlations] calculating combined correlations across tasks")
    R_sorted, R2_sorted, P_sorted = compute_all_correlations(
        tasks=tasks,
        scores=scores,
    )

    # find best feature (and task) by mean |r| across scores
    R_abs = R_sorted.abs()
    mean_by_feat_task = R_abs.groupby(level="task", axis=1).mean()
    best_feature, best_task = mean_by_feat_task.stack().idxmax()

    print(f"\n[correlations] global best combination: feature = {best_feature}, task = {best_task}")

    # plot this feature for the best task, with all three scores as panels
    plot_feature_vs_scores(
        task_name=best_task,
        feature_name=best_feature,
        scores=scores,
        save_dir=FEATURE_CORR_DIR,
    )

    print("\n[correlations] calculating correlations between language test scores")
    compute_score_correlations(scores=scores)

    print("\n[correlations] calculating correlations between demographics and language scores")
    compute_demographic_score_correlations(scores=scores)

    print("\n[correlations] all correlation analyses done.")


if __name__ == "__main__":
    run_correlation_analyses()
