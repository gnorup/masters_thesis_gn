import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from config.constants import GIT_DIRECTORY, ID_COL, SCORES
from regression.plots import format_title

# set font for all plots
plt.rcParams['font.family'] = 'Arial'
sns.set(style="whitegrid")


def plot_score_distributions(
    scores=SCORES,
    scores_path=None,
    outdir=None,
):
    """
    Plot distributions for language test scores
    """
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(scores_path)[[ID_COL, *scores]].dropna()
    df[ID_COL] = df[ID_COL].astype(str)

    score_order = [
        "PictureNamingScore",
        "SemanticFluencyScore",
        "PhonemicFluencyScore",
    ]
    score_order = [s for s in score_order if s in scores]

    fig, axes = plt.subplots(1, len(score_order), figsize=(20, 6), sharey=True)
    if len(score_order) == 1:
        axes = [axes]

    for ax, score in zip(axes, score_order):
        s = df[score].dropna().astype(float)

        # bin verbal fluency scores so they match Picture Naming visually
        if score == "PictureNamingScore":
            bin_width = 1
        else:
            bin_width = 2  # 2-point bins for fluency

        min_score = int(np.floor(s.min()))
        max_score = int(np.ceil(s.max()))
        bins = np.arange(min_score, max_score + bin_width, bin_width)

        # bin the values
        s_binned = pd.cut(s, bins=bins, right=True, include_lowest=True)
        counts = s_binned.value_counts().sort_index()

        x_vals = np.array([interval.left for interval in counts.index])
        y_vals = counts.values

        # bar plots
        ax.bar(
            x_vals,
            y_vals,
            width=bin_width,
            align="center",
            edgecolor="black",
            color="cadetblue",
        )

        xmax = max(x_vals)
        ax.set_xlim(-0.5, xmax + 0.5)
        step = 5
        xticks = np.arange(0, xmax + 1, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(t)) for t in xticks], fontsize=24)

        ax.set_xlabel(format_title(score), fontsize=28)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(False, axis="x")

        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")
            ax.spines[side].set_linewidth(0.5)
            ax.spines[side].set_visible(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Number of Participants", fontsize=28)

    plt.tight_layout()
    fig_path = os.path.join(outdir, "score_distributions.png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"saved score distribution panels to: {fig_path}")


def plot_distributions(df, columns, save=False, save_dir=None, id_column="Subject_ID"):
    """
    Plot feature or score distributions and show IQR-based outliers.
    """
    if save and save_dir is None:
        raise ValueError("save=True but save_dir is None. Please provide a save_dir.")

    for col in columns:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")

        safe_colname = col.replace("/", "_")

        if save:
            save_path = os.path.join(save_dir, f"{safe_colname}_distribution.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"saved to {save_path}")
        else:
            plt.tight_layout()

        plt.show()
        plt.close()

        # detect outliers based on IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"outliers in '{col}':")
            print(outliers[[id_column, col]])
