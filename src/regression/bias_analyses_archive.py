### bias analyses

# unpaired bootstrapping for comparison of groups
def bootstrap_r2_unpaired(oof_sub, n_boot=default_nboot, random_state=default_randomstate):
    # generate random numbers
    rng = np.random.default_rng(random_state)
    # index rows per subject
    idx_by_sid = oof_sub.groupby("Subject_ID").indices
    # subjects in the group
    subj = np.array(list(idx_by_sid.keys()))
    n = len(subj) # -> resample same size as original group
    # output array: one R^2 per bootstrap draw
    r2s = np.empty(n_boot, dtype=float)
    y = oof_sub["y_true"].to_numpy()
    ypred = oof_sub["y_pred"].to_numpy()
    # start bootstrap draw, calculate R²
    for b in range(n_boot):
        boot_ids = rng.choice(subj, size=n, replace=True)
        idxs = np.concatenate([idx_by_sid[s] for s in boot_ids])
        r2s[b] = r2_score(y[idxs], ypred[idxs])
    return r2s # contains bootstrapped R²-values for this group

# compare groups for one target, task and model with unpaired bootstrapping (for bias: test if two groups are different)
def compare_groups(
    oof_preds,
    group_col,  # select variable to split on (Gender, Country, Age)
    levels,  # allowed levels and order (for example: "f", "m" or "uk", "usa")
    task,
    target,
    model="full",
    n_boot=default_nboot,
    ci=default_ci,
    random_state=default_randomstate,
    pairs_to_test=None # select what levels to test against each other, if None -> all
):
    # subset
    df = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target) &
        (oof_preds[group_col].isin(levels))
    ].copy()

    # build list of pairs
    levels = list(levels)
    if pairs_to_test is None:
        from itertools import combinations
        pairs = list(combinations(levels, 2))
    else:
        pairs = [tuple(p) for p in pairs_to_test]

    rows = []

    # loop over pairs
    for k, (A, B) in enumerate(pairs):
        oofA = df[df[group_col] == A]
        oofB = df[df[group_col] == B]

        # observed statistic
        def r2_from_group(df):
            return r2_score(df["y_true"].to_numpy(), df["y_pred"].to_numpy())

        obs_diff = r2_from_group(oofA) - r2_from_group(oofB)

        # unpaired bootstrapping: resample subjects for each group
        if oofA.empty or oofB.empty:
            rows.append({"A": A, "B": B, "mean_diff": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan,
                         "p_raw": np.nan, "p_adj": np.nan,
                         "n_boot": n_boot, "ci": ci})
            continue

        # unpaired: independent subject resampling for each group
        r2A = bootstrap_r2_unpaired(oofA, n_boot=n_boot, random_state=random_state + 2*k)
        r2B = bootstrap_r2_unpaired(oofB, n_boot=n_boot, random_state=random_state + 2*k + 1)
        diff = r2A - r2B

        mean_d  = float(np.nanmean(diff))
        ci_low, ci_high = confidence_intervals(diff, ci=ci)
        p_boot = float(p_value_bootstrap(diff, obs_diff))

        rows.append({"A": A, "B": B, "mean_diff": mean_d,
                     "ci_low": ci_low, "ci_high": ci_high,
                     "p_raw": p_boot, "p_adj": None,
                     "n_boot": n_boot, "ci": ci})

    # Holm-Bonferroni-correction across pairs
    df_out = pd.DataFrame(rows)
    df_out["p_adj"] = adjust_pvals(df_out["p_raw"].to_numpy(), method="holm")

    cols = ["A","B","mean_diff","ci_low","ci_high", "p_raw","p_adj","n_boot","ci"]
    return df_out[cols].sort_values(["p_adj","A","B"])

# save tables for group-comparisons (bootstrapped R^2 and p-values)
def save_group_tables(
    oof_preds,
    target,
    task,
    model="full",
    group_col=None,
    levels=None,
    n_boot=default_nboot,
    ci=default_ci,
    random_state=default_randomstate,
    outdir=None,
    save_metrics=True
):
    # subset
    levels = list(levels)
    sub = oof_preds[
        (oof_preds["task"]==task) &
        (oof_preds["model"]==model) &
        (oof_preds["target"]==target) &
        (oof_preds[group_col].isin(levels))
    ].copy()

    sub[group_col] = sub[group_col].astype(CategoricalDtype(levels, ordered=True))

    # R² draws per group + summary
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=(group_col,),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=None
    )

    r2_summary = (
        boot_df.groupby(group_col, observed=True)["r2"]
        .agg(r2_mean="mean",
             r2_ci_low=lambda s: confidence_intervals(s, ci=ci)[0],
             r2_ci_high=lambda s: confidence_intervals(s, ci=ci)[1])
        .reset_index()
    )

    n_by_group = (sub.groupby(group_col, observed=True)["Subject_ID"]
                    .nunique().rename("n_subjects")).reset_index()
    r2_summary = (r2_summary.merge(n_by_group, on=group_col, how="left")
                              .assign(n_boot=n_boot))
    r2_summary.insert(0, "target", target)
    r2_summary.insert(0, "task", task)
    r2_summary.insert(0, "model", model)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    if outdir:
        r2_path = os.path.join(outdir, f"r2_{group_col}_{task}_{model}_{target}.csv")
        r2_summary.to_csv(r2_path, index=False)

    # add other metrics (RMSE and MAE) (optional)
    if save_metrics:
        met = bootstrap_metrics_from_oof(
            sub, group_cols=(group_col,), n_boot=n_boot, ci=ci, random_state=random_state
        )
        met[group_col] = pd.Categorical(met[group_col], categories=levels, ordered=True)
        met = met.sort_values([group_col, "metric"])
        met.insert(0, "target", target)
        met.insert(0, "task", task)
        met.insert(0, "model", model)
        if outdir:
            metrics_path = os.path.join(outdir, f"metrics_{group_col}_{task}_{model}_{target}.csv")
            met.to_csv(metrics_path, index=False)

    # add statistical comparisons (ΔR²)
    tests = compare_groups(
        oof_preds=oof_preds,
        group_col=group_col,
        levels=levels,
        task=task,
        target=target,
        model=model,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        pairs_to_test=None,
    )
    tests.insert(0, "target", target)
    tests.insert(0, "task", task)
    tests.insert(0, "model", model)
    tests.insert(0, "group_col", group_col)
    if outdir:
        pval_path = os.path.join(outdir, f"statistical_comparison_{group_col}_{task}_{model}_{target}.csv")
        tests.to_csv(pval_path, index=False)

    return r2_summary, tests


### bias plotting helpers

# create 3 same-size age groups
def make_equal_count_age_groups(df, age_col="Age", label_col="AgeGroup"):
    if age_col not in df.columns:
        raise KeyError(f"Column '{age_col}' not found.")
    ages = pd.to_numeric(df[age_col], errors="coerce")
    if ages.isna().all():
        raise ValueError(f"Column '{age_col}' has no numeric values.")
    # 3 equal-count bins
    q = np.quantile(ages.dropna(), [0.0, 1/3, 2/3, 1.0])

    lows  = np.floor(q[:-1]).astype(int)
    highs = np.ceil(q[1:]).astype(int)

    # ensure strictly increasing bin edges
    edges = q.copy()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9

    labels = [f"{l}–{h}" for l, h in zip(lows, highs)]

    df2 = df.copy()
    df2[label_col] = pd.cut(ages, bins=edges, include_lowest=True, right=True)
    # rename categories
    df2[label_col] = df2[label_col].cat.rename_categories(labels)
    return df2, labels

# boxplots of bootstrapped R^2 per group and p-values for group comparisons
def plot_group_box_with_pvals(
    oof_preds,
    target,
    group_col,
    levels,
    task,
    model="full",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    save_dir=None,
    pairs_to_show=None,
    box_width=0.24,
    ax=None,
    ymin=None, ymax=None, # y axis limits
    bracket_scale=2.5, # spacing between brackets
    ceil_pad=0.10, # space above top p-bracket
    x_label=None
):
    # select groups and set order
    if levels is None:
        raise ValueError("provide levels")
    levels = list(levels) if isinstance(levels, (list, tuple)) else [levels]

    df_src = oof_preds
    sub = df_src[
        (df_src["task"] == task) &
        (df_src["model"] == model) &
        (df_src["target"] == target) &
        (df_src[group_col].isin(levels))
        ].copy()

    sub[group_col] = sub[group_col].astype(CategoricalDtype(categories=levels, ordered=True))

    # bootstrapped draws of R^2 per level
    boot_df, _ = bootstrap_summary_from_oof(
        sub, group_cols=(group_col,),
        n_boot=n_boot, ci=ci, random_state=random_state, subject_set=None
    )
    boot_df = boot_df[boot_df[group_col].isin(levels)].copy()
    boot_df[group_col] = boot_df[group_col].astype(CategoricalDtype(categories=levels, ordered=True))

    pal = bias_palette(group_col, levels)

    # for subplots later
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        created_fig = True
    else:
        fig = ax.figure

    # boxplot
    sns.boxplot(
        data=boot_df, x=group_col, y="r2",
        order=levels, hue=group_col, hue_order=levels, legend=False,
        palette=pal, dodge=False, width=box_width,
        showcaps=True, showfliers=False,
        boxprops={"alpha": 1, "edgecolor": "black"},
        whiskerprops={"alpha": 1, "color": "black"},
        capprops={"alpha": 1, "color": "black"},
        whis=(2.5, 97.5), showmeans=True, meanline=True,
        meanprops=dict(color="#111", linewidth=0.9, linestyle="-"),
        medianprops=dict(color="none", linewidth=0),
        linewidth=0.8, ax=ax
    )

    ax.margins(x=0.12)
    # labels
    ticks = np.arange(len(levels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(format_bias_labels(group_col, levels))
    label = x_label if x_label is not None else format_title(group_col.replace("_label", ""))
    ax.set_xlabel(label, fontsize=12)  # <-- uses override when provided
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


    # compute tests (unpaired bootstrap differences, Holm-Bonferroni-corrected p-values)
    all_pairs = list(combinations(levels, 2))
    tests_all = compare_groups(
        oof_preds=oof_preds,
        group_col=group_col,
        levels=levels,
        task=task,
        target=target,
        model=model,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
        pairs_to_test=all_pairs,  # adjust over full family
    )

    # show only selected pairs (if any), but keep p_adj from tests_all
    if pairs_to_show is None:
        tests_df = tests_all.copy()
    else:
        want = {tuple(sorted(p)) for p in pairs_to_show}
        tests_df = tests_all[
            tests_all.apply(lambda r: tuple(sorted((r["A"], r["B"]))) in want, axis=1)
        ].copy()

    # calculate y-axis-limits if not provided
    if (ymin is not None) or (ymax is not None):
        cur_lo, cur_hi = ax.get_ylim()
        lo = ymin if ymin is not None else cur_lo
        hi = ymax if ymax is not None else cur_hi
        ax.set_ylim(lo, hi)

    lo, hi = ax.get_ylim()
    axis_range = hi - lo
    height = 0.012 * axis_range
    y_pad = (0.008 * bracket_scale) * axis_range
    rung_gap = (1.8 * bracket_scale) * height

    x_centers = ax.get_xticks()
    idx = {lvl: i for i, lvl in enumerate(levels)}
    ymax_by_level = (boot_df.groupby(group_col, observed=True)["r2"]
                     .quantile(0.98)  # try 0.95–0.98; lower = closer
                     .reindex(levels)
                     .to_numpy())

    # draw brackets with p-values
    # --- draw brackets with p-values (non-overlapping) ---
    if (tests_df is not None) and (len(tests_df) > 0):
        tests_df = tests_df.copy()
        tests_df["span"] = tests_df.apply(lambda r: abs(idx[r["A"]] - idx[r["B"]]), axis=1)

        placed = []  # store placed brackets as (i_min, i_max, y_top)

        for _, r in tests_df.sort_values("span").iterrows():
            A, B, p_adj = r["A"], r["B"], r["p_adj"]
            i, j = idx[A], idx[B]
            if i > j:
                i, j = j, i  # ensure i < j for interval logic

            # base height just above the 98th percentile of the two boxes
            y_base = max(ymax_by_level[i], ymax_by_level[j]) + y_pad
            y_top = y_base + height

            # bump up until no horizontal-overlap with already placed brackets
            def overlaps(a1, a2, b1, b2):
                # closed-interval overlap on x (indices)
                return not (a2 < b1 or b2 < a1)

            # keep bumping while colliding
            bumped = True
            while bumped:
                bumped = False
                for (pi, pj, py_top) in placed:
                    if overlaps(i, j, pi, pj):
                        # if current top is too close to an existing top, push up
                        if y_base < (py_top + rung_gap):
                            y_base = (py_top + rung_gap)
                            y_top = y_base + height
                            bumped = True

            # draw the bracket at the resolved height
            ax.plot([x_centers[i], x_centers[i], x_centers[j], x_centers[j]],
                    [y_base, y_top, y_top, y_base],
                    lw=0.8, color="#333")

            ax.text((x_centers[i] + x_centers[j]) / 2.0, y_top,
                    format_p_label(p_adj, stars=True, decimals=3,
                                   drop_leading_zero=True, use_lt_for_small=True, n_boot=n_boot),
                    ha="center", va="bottom", fontsize=8)

            # remember this bracket
            placed.append((i, j, y_top))

            # also nudge local maxima so future brackets start above
            ymax_by_level[i] = max(ymax_by_level[i], y_top + (0.2 * rung_gap))
            ymax_by_level[j] = max(ymax_by_level[j], y_top + (0.2 * rung_gap))

        # give headroom for the tallest bracket
        cur_bot, cur_top = ax.get_ylim()
        peak = max([py for (_, _, py) in placed], default=cur_top)
        maybe_top = max(cur_top, peak + ceil_pad * (cur_top - cur_bot))
        if maybe_top > cur_top:
            ax.set_ylim(cur_bot, maybe_top)

    plt.tight_layout()
    if save_dir and created_fig:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{group_col}_boxplot_{target}.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")

    if created_fig:
        plt.show()
    return ax

# show all three boxplots in one plot for bias analyses
def plot_bias_panels_ABC(
    oof_preds,
    target,
    task,
    model="full",
    save_path=None,
    bracket_scale_c=1.9
):
    # shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=False)

    # A: Gender
    plot_group_box_with_pvals(
        oof_preds, target, "Gender_label", ("f","m"),
        task=task, model=model, ax=axes[0],
        box_width=0.24,
        bracket_scale=3, ceil_pad=0.06
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # B: Country
    plot_group_box_with_pvals(
        oof_preds, target, "Country_label", ("uk","usa"),
        task=task, model=model, ax=axes[1],
        box_width=0.24,
        bracket_scale=3, ceil_pad=0.06
    )
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # C: Age group
    age_df, age_levels = make_equal_count_age_groups(oof_preds, age_col="Age", label_col="AgeGroup")
    plot_group_box_with_pvals(
        age_df, target, "AgeGroup", age_levels,
        task=task, model=model, ax=axes[2],
        pairs_to_show=[(age_levels[0], age_levels[1]),
                       (age_levels[1], age_levels[2]),
                       (age_levels[0], age_levels[2])],
        box_width=0.24, bracket_scale=bracket_scale_c, ceil_pad=0.02
    )
    axes[2].text(0.02, 0.98, "C", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # unify y-axis across A, B, C
    y0_lo, y0_hi = axes[0].get_ylim()
    y1_lo, y1_hi = axes[1].get_ylim()
    y2_lo, y2_hi = axes[2].get_ylim()
    shared_lo = min(y0_lo, y1_lo, y2_lo)
    shared_hi = max(y0_hi, y1_hi, y2_hi)
    for ax_ in axes:
        ax_.set_ylim(shared_lo, shared_hi)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return fig, axes


# panels variant using 2-group age split
def plot_bias_panels_AB_age2(
    oof_preds,
    target,
    task,
    model="full",
    save_path=None,
    include_equal_65_in_upper=True
):

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=False)

    # A: Gender
    plot_group_box_with_pvals(
        oof_preds, target, "Gender_label", ("f","m"),
        task=task, model=model, ax=axes[0],
        box_width=0.24, bracket_scale=3, ceil_pad=0.06
    )
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # B: Country
    plot_group_box_with_pvals(
        oof_preds, target, "Country_label", ("uk","usa"),
        task=task, model=model, ax=axes[1],
        box_width=0.24, bracket_scale=3, ceil_pad=0.06
    )
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # C: Age (2 groups)
    age_df, age_levels = make_age_binary_groups(
        oof_preds, age_col="Age", label_col="AgeGroup2",
        include_equal_65_in_upper=include_equal_65_in_upper
    )
    # ensure order: "<65" first, then "≥65" or ">65"
    if include_equal_65_in_upper:
        age_levels = ["<65", "≥65"]
    else:
        age_levels = ["<65", ">65"]

    plot_group_box_with_pvals(
        age_df, target, "AgeGroup2", age_levels,
        task=task, model=model, ax=axes[2],
        pairs_to_show=[tuple(age_levels)],  # only one comparison
        box_width=0.24, bracket_scale=2.2, ceil_pad=0.04
    )
    axes[2].text(0.02, 0.98, "C", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=12, fontweight="bold")

    # unify y-axis
    y0_lo, y0_hi = axes[0].get_ylim()
    y1_lo, y1_hi = axes[1].get_ylim()
    y2_lo, y2_hi = axes[2].get_ylim()
    shared_lo = min(y0_lo, y1_lo, y2_lo)
    shared_hi = max(y0_hi, y1_hi, y2_hi)
    for ax_ in axes:
        ax_.set_ylim(shared_lo, shared_hi)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return fig, axes



