# SHAP feature importance for Random Forest regression

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from config.constants import GIT_DIRECTORY, TASKS, SCORES, ID_COL
from config.feature_sets import get_linguistic_features, get_acoustic_features, get_demographic_features
from data_preparation.data_handling import (
    load_demographics,
    load_task_dataframe,
    complete_subjects,
    build_feature_sets,
)
from regression.hyperparameter_tuning import load_tuned_rf_params_regression
from regression.plots import format_title

# plot style
plt.style.use("default")
plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.family": "Arial",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# for superscripts
linguistic = get_linguistic_features()
acoustic = get_acoustic_features()
demographic = get_demographic_features()

superscripts = {
    "linguistic": "¹",
    "acoustic": "²",
    "demographics": "³",
}

def feature_category(name):
    if name in linguistic:
        return "linguistic"
    if name in acoustic:
        return "acoustic"
    if name in demographic:
        return "demographics"
    return "other"

def feature_superscript(name):
    category = feature_category(name)
    return superscripts.get(category, "")


# random forest regression: feature-importance (global SHAP values)
def stratified_cv_feature_importance(
    df,
    fold_column,
    model_type,
    model_params,
    target_column,
    feature_columns,
    save_dir=None,
    task_name=None
):

    shap_values_all = []
    base_values_all = []
    X_test_all = []
    shap_explanation = None
    shap_table = None

    for fold in sorted(df[fold_column].unique()):
        train_df = df[df[fold_column] != fold]
        test_df = df[df[fold_column] == fold]

        X_train = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
        y_train = train_df[target_column]
        y_test = test_df[target_column]

        # coerce everything to numeric
        X_train = X_train.apply(pd.to_numeric, errors="coerce")
        X_test = X_test.apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(y_train, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")

        # drop rows with any missing
        tr_mask = X_train.notna().all(axis=1) & y_train.notna()
        te_mask = X_test.notna().all(axis=1) & y_test.notna()
        X_train, y_train = X_train.loc[tr_mask], y_train.loc[tr_mask]
        X_test, y_test = X_test.loc[te_mask], y_test.loc[te_mask]

        X_train_np = X_train.to_numpy(dtype="float64")
        X_test_np = X_test.to_numpy(dtype="float64")

        # train model
        model = model_type(**(model_params or {}))
        model.fit(X_train_np, y_train.to_numpy(dtype="float64"))

        # SHAP: TreeExplainer
        explainer = shap.TreeExplainer(model, X_train_np)
        shap_vals = explainer(X_test_np)

        shap_values_all.append(shap_vals.values)
        base_values_all.append(np.array(shap_vals.base_values).reshape(-1))
        X_test_all.append(X_test)

    # create aggregated SHAP explanation object
    if shap_values_all:
        all_shap_values = np.vstack(shap_values_all)
        all_base_values = np.concatenate(base_values_all)
        all_X_test = pd.concat(X_test_all, ignore_index=True)
        shap_explanation = shap.Explanation(
            values=all_shap_values,
            base_values=all_base_values,
            data=all_X_test.values,
            feature_names=all_X_test.columns.tolist()
        )

        # save mean absolute SHAP values
        shap_table = pd.DataFrame({
            "Feature": all_X_test.columns,
            "Mean_Absolute_SHAP_Value": np.abs(all_shap_values).mean(axis=0)
        }).sort_values("Mean_Absolute_SHAP_Value", ascending=False).reset_index(drop=True)
        if save_dir and task_name:
            shap_table.to_csv(os.path.join(save_dir, f"{target_column}_{task_name}_global_shap_values.csv"), index=False)
    else:
        print("no SHAP values available")

    return shap_explanation, shap_table

# compute SHAP values for the full model for all task and score combinations
def run_shap_full_models(shap_dir=None):
    if shap_dir is None:
        shap_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest", "feature_importance")
    os.makedirs(shap_dir, exist_ok=True)

    # get tuned RF hyperparameters
    rf_params = load_tuned_rf_params_regression()

    # load data
    scores_path = os.path.join(GIT_DIRECTORY, "data", "language_scores_all_subjects.csv")
    scores_df = pd.read_csv(scores_path)
    demographics = load_demographics()

    for target in SCORES:
        print(f"\ncalculating SHAP values for target: {target}")
        for task in TASKS:
            print(f"\n -> task: {task}")

            # load dataframe
            df_task = load_task_dataframe(task, target, scores_df, demographics)

            # build feature sets for this dataframe, then pick full model
            model_features = build_feature_sets(df_task.columns)
            full_features = model_features["full"]

            # restrict to subjects with complete full-model features
            full_subjects = complete_subjects(df_task, full_features, target)
            print(f"full-model complete subjects: N={len(full_subjects)}")

            df_use = df_task[df_task[ID_COL].isin(full_subjects)].copy()
            df_use = df_use.dropna(subset=[target] + full_features)

            # ensure numeric & drop missing
            for col in full_features:
                df_use[col] = pd.to_numeric(df_use[col], errors="coerce")
            df_use[target] = pd.to_numeric(df_use[target], errors="coerce")
            df_use = df_use.dropna(subset=[target] + full_features)

            save_dir = os.path.join(shap_dir, target, task)
            os.makedirs(save_dir, exist_ok=True)

            # run SHAP CV importance
            shap_expl, shap_table = stratified_cv_feature_importance(
                df=df_use,
                fold_column="fold",
                model_type=RandomForestRegressor,
                model_params=rf_params,
                target_column=target,
                feature_columns=full_features,
                save_dir=save_dir,
                task_name=task,
            )


# create heatmap of mean absolute SHAP values for all features from all task-score-combinations
def build_shap_heatmap(shap_dir=None, out_dir=None):

    if shap_dir is None:
        shap_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest", "feature_importance")

    if out_dir is None:
        out_dir = os.path.join(shap_dir, "shap_heatmap")
    os.makedirs(out_dir, exist_ok=True)

    ORDERED_COLS = [
        ("PictureNamingScore", "picnicScene"),
        ("PictureNamingScore", "cookieTheft"),
        ("PictureNamingScore", "journaling"),
        ("SemanticFluencyScore", "picnicScene"),
        ("SemanticFluencyScore", "cookieTheft"),
        ("SemanticFluencyScore", "journaling"),
        ("PhonemicFluencyScore", "picnicScene"),
        ("PhonemicFluencyScore", "cookieTheft"),
        ("PhonemicFluencyScore", "journaling"),
    ]

    all_values = {}
    for target in SCORES:
        for task in TASKS:
            table_path = os.path.join(shap_dir, target, task, f"{target}_{task}_global_shap_values.csv")
            if not os.path.exists(table_path):
                print(f"[heatmap] missing table for {target} / {task}")
                continue

            table = pd.read_csv(table_path)
            if "Feature" not in table.columns or "Mean_Absolute_SHAP_Value" not in table.columns:
                print(f"[heatmap] invalid columns in {table_path}")
                continue

            s = pd.Series(
                table["Mean_Absolute_SHAP_Value"].values,
                index=table["Feature"].values,
            )
            all_values[(target, task)] = s

    if not all_values:
        print("[heatmap] no SHAP tables found.")
        return

    all_features = sorted(set().union(*[s.index for s in all_values.values()]))

    cols = pd.MultiIndex.from_tuples(ORDERED_COLS, names=["Score", "Task"])
    mat = pd.DataFrame(index=all_features, columns=cols, dtype=float)

    for (score, task), s in all_values.items():
        if (score, task) in mat.columns:
            mat.loc[s.index, (score, task)] = s.values

    # sort rows by overall mean
    mat["__absmean__"] = mat.mean(axis=1, skipna=True)
    mat = mat.sort_values("__absmean__", ascending=False).drop(columns="__absmean__")

    # plot
    flat_cols = [f"{sc}\n{ta}" for (sc, ta) in mat.columns.to_list()]
    plot_df = mat.copy()
    plot_df.columns = flat_cols

    plot_df.index = [f"{f}{feature_superscript(f)}" for f in plot_df.index]
    plot_df = plot_df.astype(float)

    vmax = np.nanpercentile(plot_df.values, 99)

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="white")
    mask = plot_df.isna()

    plt.figure(figsize=(14, max(6, 0.28 * len(plot_df))))
    ax = sns.heatmap(
        plot_df,
        cmap=cmap,
        vmin=0.0,
        vmax=float(vmax),
        linewidths=0,
        linecolor=None,
        cbar_kws={"label": "Mean |SHAP|"},
        square=False,
        mask=mask,
    )

    plt.gcf().patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="left")

    ax.set_xlabel("")
    ax.set_ylabel("Feature")
    ax.text(
        0.0, -0.035,
        "¹ linguistic   ² acoustic   ³ demographics",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontfamily="Arial",
    )

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    png_path = os.path.join(out_dir, "shap_heatmap.png")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()
    print("saved heatmap to:", png_path)


# prepare SHAP explanation for the full model of one task-score combination -> plots
def prepare_full_model_shap(task_name, target):

    # load data
    scores_path = os.path.join(GIT_DIRECTORY, "data", "language_scores_all_subjects.csv")
    scores_df = pd.read_csv(scores_path)
    demographics = load_demographics()

    # build task dataframe
    df_task = load_task_dataframe(task_name, target, scores_df, demographics)

    # full model features for this df
    model_features = build_feature_sets(df_task.columns)
    full_features = model_features.get("full", [])

    # restrict to subjects with complete full-model features
    full_subjects = complete_subjects(df_task, full_features, target)

    df_use = df_task[df_task[ID_COL].isin(full_subjects)].copy()
    df_use = df_use.dropna(subset=[target] + full_features)

    # numeric coercion & drop missing
    for col in full_features:
        df_use[col] = pd.to_numeric(df_use[col], errors="coerce")
    df_use[target] = pd.to_numeric(df_use[target], errors="coerce")
    df_use = df_use.dropna(subset=[target] + full_features)

    # tuned hyperparameters
    rf_params = load_tuned_rf_params_regression()

    # compute SHAP
    shap_expl, shap_table = stratified_cv_feature_importance(
        df=df_use,
        fold_column="fold",
        model_type=RandomForestRegressor,
        model_params=rf_params,
        target_column=target,
        feature_columns=full_features,
        save_dir=None,
        task_name=None,
    )

    if shap_expl is None:
        print(f"[SHAP] no SHAP Explanation for {task_name} / {target}")
        return None, None, None, [], {}

    return shap_expl, shap_table, df_use, full_features, rf_params

# create beeswarm plot for global SHAP values of one task-score combination
def plot_shap_beeswarm_full_model(task_name, target, shap_dir=None, max_display=20):

    shap_expl, shap_table, df_use, full_features, rf_params = prepare_full_model_shap(
        task_name, target
    )
    if shap_expl is None:
        return

    if shap_dir is None:
        shap_dir = os.path.join(
            GIT_DIRECTORY,
            "results",
            "regression",
            "random_forest",
            "feature_importance",
        )

    save_dir = os.path.join(shap_dir, target, task_name)
    os.makedirs(save_dir, exist_ok=True)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 10))

    # original explanation
    orig_expl = shap_expl
    orig_names = list(orig_expl.feature_names)

    # labels with superscripts ¹/²/³
    name_to_label = {n: f"{n}{feature_superscript(n)}" for n in orig_names}

    shap_expl_labeled = shap.Explanation(
        values=orig_expl.values,
        base_values=orig_expl.base_values,
        data=orig_expl.data,
        feature_names=[name_to_label[n] for n in orig_names],
    )

    # beeswarm plot
    shap.plots.beeswarm(shap_expl_labeled, max_display=max_display, show=False)

    ax = plt.gca()
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("Feature")

    footnote = "¹ linguistic   ² acoustic   ³ demographics"
    plt.figtext(
        0.0,
        0.02,
        footnote,
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="Arial",
    )

    plt.tight_layout()
    out_path = os.path.join(
        save_dir, f"{target}_{task_name}_shap_beeswarm_{max_display}.png"
    )
    plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[SHAP] saved beeswarm to: {out_path}")



# create one local SHAP waterfall plot for a randomly chosen subject
def plot_shap_waterfall_full_model(task_name, target, shap_dir, max_display=9, random_state=0):
    shap_expl, shap_table, df_use, full_features, rf_params = prepare_full_model_shap(
        task_name, target
    )
    if shap_expl is None or df_use is None:
        return

    if shap_dir is None:
        shap_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest", "feature_importance")

    save_dir = os.path.join(shap_dir, target, task_name)
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(random_state)
    rand_idx = int(rng.integers(0, len(df_use)))
    subject_id = df_use.iloc[rand_idx][ID_COL]
    subject_fold = int(df_use.iloc[rand_idx]["fold"])

    train_df = df_use[df_use["fold"] != subject_fold].copy()
    test_row = df_use[df_use[ID_COL] == subject_id].copy()

    # ensure numeric arrays
    X_train = train_df[full_features].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[target], errors="coerce")
    X_test_one = test_row[full_features].apply(pd.to_numeric, errors="coerce")

    X_train_np = X_train.to_numpy(dtype="float64")
    X_test_one_np = X_test_one.to_numpy(dtype="float64")

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train_np, y_train.to_numpy(dtype="float64"))

    # SHAP explainer
    background = shap.sample(X_train_np, min(200, len(X_train_np)), random_state=42)
    explainer = shap.TreeExplainer(
        rf,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    ex = explainer(X_test_one_np)

    # build labeled Explanation
    local_labels = [f"{n}{feature_superscript(n)}" for n in full_features]
    ex_labeled = shap.Explanation(
        values=ex.values,
        base_values=ex.base_values,
        data=X_test_one_np,
        feature_names=local_labels,
    )

    subject_id_wf = str(subject_id)
    title = f"Local SHAP Values: {format_title(target)} (Subject {subject_id_wf}, {task_name})"

    plt.close("all")
    plt.figure(figsize=(9.5, 6))
    shap.plots.waterfall(ex_labeled[0], max_display=max_display, show=False)
    plt.title(title)

    fig = plt.gcf()
    axes = fig.get_axes()
    main_ax = axes[0]

    for a in axes:
        a.grid(False)
        a.set_axisbelow(False)
        a.set_facecolor("white")

    main_ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.5,
        color="0.85",
    )
    main_ax.set_axisbelow(True)

    plt.subplots_adjust(left=0.30, right=0.96, bottom=0.15, top=0.88)
    xmin, xmax = main_ax.get_xlim()
    main_ax.set_xlim(xmin - 0.10 * (xmax - xmin), xmax + 0.15 * (xmax - xmin))

    plt.figtext(
        0.00, 0.02,
        "¹ linguistic   ² acoustic   ³ demographics",
        ha="left", va="top", fontsize=10, fontfamily="Arial"
    )

    out_path = os.path.join(
        save_dir,
        f"{task_name}_{target}_local_waterfall_subject-{subject_id_wf}.png",
    )
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] saved waterfall to: {out_path}")


# run all SHAP analyses for all task-score combinations
def run_all_shap_outputs(shap_dir=None):

    if shap_dir is None:
        shap_dir = os.path.join(GIT_DIRECTORY, "results", "regression", "random_forest", "feature_importance")

    # 1) SHAP tables for all combinations
    run_shap_full_models(shap_dir=shap_dir)

    # 2) heatmap across tasks and scores
    build_shap_heatmap(
        shap_dir=shap_dir,
        out_dir=None,
    )

    # 3) beeswarm + waterfall for each task + score
    for target in SCORES:
        for task in TASKS:
            print(f"SHAP plots for {task} -> {target}")
            plot_shap_beeswarm_full_model(
                task_name=task,
                target=target,
                shap_dir=shap_dir,
                max_display=20,
            )
            plot_shap_waterfall_full_model(
                task_name=task,
                target=target,
                shap_dir=shap_dir,
                max_display=9,
                random_state=11,
            )


if __name__ == "__main__":
    run_all_shap_outputs()
