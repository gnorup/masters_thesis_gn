# setup
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

# set font for all plots
plt.rcParams['font.family'] = 'Arial'

# plot distribution of individual features (and scores)
def plot_distributions(df, columns, save_dir, id_column="Subject_ID"):
    for col in columns:
        plt.figure(figsize=(12,4))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")

        safe_colname = col.replace("/", "_")
        save_path = os.path.join(save_dir, f"{safe_colname}_distribution.png")
        plt.savefig(save_path, dpi=300)

        plt.tight_layout()
        plt.show()
        plt.close()
        print(f"saved to {save_path}")

        # detect outliers based on IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"outliers in '{col}': ")
            print(outliers[[id_column, col]])


# identify IQR-based outliers
def identify_outliers_iqr(df, feature_cols, id_column ="Subject_ID", iqr_multiplier = 1.5, save_path=None):
    subject_outlier_map = {}
    all_outlier_ids = set()
    total_subjects = len(df)

    iqr_outliers = []

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_ids = outliers[id_column].tolist()
        subject_outlier_map[col] = outlier_ids
        all_outlier_ids.update(outlier_ids)

        n_outliers = len(outlier_ids)
        percent_outliers = (n_outliers / total_subjects) * 100
        iqr_outliers.append({
            "feature": col,
            "n_outliers": n_outliers,
            "percent_outliers": round(percent_outliers, 2)
        })

        print(f"{col}: {n_outliers} subjects ({percent_outliers:.2f}%)")

    print(f"\ntotal unique subjects flagged as outliers in at least one feature: {len(all_outlier_ids)}")

    outlier_df = pd.DataFrame(iqr_outliers).sort_values(by="percent_outliers", ascending=False)
    if save_path:
        outlier_df.to_csv(save_path, index=False)
        print(f"\nsaved iqr-outlier summary to: {save_path}")

    return list(all_outlier_ids), subject_outlier_map, outlier_df


# clean feature set based on missing values and IQR
def clean_feature_set(task_name, feature_df, missing_csv, outlier_csv,
                      missing_threshold=20.0, outlier_threshold=5.0,
                      save_dir=None):
    missing_df = pd.read_csv(missing_csv, index_col=0)
    iqr_df = pd.read_csv(outlier_csv, index_col=0)

    missing_filtered = missing_df[
        (missing_df["missing_percent"] > missing_threshold) |
        (missing_df["zero_percent"] > missing_threshold)
    ].index.tolist()

    outlier_filtered = iqr_df[iqr_df["percent_outliers"] > outlier_threshold].index.tolist()

    all_to_remove = sorted(set(missing_filtered + outlier_filtered))
    cleaned_df = feature_df.drop(columns=all_to_remove, errors="ignore")

    print(f"features removed due to missing/zero values: {sorted(missing_filtered)}")
    print(f"features removed due to IQR outliers: {sorted(outlier_filtered)}")

    # save cleaned data frame
    save_path = os.path.join(save_dir, f"{task_name}_cleaned.csv")
    cleaned_df.to_csv(save_path, index=False)
    print(f"saved cleaned feature set ({len(all_to_remove)} features removed) to: {save_path}")

    return cleaned_df, all_to_remove


# correlation matrix
def correlation_matrix(df, feature_cols, save_path=None, method="pearson", figsize=(20, 18), task_name=None):
    corr = df[feature_cols].corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", annot=False, center=0)
    plt.xticks(rotation=90, fontsize=8, ha="right")
    plt.yticks(rotation=0, fontsize=8)
    plt.title(f"Correlation Matrix {task_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved correlation matrix to: {save_path}")

    plt.show()

    return corr


# feature-importance
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
    X_test_all = []

    for fold in sorted(df[fold_column].unique()):
        train_df = df[df[fold_column] != fold]
        test_df = df[df[fold_column] == fold]

        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_test, y_test = test_df[feature_columns], test_df[target_column]

        # train model
        model = model_type(**(model_params or {}))
        model.fit(X_train, y_train)

        # compute SHAP-values
        try:
            explainer = shap.Explainer(model, X_train)
            shap_vals = explainer(X_test)
            shap_values_all.append(shap_vals.values)
            X_test_all.append(X_test)
        except Exception as e:
            print(f"SHAP failed on fold {fold}: {e}")

    # create aggregated SHAP explanation object
    shap_explanation = None
    if shap_values_all and X_test_all:
        all_shap_values = np.vstack(shap_values_all)
        all_X_test = pd.concat(X_test_all)
        shap_explanation = shap.Explanation(
            values=all_shap_values,
            base_values=np.mean(all_shap_values, axis=0),
            data=all_X_test,
            feature_names=all_X_test.columns.tolist()
        )

        # save SHAP values table with mean
        shap_table = pd.DataFrame({
            "Feature": all_X_test.columns,
            "Mean_SHAP_Value": np.abs(all_shap_values).mean(axis=0)
        }).sort_values("Mean_SHAP_Value", ascending=False).reset_index(drop=True)
        if save_dir and task_name:
            shap_table.to_csv(os.path.join(save_dir, f"{task_name}_{target_column}_shap_values_table.csv"), index=False)

    return shap_explanation, shap_table
