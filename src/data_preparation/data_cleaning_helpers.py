import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY

### 1) filter for people with missing tasks (when n_words < 15)

def filter_invalid_tasks(df, min_words, id_column, n_words_col):
    """
    Remove subjects where n_words < min_words.
    """
    mask = df[n_words_col] >= min_words
    removed_ids = df.loc[~mask, id_column].tolist()

    filtered = df.loc[mask].copy()
    print(f"[filter_invalid_tasks] removed {len(removed_ids)} subjects with {n_words_col} < {min_words}")

    return filtered, removed_ids

### 2) filter for selected features

def drop_all_egemaps_except(df, keep_columns):
    """
    Drop all eGeMAPS_* columns except those in keep_columns.
    """
    keep_columns = set(keep_columns)
    eg_cols = [c for c in df.columns if c.startswith("eGeMAPS_")]
    drop_cols = [c for c in eg_cols if c not in keep_columns]

    filtered = df.drop(columns=drop_cols, errors="ignore")
    print(f"[drop_all_egemaps_except] dropped {len(drop_cols)} eGeMAPS columns, kept {len(eg_cols) - len(drop_cols)}.")

    return filtered, drop_cols

### 3) remove features that have more than 20% missing/zero values

def compute_missing_zero_table(df, feature_cols):
    """
    Compute missing/zero counts and percentages per feature.
    """
    missing_counts = df[feature_cols].isna().sum()
    zero_counts = (df[feature_cols] == 0).sum()
    total_rows = len(df)

    missing_data = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": (missing_counts / total_rows) * 100,
        "zero_count": zero_counts,
        "zero_percent": (zero_counts / total_rows) * 100,
    }).sort_values(by=["missing_percent", "zero_percent"], ascending=False)

    return missing_data

def filter_features_by_missing_zero(df, missing_table, threshold=20.0):
    """
    Remove features where missing_percent or zero_percent > threshold.
    """
    to_remove = missing_table[
        (missing_table["missing_percent"] > threshold) |
        (missing_table["zero_percent"] > threshold)
    ].index.tolist()

    filtered = df.drop(columns=to_remove, errors="ignore")
    print(f"[filter_features_by_missing_zero] removed {len(to_remove)} features (> {threshold}% missing/zeros).")

    return filtered, to_remove

### 4) remove features that have more than 10% outliers (defined by IQR)

def identify_outliers_iqr(df, feature_cols, id_column="Subject_ID", iqr_multiplier=1.5):
    """
    Identify IQR-based outliers per feature.
    """
    total_subjects = len(df)
    iqr_outliers = []

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        n_outliers = len(outliers)
        percent_outliers = (n_outliers / total_subjects) * 100

        iqr_outliers.append({
            "feature": col,
            "n_outliers": n_outliers,
            "percent_outliers": round(percent_outliers, 2),
        })

    outlier_df = pd.DataFrame(iqr_outliers).sort_values(
        by="percent_outliers", ascending=False
    )

    return outlier_df

def filter_features_by_outliers(df, outlier_df, percent_threshold=10.0):
    """
    Remove features with percent_outliers > percent_threshold.
    """
    to_remove = outlier_df[outlier_df["percent_outliers"] > percent_threshold]["feature"].tolist()
    filtered = df.drop(columns=to_remove, errors="ignore")
    print(f"[filter_features_by_outliers] removed {len(to_remove)} features (> {percent_threshold}% IQR-outliers).")
    return filtered, to_remove

### 5) remove strongly intercorrelated features (one of a pair of r >|.95| correlated features)

def correlation_matrix(df, feature_cols, save_path=None, method="pearson", task_name=None):
    """
    Plot correlation matrix for features of given task.
    """
    corr = df[feature_cols].corr(method=method)

    plt.figure(figsize=20, 18)
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

def drop_correlated_features_manual(df, features_to_drop):
    """
    Drop the set of correlated features chosen from |r| > .95 inspection.
    """
    cols_present = [c for c in features_to_drop if c in df.columns]
    filtered = df.drop(columns=cols_present, errors="ignore")
    print(f"[drop_correlated_features_manual] dropped {len(cols_present)} correlated features.")
    return filtered, cols_present

### 6) impute mean for missing values

def impute_mean_dataframe(df, exclude_cols):
    """
    Impute missing values in all numeric columns with the column mean.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.difference(exclude_cols)

    for col in numeric_cols:
        if df[col].isna().any():
            mean_val = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_val)
            print(f"[impute_mean_dataframe] imputed missing values in '{col}' with mean={mean_val:.4f}")

    return df