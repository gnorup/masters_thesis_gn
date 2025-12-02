import os
import pandas as pd

from config.constants import GIT_DIRECTORY, TASKS, ID_COL
from data_preparation.data_cleaning_helpers import (
    filter_invalid_tasks,
    drop_all_egemaps_except,
    compute_missing_zero_table,
    filter_features_by_missing_zero,
    identify_outliers_iqr,
    filter_features_by_outliers,
    drop_correlated_features_manual,
    impute_mean_dataframe,
    correlation_matrix
)

# paths
FEATURES_DIR = os.path.join(GIT_DIRECTORY, "results", "features")
OUTPUT_DIR = os.path.join(FEATURES_DIR, "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MISSING_DIR = os.path.join(GIT_DIRECTORY, "results", "data_preparation", "missing_values")
OUTLIER_DIR = os.path.join(GIT_DIRECTORY, "results", "data_preparation", "outliers")
CORR_DIR = os.path.join(GIT_DIRECTORY, "results", "data_preparation", "correlated_features")
os.makedirs(MISSING_DIR, exist_ok=True)
os.makedirs(OUTLIER_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)

# eGeMAPS selection: keep shimmer, jitter and top 10 most important features from Niemelä et al. (2024)
eGeMAPS = [
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "eGeMAPS_F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "eGeMAPS_logRelF0-H1-A3_sma3nz_amean",
    "eGeMAPS_loudness_sma3_amean",
    "eGeMAPS_loudness_sma3_percentile20.0",
    "eGeMAPS_mfcc3_sma3_amean",
    "eGeMAPS_mfcc3V_sma3nz_amean",
    "eGeMAPS_slopeUV0-500_sma3nz_amean",
    "eGeMAPS_slopeV0-500_sma3nz_amean",
    "eGeMAPS_jitterLocal_sma3nz_amean",
    "eGeMAPS_shimmerLocaldB_sma3nz_amean",
]

# features from intercorrelated feature pairs to drop -> adjust according to correlation matrix
# (selected from pairs based on feature importance in previous studies / NaNs)
corr_drop = {
    "cookieTheft": [
        "mattr_50",
        "mattr_40",
        "mattr_20",
        "ttr",
        "speech_rate_words",
        "long_pause_count",
        "pause_ratio",
    ],
    "picnicScene": [
        "mattr_50",
        "mattr_40",
        "mattr_20",
        "speech_rate_words",
        "long_pause_count",
        "ttr",
        "pause_word_ratio",
        "pause_ratio",
        "eGeMAPS_mfcc3_sma3_amean",
    ],
    "journaling": [
        "mattr_50",
        "mattr_40",
        "mattr_20",
        "speech_rate_words",
        "long_pause_count",
        "ttr",
        "INTJ",
        "pause_ratio",
    ],
}

# run feature cleaning
def clean_task(task_name):
    """
    Clean feature set for given task.
    """
    print(f"\nprocessing task: {task_name}")

    # load base features & language scores
    features_path = os.path.join(FEATURES_DIR, f"{task_name}.csv")
    df_features = pd.read_csv(features_path)

    df_features[ID_COL] = df_features[ID_COL].astype(str)

    ### 1) filter for people with missing tasks (when n_words < 15)
    df_features, removed_short = filter_invalid_tasks(
        df_features,
        min_words=15,
        id_column=ID_COL,
        n_words_col="n_words",
    )

    print(f"[1] removed {len(removed_short)} (short n_words)")

    ### 2) filter for selected features
    df_features, dropped_eg = drop_all_egemaps_except(df_features, keep_columns=eGeMAPS)

    print(f"[2] dropped {len(dropped_eg)} eGeMAPS features.")

    ### 3) remove features that have more than 20% missing/zero values
    feature_cols = [c for c in df_features.columns if c not in [ID_COL]]
    missing_table = compute_missing_zero_table(df_features, feature_cols)
    missing_save_path = os.path.join(MISSING_DIR, f"{task_name}_missing_data.csv")
    missing_table.to_csv(missing_save_path)

    print(f"[3] saved missing/zero table to: {missing_save_path}")

    df_feat, removed_missing_zero = filter_features_by_missing_zero(
        df_features,
        missing_table,
        threshold=20.0,
    )
    print(f"[3] removed {len(removed_missing_zero)} features (> 20% missing/zeros).")

    # recompute feature_cols after dropping
    feature_cols = [c for c in df_feat.columns if c != ID_COL]

    ### 4) remove features that have more than 10% outliers (defined by IQR)
    outlier_df = identify_outliers_iqr(
        df_feat,
        feature_cols=feature_cols,
        iqr_multiplier=1.5,
    )
    outlier_save_path = os.path.join(OUTLIER_DIR, f"{task_name}_iqr_outliers.csv")
    outlier_df.to_csv(outlier_save_path, index=False)

    print(f"[4] saved IQR outlier summary to: {outlier_save_path}")

    df_feat, removed_outliers = filter_features_by_outliers(
        df_feat,
        outlier_df,
        percent_threshold=10.0,
    )

    print(f"[4] removed {len(removed_outliers)} features (> 10% outliers).")

    # update feature list again
    feature_cols = [c for c in df_feat.columns if c != ID_COL]

    ### 5) remove strongly intercorrelated features (one of a pair of r >|.95| correlated features)

    # compute correlation matrix and save heatmap + correlated pairs
    save_correlation_matrix = os.path.join(CORR_DIR, f"{task_name}_correlation_matrix.png")
    corr_matrix = correlation_matrix(df_feat, feature_cols, save_path=save_correlation_matrix, task_name=task_name)

    threshold = 0.95
    highly_correlated = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                highly_correlated.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(corr_val, 4),
                })

    highly_correlated_df = pd.DataFrame(highly_correlated)
    correlated_features_path = os.path.join(CORR_DIR, f"{task_name}_correlated_features.csv")
    highly_correlated_df.to_csv(correlated_features_path, index=False)

    print(f"[5] found {len(highly_correlated)} highly correlated feature pairs (|r| > {threshold})")
    print(f"[5] saved highly correlated feature pairs to {correlated_features_path}")

    corr_drop_list = corr_drop.get(task_name, [])
    df_feat, removed_corr = drop_correlated_features_manual(
        df_feat,
        corr_drop_list,
    )

    print(f"[5] dropped {len(removed_corr)} correlated features.")

    ### 6) impute mean for missing values
    df_clean = impute_mean_dataframe(df_feat, exclude_cols=[ID_COL])

    # save final cleaned feature sets
    final_path = os.path.join(OUTPUT_DIR, f"{task_name}_cleaned.csv")
    df_clean.to_csv(final_path, index=False)

    print(f"[final] saved cleaned feature set for {task_name} to: {final_path}")
    print(f"[final] shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")


def main():
    for task in TASKS:
        clean_task(task)

if __name__ == "__main__":
    main()