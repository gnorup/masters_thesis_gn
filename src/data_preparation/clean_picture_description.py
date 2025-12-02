"""
Clean picture description feature sets so that they use the intersection of features present in both cleaned picture
description tasks (cookieTheft and picnicScene).
"""

import os
import pandas as pd

from config.constants import GIT_DIRECTORY, ID_COL
from data_preparation.data_cleaning_helpers import impute_mean_dataframe

# paths
FEATURES_DIR = os.path.join(GIT_DIRECTORY, "results", "features")
CLEANED_DIR = os.path.join(FEATURES_DIR, "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)

# inputs from picture description feature extraction
PD_INPUTS = {
    "picture_description": os.path.join(FEATURES_DIR, "picture_description.csv"),
    "picture_description_2min": os.path.join(FEATURES_DIR, "picture_description_2min.csv"),
    "picture_description_1min": os.path.join(FEATURES_DIR, "picture_description_1min.csv"),
}

# cleaned cookieTheft and picnicScene (from clean_feature_sets.py)
CT_CLEANED = os.path.join(CLEANED_DIR, "cookieTheft_cleaned.csv")
PS_CLEANED = os.path.join(CLEANED_DIR, "picnicScene_cleaned.csv")


def get_feature_intersection(ct_df, ps_df):
    """
    Derive intersection of features from cleaned cookieTheft and picnicScene.
    """
    feat_ct = set(ct_df.columns) - {ID_COL}
    feat_ps = set(ps_df.columns) - {ID_COL}
    feature_intersection = sorted(feat_ct & feat_ps)

    print(f"cookieTheft_cleaned: {len(feat_ct)} feature cols, picnicScene_cleaned: {len(feat_ps)} feature cols, "
          f"feature intersection: {len(feature_intersection)}")

    return feature_intersection


def clean_single_pd(path_in, path_out, feature_intersection):
    """
    Clean a single picture-description file so that only the intersection of features from cT and pS is kept and
    remaining missing values are mean-imputed.
    """
    if not os.path.exists(path_in):
        print(f"[skip] missing input: {path_in}")
        return

    df = pd.read_csv(path_in)
    df[ID_COL] = df[ID_COL].astype(str)

    before = len(df)

    # drop meta column if present
    df = df.drop(columns=[c for c in {"duration_used_sec"} if c in df.columns], errors="ignore")

    # keep only Subject_ID + feature_intersection (drop any extra columns)
    keep_cols = [ID_COL] + [c for c in feature_intersection if c in df.columns]
    df = df[keep_cols].copy()

    # mean-impute numeric columns
    df = impute_mean_dataframe(df, exclude_cols=[ID_COL])

    # enforce consistent column order
    ordered_cols = [ID_COL] + [c for c in feature_intersection if c in df.columns]
    df = df.reindex(columns=ordered_cols)

    after = len(df)
    print(f"[subjects] {os.path.basename(path_in)}: kept {after} / {before} rows")
    print(f"[columns] {os.path.basename(path_out)}: {len(df.columns)} columns")

    df.to_csv(path_out, index=False)
    print(f"[saved] {path_out}  shape={df.shape}")


def main():
    # check CT/PS cleaned exist and load them
    if not os.path.exists(CT_CLEANED) or not os.path.exists(PS_CLEANED):
        raise FileNotFoundError(
            "Missing cleaned CT/PS files. Please run clean_feature_sets.py first.\n"
            f"Expected:\n - {CT_CLEANED}\n - {PS_CLEANED}"
        )

    ct_df = pd.read_csv(CT_CLEANED)
    ps_df = pd.read_csv(PS_CLEANED)
    ct_df[ID_COL] = ct_df[ID_COL].astype(str)
    ps_df[ID_COL] = ps_df[ID_COL].astype(str)

    feature_intersection = get_feature_intersection(ct_df, ps_df)

    outputs = {
        "picture_description": os.path.join(CLEANED_DIR, "picture_description_cleaned.csv"),
        "picture_description_2min": os.path.join(CLEANED_DIR, "picture_description_2min_cleaned.csv"),
        "picture_description_1min": os.path.join(CLEANED_DIR, "picture_description_1min_cleaned.csv"),
    }

    # process each picture-description version
    for key, in_path in PD_INPUTS.items():
        out_path = outputs[key]
        print(f"\n[process] {key}")
        clean_single_pd(path_in=in_path, path_out=out_path, feature_intersection=feature_intersection)

    print("\ndone: cleaned picture-description feature sets saved in:")
    for p in outputs.values():
        print(" -", p)


if __name__ == "__main__":
    main()