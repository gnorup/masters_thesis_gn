import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from config.constants import GIT_DIRECTORY, ID_COL
from config.feature_sets import get_linguistic_features, get_acoustic_features, get_demographic_features


def load_demographics(demographics_csv=None):
    """
    Load and prepare demographics in a consistent way across all analyses
    """
    if demographics_csv is None:
        demographics_csv = os.path.join(GIT_DIRECTORY, "data", "demographics_data.csv")

    df = pd.read_csv(demographics_csv)

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)

    # normalize selected string columns
    for col in ["Gender", "Education", "Country"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.lower()
                .str.strip()
            )

    # Age numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Socioeconomic (MacArthur) numeric
    if "Socioeconomic" in df.columns:
        df["Socioeconomic"] = pd.to_numeric(df["Socioeconomic"], errors="coerce")

    # Gender -> f:0, m:1
    if "Gender" in df.columns:
        gender_map = {
            "f": 0, "female": 0,
            "m": 1, "male": 1,
        }
        df["Gender"] = df["Gender"].map(gender_map)

    # Education -> 1–6, then Education_level (0/1/2)
    if "Education" in df.columns:
        education_map = {
            "less_than_highschool": 1,
            "high_school": 2,
            "vocational": 3,
            "bachelor": 4,
            "master": 5,
            "phd": 6,
            "no_answer": np.nan,
            "": np.nan,
        }
        df["Education"] = df["Education"].map(education_map)
        df["Education_level"] = df["Education"].map(
            {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
        )

    # Country -> uk:0, usa:1
    if "Country" in df.columns:
        df["Country"] = df["Country"].map(
            {
                "uk": 0,
                "usa": 1, "us": 1,
            }
        )

    # drop Language (redundant)
    df = df.drop(columns=["Language"], errors="ignore")

    # prepare so that it supports integers and missing values
    for col in ["Gender", "Education", "Education_level", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    keep = [ID_COL, "Age", "Gender", "Education", "Education_level", "Socioeconomic", "Country"]
    cols = [ID_COL] + [
        c for c in keep if c in df.columns and c != ID_COL
    ]

    return df[cols].copy()

def load_task_dataframe(task_name, target, scores, demographics):
    """
    Load and merge features and stratified folds
    """
    features_path = os.path.join(GIT_DIRECTORY, f"results/features/cleaned/{task_name}_cleaned.csv")
    folds_path = os.path.join(GIT_DIRECTORY, "data/stratified_folds.csv")
    features = pd.read_csv(features_path)
    folds = pd.read_csv(folds_path)
    scores = scores.copy()
    demographics = demographics.copy()

    for df in (features, scores, demographics, folds):
        df[ID_COL] = df[ID_COL].astype(str)
    df = pd.merge(features, scores[[ID_COL, target]], on=ID_COL)
    df = pd.merge(df, demographics, on=ID_COL)
    df = pd.merge(df, folds[[ID_COL, "fold"]], on=ID_COL)

    return df

def get_model_feature_list(df_columns, selected_features, target_name=None):
    """
    Get features for certain model
    """
    drop = {ID_COL, "fold"}
    if target_name:
        drop.add(target_name)

    feature_list = [f for f in selected_features if f not in drop and f in df_columns]

    return feature_list

def build_feature_sets(df_columns):
    """
    Model specification (what features belong into what model)
    """
    linguistic = get_linguistic_features()
    acoustic = get_acoustic_features()
    demographic = get_demographic_features()

    configs = {
        "demographics": demographic,
        "acoustic": sorted(list(acoustic)),
        "linguistic": sorted(list(linguistic)),
        "linguistic+acoustic": sorted(list(acoustic | linguistic)),
        "full": sorted(list(acoustic | linguistic)) + demographic
    }
    for k in list(configs.keys()):
        configs[k] = get_model_feature_list(df_columns, configs[k], target_name=None)

    return configs

def complete_subjects(df, feature_cols, target_name):
    """
    Get Subject_IDs with no missing in target & all features
    """
    need = [target_name] + (feature_cols if len(feature_cols) > 0 else [])

    return set(df.dropna(subset=need)[ID_COL])

def subjects_with_all_features_and_scores(full_subjects, scores_df, score_cols):
    """
    Get Subject_IDs with no missing in target, all features & scores
    """
    if ID_COL not in scores_df.columns:
        raise ValueError("scores_df must contain 'Subject_ID'")
    all_scores = set(scores_df.dropna(subset=list(score_cols))[ID_COL])

    return set(full_subjects) & all_scores

def subject_intersection_for_score(oof_all, target, tasks, models=None):
    """
    Subject intersection for score for plotting
    """
    pools = []
    for t in tasks:
        sub = oof_all[(oof_all["target"] == target) & (oof_all["task"] == t)]
        if models is not None:
            sub = sub[sub["model"].isin(models)]
        pools.append(set(sub[ID_COL].dropna().unique()))

    return set.intersection(*pools) if pools else set()

def subset_in_order(oof_preds, task, model, target, group_col, levels):
    """
    Create subset of data
    """
    levels = list(levels)
    sub = oof_preds[
        (oof_preds["task"] == task) &
        (oof_preds["model"] == model) &
        (oof_preds["target"] == target) &
        (oof_preds[group_col].isin(levels))
    ].copy()
    sub[group_col] = sub[group_col].astype(CategoricalDtype(categories=levels, ordered=True))
    return sub, levels

def make_age_binary_groups(df, age_col="Age", label_col="AgeGroup2"):
    """
    Prepare age groups: 2 age bins (<65 & ≥65) -> for bias analyses
    """

    ages = pd.to_numeric(df[age_col], errors="coerce")

    bins = [float("-inf"), 65, float("inf")]
    right = True
    labels = ["<65", "≥65"]

    df2 = df.copy()
    df2[label_col] = pd.cut(ages, bins=bins, include_lowest=True, right=right)
    df2[label_col] = df2[label_col].cat.rename_categories(labels)

    return df2, labels