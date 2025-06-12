import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

"""
Create stratified splits for 5-fold cross-validation -> split data once and save to csv-file.
"""

def create_stratification_group(df):
    df = df.copy()

    # age - 3 groups: <65, 65-75, >75 (based on Maercker, 2015; UN)
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 64, 75, np.inf],
        labels=["under65", "65to75", "over75"]
    )

    # education - 3 groups (based on ISCED; eurostat)
    def map_education(edu):
        edu = str(edu).lower()
        if "less_than_highschool" in edu:
            return "low"
        elif "high_school" in edu or "vocational" in edu:
            return "medium"
        elif "bachelor" in edu or "master" in edu or "phd" in edu:
            return "high"
        else:
            return "unknown"

    df["edu_group"] = df["Education"].apply(map_education)

    # SES - 3 groups: low (1-3), middle (4-7), high (8-10) (based on MacArthur scale; Chen et al., 2012)
    def map_ses(s):
        try:
            s = int(s)
            if 1 <= s <= 3:
                return "low"
            elif 4 <= s <= 7:
                return "middle"
            elif 8 <= s <= 10:
                return "high"
            else:
                return "unknown"
        except:
            return "unknown"

    df["ses_group"] = df["Socioeconomic"].apply(map_ses)

    # clean: remove invalid categories before combining
    invalid_values = ["no_answer", "other", "unknown", "nan"]
    df = df[
        ~df["Gender"].isin(invalid_values) &
        ~df["Country"].isin(invalid_values) &
        ~df["edu_group"].isin(invalid_values) &
        ~df["ses_group"].isin(invalid_values) &
        df["age_group"].notna()
        ].reset_index(drop=True)

    # combine into one label
    df["strat_group"] = (
            df["Gender"].astype(str) + "_" +
            df["Country"].astype(str) + "_" +
            df["age_group"].astype(str) + "_" +
            df["edu_group"].astype(str) + "_" +
            df["ses_group"].astype(str)
    )

    return df

def create_and_save_stratified_folds(demographics_path, output_path, n_splits=5):
    # load demographics
    demographics = pd.read_csv(demographics_path)
    df = demographics.copy()

    # create stratification group and clean
    df = create_stratification_group(df)

    # assign folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # shuffles data before splitting, then creates splits for stratified cross-validation
    df["fold"] = -1 # new column: fold, -1 as placeholder
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, df["strat_group"])): # splits based on distribution of values in strat_group
        df.loc[test_idx, "fold"] = fold_idx + 1 # folds labeled 1 to 5

    # save
    df.to_csv(output_path, index=False)
    print(f"Saved folds to: {output_path}")

def load_fold_split(df, fold):
    # return train and test split for fold index
    train_df = df[df["fold"] != fold].reset_index(drop=True) # filters data to include rows not equal to current fold (train-set)
    test_df = df[df["fold"] == fold].reset_index(drop=True) # filters data to include rows with current fold (test-set)
    return train_df, test_df

