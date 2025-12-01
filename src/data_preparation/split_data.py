# generate and save stratified folds

import os
from config.constants import GIT_DIRECTORY
from data_preparation.stratified_splits import create_and_save_stratified_folds

def create_splits():
    demographics_path = os.path.join(GIT_DIRECTORY, "data/demographics_data.csv")

    # main 5-fold CV
    five_fold_path = os.path.join(GIT_DIRECTORY, "data", "stratified_folds.csv")
    create_and_save_stratified_folds(
        demographics_path=demographics_path,
        output_path=five_fold_path,
        n_splits=5,
    )

    # half split for hyperparameter tuning
    half_path = os.path.join(GIT_DIRECTORY, "data", "stratified_folds_half.csv")
    create_and_save_stratified_folds(
        demographics_path=demographics_path,
        output_path=half_path,
        n_splits=2,
    )
    print("[split_data] stratified folds successfully created.")
