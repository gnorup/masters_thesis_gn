import os
import sys

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY
from data_preparation.stratified_splits import create_and_save_stratified_folds

# set file paths
features_path = os.path.join(GIT_DIRECTORY, "results/features/filtered/cookieTheft_filtered.csv")
demographics_path = os.path.join(GIT_DIRECTORY, "data/demographics_data.csv")
output_path = os.path.join(GIT_DIRECTORY, "data", "cookieTheft_stratified_folds.csv")

# generate and save stratified folds
create_and_save_stratified_folds(
    features_path=features_path,
    demographics_path=demographics_path,
    id_column="Subject_ID",
    output_path=output_path,
    n_splits=5  # 5-fold cross-validation
)
