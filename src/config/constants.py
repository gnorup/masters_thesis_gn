# project-wise constants, such as paths

# paths
DATA_DIRECTORY = "/Volumes/g_psyplafor_methlab_data$/Speech_data/LanguageHealthyAging/data"
WORD_TIMESTAMPS_DATA = "/Volumes/g_psyplafor_methlab$/Students/Gila/word_timestamps/{task}/google/timestamps"

GIT_DIRECTORY = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/"

# defaults
ID_COL = "Subject_ID"
SCORES = ["PictureNamingScore", "SemanticFluencyScore", "PhonemicFluencyScore"]
TASKS = ["picnicScene", "cookieTheft", "journaling"]
MODELS = ["baseline","demographics","acoustic","linguistic","linguistic+acoustic","full"]
PD_TASKS = ["picture_description_1min", "picture_description_2min", "picture_description"]

RANDOM_STATE = 42
N_BOOT = 1000
CI = 0.95
ALPHA = 0.05
METRICS = ("R2", "MAE", "RMSE")