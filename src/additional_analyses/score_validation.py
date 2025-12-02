import os
import pandas as pd

from additional_analyses.icc import compute_icc

from config.constants import GIT_DIRECTORY

# paths
scores_path  = os.path.join(GIT_DIRECTORY, "data/language_scores_all_subjects.csv")
manual_scores = os.path.join(GIT_DIRECTORY, "resources/Score_Validierung.xlsx")
out_dir = os.path.join(GIT_DIRECTORY, "results/descriptives/ICC")
os.makedirs(out_dir, exist_ok=True)

# load data
# automatic scores
automatic_scores = pd.read_csv(scores_path)
# manual scores
xls = pd.ExcelFile(manual_scores)
sheet_name = "scores_manuell"
manual_scores = pd.read_excel(manual_scores, sheet_name=sheet_name, engine="openpyxl")

# merge manual & automatic on Subject_ID
scores = manual_scores.merge(automatic_scores, on="Subject_ID", how="inner")

# pairing for ICC
pairing_map = {
    "SemanticFluency": ("semantic_fluency_score_m", "SemanticFluencyScore"),
    "PhonemicFluency": ("phonemic_fluency_score_m", "PhonemicFluencyScore"),
    "PictureNaming": ("picture_naming_score_m", "PictureNamingScore"),
}

# ICCs (2,1) absolute agreement
rows = []
for label, (mcol, acol) in pairing_map.items():
    rows += compute_icc(scores, mcol, acol, label)

df = pd.DataFrame(rows)
path = os.path.join(out_dir, "icc_overall.csv")
df.to_csv(path, index=False)
print(f"saved ICCs to: {path}")
