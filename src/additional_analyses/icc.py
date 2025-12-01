import pandas as pd
import numpy as np
import pingouin as pg

def compute_icc(scores_df, manual_col, auto_col, score_label):
    """
    Compute ICC (2,1) (absolute agreement) between manual and automatic scores
    """
    pair_df = scores_df[[manual_col, auto_col]].dropna().copy()
    n_subjects = len(pair_df)
    if n_subjects < 2:
        return []

    # ratings: manual scores & automatic scores
    ratings = pd.concat(
        [
            pair_df[manual_col].reset_index(drop=True),
            pair_df[auto_col].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    # rater labels: "manual" repeated n times, then "automatic" repeated n times
    rater_labels = np.repeat(["manual", "automatic"], repeats=n_subjects)

    subject_ids = np.tile(np.arange(n_subjects), 2)

    long_df = pd.DataFrame(
        {
            "subject": subject_ids,
            "rater": rater_labels,
            "score": ratings,
        }
    )

    icc_table = pg.intraclass_corr(
        data=long_df,
        targets="subject",
        raters="rater",
        ratings="score",
    )

    results = []
    for _, row in icc_table.iterrows():
        if row["Type"] == "ICC2":
            results.append(
                {
                    "Score": score_label,
                    "ICC_Type": row["Type"],
                    "ICC": row["ICC"],
                    "CI95_lower": row["CI95%"][0],
                    "CI95_upper": row["CI95%"][1],
                    "n": n_subjects,
                }
            )

    return results
