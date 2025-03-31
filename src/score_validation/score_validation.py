import pandas as pd
file_path = "/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/Score_Validierung.xlsx" # path to excel file
df = pd.read_excel(file_path, engine="openpyxl") # read excel file
print(df.head()) # display first few rows

dfs = pd.read_excel(file_path, sheet_name=None, engine="openpyxl") # load all sheets
print(dfs.keys()) #print sheet names: scores_manuell and language_task_scores -> access sheet: df_sheetname = dfs["sheetname"]

df_sheet1 = dfs["scores_manuell"] # sheet with manually computed scores

df_sheet2 = dfs["language_task_scores"] # sheet with scores computed by algorithm

### prep to compare scores

df_sheet2.rename(columns={"study_submission_id": "subject_id"}, inplace=True) # make common column
print(df_sheet2.columns)

df_merged = df_sheet1.merge(df_sheet2, on="subject_id", suffixes=("_sheet1", "_sheet2")) # combine same subjects for comparison
print(df_merged.head())

print(df_merged.shape) #just to make sure that all the subjects were included
print(df_merged.columns)

### descriptive statistics

print(df_merged[['semantic_fluency_score_m', 'phonemic_fluency_score_m', 'picture_naming_score_m',
       'phonemic_fluency_score', 'semantic_fluency_score',
       'picture_naming_score']].describe())

# visualization of means
import matplotlib.pyplot as plt

score_pairs = [
    ("semantic_fluency_score_m", "semantic_fluency_score"),
    ("phonemic_fluency_score_m", "phonemic_fluency_score"),
    ("picture_naming_score_m", "picture_naming_score")
]

# calculate means for each pair
mean_values = {}
for score_m, score_auto in score_pairs:
    mean_values[score_m] = df_merged[score_m].mean()
    mean_values[score_auto] = df_merged[score_auto].mean()

# prepare data for plotting
score_labels = []
manual_means = []
auto_means = []

# for x axis labels
for score_m, score_auto in score_pairs:
    if score_m == "semantic_fluency_score_m":
        score_labels.append("Semantic Fluency Score")
    elif score_m == "phonemic_fluency_score_m":
        score_labels.append("Phonemic Fluency Score")
    elif score_m == "picture_naming_score_m":
        score_labels.append("Picture Naming Score")

    manual_means.append(mean_values[score_m])
    auto_means.append(mean_values[score_auto])

# position bars for same score next to each other
x = range(len(score_labels))
width = 0.35  # width of bars

# open figure for plots
fig, ax = plt.subplots(figsize=(10, 6))

# bars for manually computed scores (shifted left by half the width)
ax.bar([p - width / 2 for p in x], manual_means, width, label='Manual', color='b', align='center')

# bars for automatically computed scores (shifted right by half the width)
ax.bar([p + width / 2 for p in x], auto_means, width, label='Automatic', color='orange', align='center')

# labels
ax.set_ylabel('Mean Scores')
ax.set_title('Comparison of Manual and Automatic Scores')
ax.set_xticks(x)  # set x-ticks at the position of the bars
ax.set_xticklabels(score_labels)
ax.legend()

# save plot
plt.savefig('/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/means_comparison.png', dpi=600)

### linear regression
# libraries for regression and plotting
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# list of score pairs to compare
score_pairs = [
    ("semantic_fluency_score_m", "semantic_fluency_score"),
    ("phonemic_fluency_score_m", "phonemic_fluency_score"),
    ("picture_naming_score_m", "picture_naming_score")
]

# loop through each score pair and run regression
for score_m, score_auto in score_pairs:
    # data for regression (manually computed score x, automatically computed score y)
    X = df_merged[[score_m]]
    y = df_merged[score_auto]

    model = LinearRegression() # initialize Linear Regression model

    model.fit(X, y) # fit model

    y_pred = model.predict(X) # regression line -> predicted values

    plt.figure(figsize=(8, 6)) # plot for current score pair

    # scatter plot of original data points
    sns.scatterplot(x=X[score_m], y=y, label="Data points", color='b')

    sns.lineplot(x=X[score_m], y=y_pred, label="Regression Line", color='orange') # regression line

    # add perfect alignment, diagonal line (x = y)
    lims = [min(X[score_m].min(), y.min()), max(X[score_m].max(), y.max())]  # limits for diagonal
    plt.plot(lims, lims, '--', color='gray', label="Perfect Alignment (x = y)")  # add diagonal line to plot

    # labels and title
    plt.xlabel(f'{score_m} (Manual)')
    plt.ylabel(f'{score_auto} (Automatic)')
    plt.title(f'Linear Regression: {score_m} vs. {score_auto}')
    plt.legend() # Show legend

    # save plot
    plot_filename = f"/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/{score_auto}_regression.png"
    plt.savefig(plot_filename, dpi=600)

    # print regression results for current pair
    print(f"Linear regression results for {score_m} vs. {score_auto}:")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(f"R^2 score: {model.score(X, y)}")

#### intraclass correlation

import pingouin as pg
import pandas as pd

icc_results_list = [] # initialize list to store results

# loop through each score pair and compute ICC
for score_m, score_auto in score_pairs:
    # reshape data to long format
    df_long = pd.DataFrame({
        "subject": list(df_merged.index) * 2,  # subjects repeated twice (once per rater)
        "rater": ["Manual"] * len(df_merged) + ["Automatic"] * len(df_merged),
        "score": list(df_merged[score_m]) + list(df_merged[score_auto])
    })

    # compute ICC
    icc_results = pg.intraclass_corr(data=df_long, targets="subject", raters="rater", ratings="score")

    # extract ICC(2,1) (absolute agreement) & confidence interval
    icc_value = icc_results[icc_results["Type"] == "ICC2"]["ICC"].values[0]
    confidence_interval = icc_results[icc_results["Type"] == "ICC2"][["CI95%"]].values[0][0]

    # store results in a list
    icc_results_list.append({
        "Score Pair": f"{score_m} vs. {score_auto}",
        "ICC(2,1)": round(icc_value, 3),
        "95% CI": confidence_interval
    })

# convert results list to DataFrame and save to CSV file
df_icc = pd.DataFrame(icc_results_list) # convert to data frame
icc_file_path = "/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/icc_results.csv" # define file path
df_icc.to_csv(icc_file_path, index=False) # save to CSV file
print(f"ICC results saved to: {icc_file_path}")

### CSV file with all ICC + regression results

import pingouin as pg
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

all_results = [] # create an empty list to store all results (ICC + Regression)

# loop through each score pair
for score_m, score_auto in score_pairs:
    score_type = score_m.replace("_score_m", "")  # extract base name

    # compute ICC
    df_icc = pd.DataFrame({
        "subject": list(range(len(df_merged))) * 2,  # duplicate each subject ID so that both raters ("manual" and "automatic") are properly matched
        "rater": ["manual"] * len(df_merged) + ["automatic"] * len(df_merged),
        "score": pd.concat([df_merged[score_m], df_merged[score_auto]], ignore_index=True)
    })

    icc_all = pg.intraclass_corr(data=df_icc, targets="subject", raters="rater", ratings="score")

    # compute linear regression
    X = df_merged[[score_m]]  # independent variable (manual score)
    y = df_merged[score_auto]  # dependent variable (automatic score)

    model = LinearRegression()
    model.fit(X, y)

    regression_coef = model.coef_[0]
    regression_intercept = model.intercept_
    regression_r2 = model.score(X, y)

    # store results
    for _, row in icc_all.iterrows():
        all_results.append({
            "score_type": score_type,
            "ICC_type": row["Type"],
            "ICC": row["ICC"],
            "CI95%_lower": row["CI95%"][0],
            "CI95%_upper": row["CI95%"][1],
            "F": row["F"],
            "df1": row["df1"],
            "df2": row["df2"],
            "p_value": row["pval"],
            "Regression_Coefficient": regression_coef if row["Type"] == "ICC2" else None,  # store only once
            "Regression_Intercept": regression_intercept if row["Type"] == "ICC2" else None,
            "Regression_R2": regression_r2 if row["Type"] == "ICC2" else None
        })

# convert results to data frame and save to CSV file
results_df = pd.DataFrame(all_results) # convert to data frame
results_df.to_csv("/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/icc_regression_results.csv", index=False) # save to CSV



### only ICC2,1 and ICC3,1 but with correlation coefficients -

import pingouin as pg
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

all_results = []  # create an empty list to store all results (ICC + Regression + Correlation)

# loop through each score pair
for score_m, score_auto in score_pairs:
    score_type = score_m.replace("_score_m", "")  # extract base name

    # Compute ICC
    df_icc = pd.DataFrame({
        "subject": list(range(len(df_merged))) * 2,  # duplicate each subject ID so that both raters ("manual" and "automatic") are properly matched
        "rater": ["manual"] * len(df_merged) + ["automatic"] * len(df_merged),
        "score": pd.concat([df_merged[score_m], df_merged[score_auto]], ignore_index=True)
    })

    icc_all = pg.intraclass_corr(data=df_icc, targets="subject", raters="rater", ratings="score")

    # Compute linear regression
    X = df_merged[[score_m]]  # independent variable (manual score)
    y = df_merged[score_auto]  # dependent variable (automatic score)

    model = LinearRegression()
    model.fit(X, y)

    regression_coef = model.coef_[0]
    regression_intercept = model.intercept_
    regression_r2 = model.score(X, y)

    # Compute correlation coefficient
    correlation_coeff = np.corrcoef(df_merged[score_m], df_merged[score_auto])[0, 1]

    # Store results for ICC (2,1) and ICC (3,1)
    for _, row in icc_all.iterrows():
        if row["Type"] in ["ICC2", "ICC3"]:
            all_results.append({
                "score_type": score_type,
                "ICC_type": row["Type"],
                "ICC": row["ICC"],
                "CI95%_lower": row["CI95%"][0],
                "CI95%_upper": row["CI95%"][1],
                "F": row["F"],
                "df1": row["df1"],
                "df2": row["df2"],
                "p_value": row["pval"],
                "Regression_Coefficient": regression_coef if row["Type"] == "ICC2" else None,  # store only once
                "Regression_Intercept": regression_intercept if row["Type"] == "ICC2" else None,
                "Regression_R2": regression_r2 if row["Type"] == "ICC2" else None,
                "Correlation_Coefficient": correlation_coeff  # Include correlation coefficient for both types
            })

# Convert results to data frame and save to CSV file
results_df = pd.DataFrame(all_results)  # convert to data frame
results_df.to_csv("/Users/gilanorup/Desktop/Studium/MSc/MA/Score_Validierung/icc_regression_correlation_results.csv", index=False)  # save to CSV

