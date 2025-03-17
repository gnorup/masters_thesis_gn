import os  # for handling file paths and directories
import pandas as pd  # for handling CSV files efficiently
import re  # for text preprocessing using regular expressions
import stanza  # for NLP processing, including sentence segmentation

# Load Stanza pipeline with sentence segmentation and tokenization
stanza.download("en")  # Ensure the English model is downloaded
nlp = stanza.Pipeline(lang="en", processors="tokenize", tokenize_pretokenized=False, tokenize_no_ssplit=False)

# Define task
task = "cookieTheft"

# Base and output directories
base_dir = "/Volumes/methlab_data/LanguageHealthyAging/data"
output_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/results/features"
os.makedirs(output_dir, exist_ok=True)

# Define the output file
output_file = os.path.join(output_dir, f"{task}.csv")

# Mean Words Per Sentence Analysis
def process_mwps():
    mwps_data = []
    for folder_number in range(41, 1371):
        folder_path = os.path.join(base_dir, str(folder_number), "ASR")
        file_path = os.path.join(folder_path, "transcriptions.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_row = df[df['task'] == task]['text_google']

            if not text_row.empty:
                text = text_row.iloc[0]
                if isinstance(text, float):  # Handle NaN values
                    text = ""  # Replace NaN with an empty string
                words = re.findall(r"\b\w+\b", str(text))  # Ensure text is a string
                sentences = len(nlp(text).sentences)
                mwps = round(len(words) / sentences, 2) if sentences > 0 else 0
            else:
                mwps = None

            mwps_data.append([folder_number, mwps])

    # check if file already exists
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
    else:
        df_existing = pd.DataFrame()

    # ensure Subject_ID is present
    df_mwps = pd.DataFrame(mwps_data, columns=["Subject_ID", "mwps"])
    if "Subject_ID" not in df_existing:
        df_existing["Subject_ID"] = df_mwps["Subject_ID"]

    # merge new feature
    df_existing = df_existing.merge(df_mwps, on="Subject_ID", how="outer")

    # save updated file
    df_existing.to_csv(output_file, index=False)
    print(f"updated {output_file} with mwps")

# Run MWPS analysis
if __name__ == "__main__":
    process_mwps()
    print("MWPS analysis complete")
