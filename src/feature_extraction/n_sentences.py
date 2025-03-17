import os  # for handling file paths and directories
import pandas as pd  # for handling CSV files efficiently
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

# Sentence Count Analysis
def process_sentence_counts():
    sentence_data = []
    for folder_number in range(41, 1371):
        folder_path = os.path.join(base_dir, str(folder_number), "ASR")
        file_path = os.path.join(folder_path, "transcriptions.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_row = df[df['task'] == task]['text_google']

            if not text_row.empty:
                text = text_row.iloc[0]
                if isinstance(text, float):  # Check for NaN values
                    text = ""
                sentences = len(nlp(str(text)).sentences)  # Ensure text is a string
            else:
                sentences = None

            sentence_data.append([folder_number, sentences])

    # check if file already exists
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
    else:
        df_existing = pd.DataFrame()

    # ensure Subject_ID is present
    df_sentences = pd.DataFrame(sentence_data, columns=["Subject_ID", "n_sentences"])
    if "Subject_ID" not in df_existing:
        df_existing["Subject_ID"] = df_sentences["Subject_ID"]

    # merge new feature
    df_existing = df_existing.merge(df_sentences, on="Subject_ID", how="outer")

    # save updated file
    df_existing.to_csv(output_file, index=False)
    print(f"updated {output_file} with n_sentences")

# Run sentence count analysis
if __name__ == "__main__":
    process_sentence_counts()
    print("sentence count analysis complete")