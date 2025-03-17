import os  # for handling file paths and directories
import pandas as pd  # for handling CSV files efficiently
import re  # regular expressions module, clean and manipulate text

def word_count(text):
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'^"|"$', '', text)  # Remove leading/trailing quotation marks
    text = re.sub(r'\b(um|uh)\b', '', text)  # Remove filler words 'um' and 'uh'
    text = re.sub(r'[^\w\s\'-]', '', text)  # Remove punctuation, keep apostrophes and hyphens
    words = re.split(r'[ ,\.]+', text)  # Split by spaces, commas, or periods
    words = [word for word in words if word]  # Remove empty strings (from excessive splits)
    return len(words)  # Return the number of words in the list

def process_word_count():
    # Define task
    task = "cookieTheft"

    # Define base and output directories
    base_dir = "/Volumes/methlab_data/LanguageHealthyAging/data"
    output_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/results/features"
    os.makedirs(output_dir, exist_ok=True)

    # define output file
    output_file = os.path.join(output_dir, f"{task}.csv")

    # Initialize list to store combined data
    word_count_data = []

    # Loop through folder numbers
    for folder_number in range(41, 1371):
        folder_path = os.path.join(base_dir, str(folder_number), "ASR")
        file_path = os.path.join(folder_path, "transcriptions.csv")  # Build full path to each subject's file

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_row = df[df['task'] == task]['text_google']

            if not text_row.empty:
                text = text_row.iloc[0]
                if isinstance(text, float):  # Handle NaN values
                    text = ""
                word_count_value = word_count(text)
            else:
                word_count_value = None  # If task is missing, append None

            word_count_data.append([folder_number, word_count_value])

    # Check if the file already exists
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
    else:
        df_existing = pd.DataFrame()

    # Ensure Subject_ID is present
    df_word_count = pd.DataFrame(word_count_data, columns=["Subject_ID", "n_words"])
    if "Subject_ID" not in df_existing:
        df_existing["Subject_ID"] = df_word_count["Subject_ID"]

    # Merge new feature
    df_existing = df_existing.merge(df_word_count, on="Subject_ID", how="outer")

    # Save updated file
    df_existing.to_csv(output_file, index=False)
    print(f"Updated {output_file} with n_words")

if __name__ == "__main__":
    process_word_count()
    print("Word count analysis complete")