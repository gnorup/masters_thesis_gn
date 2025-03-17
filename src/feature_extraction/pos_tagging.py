# THIS SHIT DOESNT WORK
import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure NLTK resources are downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Mapping NLTK POS tags to general categories
POS_MAP = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",  # Nouns
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",  # Verbs
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",  # Adjectives
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",  # Adverbs
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON"  # Pronouns
}

# Function to preprocess text and count POS tags
def count_pos_nltk(text):
    text = text.strip()
    text = re.sub(r'^"|"$', '', text)  # Remove leading/trailing quotation marks
    text = re.sub(r'[^\w\s\'-]', '', text)  # Remove punctuation, keep apostrophes and hyphens

    words = word_tokenize(text)  # Tokenize text
    tagged_words = pos_tag(words)  # Get POS tags

    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0, "OTHER": 0}

    total_words = len(words)
    for _, tag in tagged_words:
        category = POS_MAP.get(tag, "OTHER")  # Map POS tags
        pos_counts[category] += 1

    # Compute ratios
    if total_words > 0:
        for key in pos_counts:
            pos_counts[key] = pos_counts[key] / total_words
    else:
        pos_counts = {k: None for k in pos_counts}

    return pos_counts

def process_pos_counts_nltk():
    task = "cookieTheft"
    base_dir = "/Volumes/methlab_data/LanguageHealthyAging/data"
    output_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/results/features"
    output_file = os.path.join(output_dir, "pos_ratios_nltk.csv")
    os.makedirs(output_dir, exist_ok=True)

    pos_data = []
    for folder_number in range(41, 1371):
        folder_path = os.path.join(base_dir, str(folder_number), "ASR")
        file_path = os.path.join(folder_path, "transcriptions.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_row = df[df['task'] == task]['text_google']

            if not text_row.empty:
                text = text_row.iloc[0]
                if isinstance(text, float):  # Handle NaN values
                    text = ""
                pos_counts = count_pos_nltk(text)
            else:
                pos_counts = {k: None for k in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "OTHER"]}
        else:
            pos_counts = {k: None for k in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "OTHER"]}

        pos_data.append([folder_number] + list(pos_counts.values()))

    columns = ["Subject_ID"] + [f"{pos}_{task}" for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "OTHER"]]
    df_pos = pd.DataFrame(pos_data, columns=columns)
    df_pos.to_csv(output_file, index=False)

    print("POS ratio analysis using NLTK complete")

if __name__ == "__main__":
    process_pos_counts_nltk()
    print("POS count and ratio analysis using NLTK complete")
