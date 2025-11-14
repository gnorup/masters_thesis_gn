import pandas as pd
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

def article_pause_contentword(word_timestamp_file, pause_threshold=0.15):
    """
    Count the number of cases where pauses follow articles and precede content words.
    Feature concept based on Vincze et al. (2022).
    """
    df = pd.read_csv(word_timestamp_file)

    # pos-tagging on entire transcription
    tokens = df["word"].astype(str).tolist()
    doc = Doc(nlp.vocab, words=tokens)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    tags = [token.tag_ for token in doc]

    # add pos-column
    if len(tags) != len(df):
        print(f"mismatch: {len(tags)} tags vs {len(df)} words -> check")
        return None

    df["pos"] = tags

    # add pos-row "PAUSE" where duration between words > threshold
    pause_rows = []
    for i in range(1, len(df)):
        pause = df.loc[i, "start"] - df.loc[i - 1, "end"]
        if pause > pause_threshold:
            pause_row = {
                "word": "[pause]",
                "start": df.loc[i - 1, "end"],
                "end": df.loc[i, "start"],
                "pos": "PAUSE"
            }
            pause_rows.append((i, pause_row))

    if pause_rows:
        for idx, pause_row in reversed(pause_rows):
            df = pd.concat([df.iloc[:idx], pd.DataFrame([pause_row]), df.iloc[idx:]], ignore_index=True)

    # define pos_categories
    def categorize(tag):
        if tag in {"UH", "PAUSE"}:
            return "PAUSE"
        elif tag in {"DT"}:
            return "ARTICLE"
        elif tag in {"NN", "NNS", "NNP", "NNPS", "VBG", "VBN", "JJ", "JJR", "JJS"}:
            return "CONTENT"
        else:
            return "OTHER"

    df["pos_category"] = df["pos"].apply(categorize)

    # count patterns
    sequence = df["pos_category"].tolist()
    patterns = [
        ["ARTICLE", "PAUSE", "CONTENT"],
        ["ARTICLE", "PAUSE", "PAUSE", "CONTENT"],
        ["ARTICLE", "PAUSE", "PAUSE", "PAUSE", "CONTENT"],
        ["ARTICLE", "PAUSE", "ARTICLE", "CONTENT"],
        ["ARTICLE", "PAUSE", "ARTICLE", "PAUSE", "CONTENT"]
    ]

    article_pause_count = 0
    for i in range(len(sequence)):
        for pattern in patterns:
            if sequence[i:i + len(pattern)] == pattern:
                article_pause_count += 1
                break

    return article_pause_count