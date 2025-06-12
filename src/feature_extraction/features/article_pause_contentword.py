# number of pauses that follow an article and precede content words (feature idea: Vincze et al., 2022 -> might indicate word-finding difficulties)
# pause definition consistent with other features: Fraser et al. (2014)

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def pause_after_article(word_timestamp_file, pause=0.15):
    df = pd.read_csv(word_timestamp_file)

    full_text = " ".join(df["word"].tolist()) # combine words for POS tagging
    doc = nlp(full_text)
    df["pos"] = [token.pos_ for token in doc] # new column with POS tags

    article = {"DET"}
    content_words = {"NOUN", "VERB", "ADJ"}

    article_pause_count = 0

    for i in range(len(df)-1):
        # go through consecutive words (to find cases where article -> content-word)
        current_word = df.loc[i, "pos"]
        next_word = df.loc[i+1, "pos"]
        # check if silence between words meets pause-criteria (start of next - end of current)
        pause_duration = df.loc[i+1, "start"] - df.loc[i, "end"]
        # check if pause is after article and before content-word -> count those occurrences
        if current_word in article and pause_duration > pause and next_word in content_words:
            article_pause_count += 1

    return article_pause_count