import re
from feature_extraction.linguistic import n_words, clean_text

def filler_word_ratio(text):
    """
    Calculate the proportion of filler words in the transcript.
    Feature definition based on Hamrick et al. (2023).
    """
    cleaned = clean_text(text)

    # custom list of common filler words / phrases; based on Hamrick et al. (2023), James & Goring (2018),
    # Horton et al. (2010), Asgari et al., 2017, Tucker & Mukai (2022), Wright (2016) and Mortensen et al. (2006).
    filler_words = {"um", "uh", "err", "hmm", "like", "well", "you know", "i mean"}

    total_words = n_words(text)
    if total_words == 0:
        return None

    filler_count = sum(
        len(re.findall(r'\b' + re.escape(phrase) + r'\b', cleaned)) # count occurrence of phrases
        for phrase in filler_words
    )

    return filler_count / total_words

# Note: overlapping filler expressions may be double-counted if filler_words contain subphrases