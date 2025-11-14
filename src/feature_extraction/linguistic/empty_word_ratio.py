from feature_extraction.linguistic import tokenize

EMPTY_WORDS = {"thing", "place", "stuff"} # list by Hamrick et al. (2023)

def empty_word_ratio(text):
    """
    Calculate the proportion of empty words in the transcript.
    Feature definition based on Hamrick et al. (2023).
    """
    words = tokenize(text)
    total_words = len(words)

    if total_words == 0:
        return None

    count = sum(1 for w in words if w in EMPTY_WORDS)
    return count / total_words