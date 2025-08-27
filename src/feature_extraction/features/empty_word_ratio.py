# source: Hamrick et al. (2023)

from feature_extraction.features import tokenize

empty_words = {"thing", "place", "stuff"} # list by Hamrick et al. (2023)

def empty_word_ratio(text):
    words = tokenize(text)
    total_words = len(words)

    if total_words == 0:
        return None

    count = sum(1 for w in words if w in empty_words)
    return count / total_words
