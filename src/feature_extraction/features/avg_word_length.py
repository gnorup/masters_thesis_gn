# source: Fraser et al. (2016); Robin et al. (2023)

from feature_extraction.features import tokenize

def avg_word_length(text):
    words = tokenize(text)
    if not words:
        return None
    total_length = sum(len(w) for w in words)
    return total_length / len(words)