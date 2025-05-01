# source: Fraser et al., 2016

from .lexical_diversity_features import tokenize

def average_word_length(text):
    words = tokenize(text)
    if not words:
        return 0
    total_length = sum(len(w) for w in words)
    return total_length / len(words)