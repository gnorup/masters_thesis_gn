# sources for features:
# Brunét's Index: Hamrick et al. (2023); Fraser et al. (2014); Huang et al. (2024)
# Honoré Statistic: Hamrick et al. (2023); Fraser et al. (2014); Huang et al. (2024)
# Guiraud's Statistic: Vermeer (2000)

import numpy as np

from feature_extraction.features import tokenize


def brunets_index(text):
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_tokens ** (n_types ** (-0.165))
    else:
        return None

def honores_statistic(text):
    words = tokenize(text)
    n_words = len(words)
    n_types = len(set(words))
    n_words_with_one_occurrence = len([w for w in set(words) if words.count(w) == 1])

    denominator = (1 - n_words_with_one_occurrence / n_types) if n_types != 0 else None

    if n_words == 0 or denominator == 0 or denominator is None:
        return None
    else:
        return (100 * np.log(n_words)) / denominator

def guirauds_statistic(text):
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_types / np.sqrt(n_tokens)
    else:
        return None