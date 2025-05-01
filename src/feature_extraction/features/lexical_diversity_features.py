# sources for features: Hamrick et al., 2023, Vermeer, 2000

import numpy as np

from .n_words import clean_text


def tokenize(text):
    cleaned = clean_text(text)
    return cleaned.split()


def brunets_index(text):
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_tokens ** (n_types ** (-0.165))
    else:
        return 0

def honores_statistic(text):
    words = tokenize(text)
    n_words = len(words)
    n_types = len(set(words))
    n_words_with_one_occurrence = len([w for w in set(words) if words.count(w) == 1])

    if n_words == 0 or n_types == 0 or (1 - n_words_with_one_occurrence / n_types) == 0:
        return 0
    else:
        return (100 * np.log(n_words)) / (1 - n_words_with_one_occurrence / n_types)

def guirauds_statistic(text):
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_types / np.sqrt(n_tokens)
    else:
        return 0