import numpy as np

from feature_extraction.linguistic import tokenize

def brunets_index(text):
    """
    Calculate Brunet's index (Brunet, 1978) as a measure of lexical diversity.
    Feature definition based on Fraser et al. (2014) and Hamrick et al. (2023).
    """
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_tokens ** (n_types ** (-0.165))
    else:
        return None

def honores_statistic(text):
    """
    Calculate Honoré's statistic (Honoré, 1979) as a measure of lexical diversity.
    Feature definition based on Fraser et al. (2014) and Hamrick et al. (2023).
    """
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    n_words_with_one_occurrence = len([w for w in set(words) if words.count(w) == 1])

    denominator = (1 - n_words_with_one_occurrence / n_types) if n_types != 0 else None

    if n_tokens == 0 or denominator == 0 or denominator is None:
        return None
    else:
        return (100 * np.log(n_tokens)) / denominator

def guirauds_statistic(text):
    """
    Calculate Guiraud's statistic (Guiraud, 1959) as a measure of lexical diversity.
    Feature definition based on Vermeer (2000).
    """
    words = tokenize(text)
    n_tokens = len(words)
    n_types = len(set(words))
    if n_tokens > 0:
        return n_types / np.sqrt(n_tokens)
    else:
        return None