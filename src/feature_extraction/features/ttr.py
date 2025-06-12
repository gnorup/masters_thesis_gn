# feature idea: Fraser et al. (2014); Hamrick et al. (2023)
# (also used by Huang et al., 2024; Beltrami et al., 2018)

# Type-Token Ratio -> lexical diversity; number of unique words / total number of words

from feature_extraction.features import tokenize

def ttr(text):
    words = tokenize(text)
    if not words:
        return None
    return len(set(words)) / len(words)
    # set(words) keeps only unique words, len(words) total number of words -> TTR = types / tokens
