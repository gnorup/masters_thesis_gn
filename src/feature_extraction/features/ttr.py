# feature idea: Hamrick et al. (2023)

# Type-Token Ratio -> lexical diversity; number of unique words / total number of words

from feature_extraction.features import n_words

def ttr(text):
    words = n_words(text, return_words=True) # uses list from n_words
    return len(set(words)) / len(words) if words else None
    # set(words) keeps only unique words, len(words) total number of words -> TTR = types / tokens
