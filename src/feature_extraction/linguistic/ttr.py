from feature_extraction.linguistic import tokenize

def ttr(text):
    """
    Calculate Type-Token Ratio (TTR; Miller, 1981) for words in the transcript.
    Feature definition based on Fraser et al. (2014) and Hamrick et al. (2023).
    """
    words = tokenize(text)
    if not words:
        return None

    return len(set(words)) / len(words)
    # set(words) keeps only unique words, len(words) total number of words -> TTR = types / tokens