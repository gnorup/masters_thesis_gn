from feature_extraction.linguistic import tokenize

def avg_word_length(text):
    """
    Calculate the average number of letters per word in the transcript.
    Feature based on Fraser et al. (2016) and Robin et al. (2023).
    """
    words = tokenize(text)
    if not words:
        return None
    total_length = sum(len(w) for w in words)
    return total_length / len(words)