from feature_extraction.features import n_words

def speech_rate(text, total_duration): # transcription & length of audio in seconds from csv-file
    """
    Words per second (includes pauses).
    """
    word_count = n_words(text) # using the same word-count for consistency
    return word_count / total_duration if total_duration > 0 else None