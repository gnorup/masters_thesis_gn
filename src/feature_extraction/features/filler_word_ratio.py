import re # regular expression module for word tokenization

def filler_word_ratio(text, filler_words=None):
    """
    Calculates the proportion of filler words in a text.
    """
    if isinstance(text, float):  # handle NaN
        text = ""

    # todo: why these filler words?
    if filler_words is None:
        filler_words = {"um", "uh", "er", "hmm", "like", "well", "you know", "i mean", "so"} # custom list of common filler words / phrases

    # tokenize text to lowercase words
    # todo: isn't this the same as n_words?
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)] # gets actual words without punctuation and symbols
    total_words = len(words)

    # count filler words
    filler_count = 0
    i = 0
    while i < len(words): # loop through text word by word, index-based
        matched = False
        for phrase in sorted(filler_words, key=lambda x: -len(x.split())):  # check multi-word fillers first
            phrase_tokens = phrase.split() # slice current section of words list
            if words[i:i + len(phrase_tokens)] == phrase_tokens: # if slice matches phrase, it's counted as a filler
                filler_count += 1
                i += len(phrase_tokens) # skips ahead len(phrase_tokens) when matched, so that it doesn't double-count
                matched = True
                break
        if not matched:
            i += 1

    return filler_count / total_words if total_words > 0 else None # ratio of filler words to total words (None if text is empty)
