# feature idea: Luz et al. (2021b); window size based on Cunningham & Haley (2020)

# Moving-Average Type-Token Ratio (MATTR)
# Calculates TTR in a sliding window and averages these local TTRs across the text.
# This smooths out the bias that occurs when TTR decreases in longer texts.

from feature_extraction.features import n_words

def mattr(text, window_size=50): # default: window of 50 words
    words = n_words(text, return_words=True) # uses word-list from n_words
    if len(words) < window_size: # returns none if text is too short to fill a single window
        return None
    ttrs = [
        len(set(words[i:i + window_size])) / window_size # unique words (types) / window size (tokens)
        for i in range(len(words) - window_size + 1) # slide window over text one word at a time
    ]
    return sum(ttrs) / len(ttrs) # averages local TTRs to get MATTR
