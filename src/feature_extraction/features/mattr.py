# feature idea: Luz et al. (2021b); window size based on Fraser et al. (2014) & Covington & McFall (2010)

# Moving-Average Type-Token Ratio (MATTR)
# Calculates TTR in a sliding window and averages these local TTRs across the text.
# This smooths out the bias that occurs when TTR decreases in longer texts.

from feature_extraction.features import n_words

def mattr(text, window_sizes=[10, 20, 30, 40, 50]): # calculate MATTR with multiple window sizes
    words = n_words(text, return_words=True) # uses word-list from n_words
    if not words:
        return {f"mattr_{w}": None for w in window_sizes}
    results = {}
    for window_size in window_sizes:
        if len(words) < window_size: # returns none if text is too short to fill a single window
            results[f"mattr_{window_size}"] = None
        else:
            ttrs = [
                len(set(words[i:i + window_size])) / window_size # unique words (types) / window size (tokens)
                for i in range(len(words) - window_size + 1) # slide window over text one word at a time
            ]
            results[f"mattr_{window_size}"] = sum(ttrs) / len(ttrs) # averages local TTRs to get MATTR
    return results
