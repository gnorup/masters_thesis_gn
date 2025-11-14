from feature_extraction.linguistic import tokenize

def mattr(text, window_sizes=None):
    """
    Calculate Moving-Average Type-Token Ratio (MATTR) for multiple window sizes.
    Window sizes based on Covington & McFall (2010) and Fraser et al. (2014).
    Feature definition based on Fraser et al. (2014) and Huang et al. (2024).
    """
    if window_sizes is None:
        window_sizes = [10, 20, 30, 40, 50]

    words = tokenize(text)

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