# moving average TTR -> calculate TTR in window and average TTRs over all windows (bc TTR drops just because text gets longer)

import re # regular expressions for word tokenization

def mattr(text, window_size=50): # text input; window 50 words for now
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)] # tokenizes text into words only, removes punctuation and converts to lowercase only
    if len(words) < window_size: # none if text is too short to fill a single window
        return None
    ttrs = []
    for i in range(len(words) - window_size + 1): # slides window of defined length over word list
        window = words[i:i+window_size]
        ttrs.append(len(set(window)) / window_size) # compute TTR for unique words in window-size
    return sum(ttrs) / len(ttrs) # averages local TTRs to get final MATTR
