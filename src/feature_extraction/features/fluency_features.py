# source: Fraser et al. (2014, 2016)
# frequencies of different types of filled pauses (um, uh, ah, er)

from feature_extraction.features import tokenize

def filled_pause_ratio(text, target):
    words = tokenize(text)
    total_words = len(words)

    if total_words == 0:
        return None

    count = sum(1 for w in words if w == target)
    return count / total_words

def calculate_fluency_features(text):
    return{
        "um_ratio": filled_pause_ratio(text, "um"),
        "uh_ratio": filled_pause_ratio(text, "uh"),
        "er_ratio": filled_pause_ratio(text, "er"),
        "ah_ratio": filled_pause_ratio(text, "ah")
    }