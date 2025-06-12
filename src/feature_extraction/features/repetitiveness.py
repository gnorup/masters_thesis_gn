# feature idea: Shankar et al. (2025); Slegers et al. (2018)
# immediate repetitions (adjacent words): de Lira et al. (2011); Croisile et al. (1996)

from feature_extraction.features import tokenize

def adjacent_repetitions(text):
    words = tokenize(text)
    total_words = len(words)

    if total_words == 0:
        return None

    repetitions = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
    return repetitions / total_words