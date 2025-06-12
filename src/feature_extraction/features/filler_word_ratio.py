# feature idea: Hamrick et al. (2023); Vincze et al. (2022)

# sources for chosen filler words: Hamrick_et_al_2023; James_&_Goring_2018; Horton_et_al_2011; Asgari_et_al_2017;
# Tucker_&_Mukai_2022; Wright_2016; Mortensen_et_al_2006

import re # regular expression module for word tokenization
from feature_extraction.features import n_words, clean_text

def filler_word_ratio(text):
    cleaned = clean_text(text)
    # custom list of common filler words / phrases
    filler_words = {"um", "uh", "err", "hmm", "like", "well", "you know", "i mean"}

    total_words = n_words(text)
    if total_words == 0:
        return None

    filler_count = sum(
        len(re.findall(r'\b' + re.escape(phrase) + r'\b', cleaned)) # counts occurrence of phrases
        for phrase in filler_words
    )

    return filler_count / total_words # ratio of filler words to total words (None if text is empty)

# Note: overlapping filler expressions may be double-counted if filler_words contain subphrases