# feature idea: Lindsay et al. (2021); Beltrami et al. (2018); Hernández-Domínguez et al. (2018)

import spacy
from scipy.stats import entropy

# load spaCy model (install with `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

def pos_ratios_spacy(text):
    """
    Compute the following POS ratios using spaCy:
    - Ratios for all POS categories (each count divided by total words)
    - Open-to-Closed Class Ratio (Open/Closed)
    - Structural ratios: NOUN/VERB, PRON/NOUN, DET/NOUN, AUX/VERB
    - POS Entropy (syntactic variety)
    - Lexical density (content words / all words)
    """

    if isinstance(text, float):  # handle NaN values
        text = ""

    doc = nlp(text)

    # initialize POS counts
    counts = {
        "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0,
        "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0,
        "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0, "OTHER": 0
    }

    total_words = 0  # word counter
    content_words = 0 # for lexical density

    for token in doc:
        if token.is_alpha:  # only count actual words (ignore punctuation, numbers, etc.)
            total_words += 1
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                content_words += 1
            if token.pos_ in counts: # gets universal POS tag
                counts[token.pos_] += 1
            else:
                counts["OTHER"] += 1  # catch any unclassified tokens

    # compute POS ratios
    pos_ratios = {pos: (counts[pos] / total_words if total_words > 0 else None) for pos in counts}

    # compute lexical density
    lexical_density = content_words / total_words if total_words > 0 else None

    # compute structural linguistic ratios
    noun_verb_ratio = counts["NOUN"] / counts["VERB"] if counts["VERB"] > 0 else None
    pronoun_noun_ratio = counts["PRON"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    determiner_noun_ratio = counts["DET"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    aux_verb_ratio = counts["AUX"] / counts["VERB"] if counts["VERB"] > 0 else None # how often auxiliary verbs support full verbs (for grammatical complexity)

    # Open-to-Closed Class Ratio
    open_class_count = sum(counts[pos] for pos in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]) # words that carry content / meaning
    closed_class_count = sum(counts[pos] for pos in ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]) # more grammatical words
    open_closed_ratio = open_class_count / closed_class_count if closed_class_count > 0 else None # high = more semantic richness; low = more functional / grammatical speech

    # combine all ratios in a single dictionary
    pos_ratios.update({
        "NOUN/VERB": noun_verb_ratio,
        "PRON/NOUN": pronoun_noun_ratio,
        "DET/NOUN": determiner_noun_ratio,
        "AUX/VERB": aux_verb_ratio,
        "OPEN/CLOSED": open_closed_ratio,
        "LEXICAL_DENSITY": lexical_density
    })

    return pos_ratios
