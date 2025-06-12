# feature idea: Lindsay et al. (2021)
# sources: (DOUBLE-CHECK DEFINITIONS)
# POS ratios: Lindsay et al. (2021); Fraser et al. (2016); Huang et al. (2024)
# -> for pronoun: Robin et al. (2023); Slegers et al. (2018); Fraser et al. (2014)
# noun/verb ratio: Lindsay et al. (2021); Fraser et al. (2014); Fraser et al. (2016); Vincze et al. (2022)
# pronoun/noun ratio: Lindsay et al. (2021); Slegers et al. (2018); Fraser et al. (2016); Vincze et al. (2022)
# determiner/noun ratio: Lindsay et al. (2021)
# auxiliary/verb ratio: Hernández-Domínguez et al. (2018)
# open-to-closed class ratio: Lindsay et al. (2021); Petti et al. (2020)
# information words ratio: Huang et al. (2024)

import spacy

from feature_extraction.features import clean_text

# load spaCy model (install with `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

def pos_ratios_spacy(text):
    """
    Compute the following POS ratios using spaCy:
    - Ratios for all POS categories (each count divided by total words)
    - Structural ratios: NOUN/VERB, PRON/NOUN, DET/NOUN, AUX/VERB
    - Open-to-Closed Class Ratio (Open/Closed)
    """

    text = clean_text(text)
    doc = nlp(text)

    # initialize POS counts
    counts = {
        "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0,
        "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0,
        "SCONJ": 0, "VERB": 0, "OTHER": 0
    }

    total_words = 0  # word counter
    information_words = 0

    for token in doc:
        if token.is_alpha:  # only count actual words (ignore punctuation, numbers, etc.)
            total_words += 1
            if token.pos_ in ["NOUN", "VERB", "ADJ", "NUM"]:
                information_words += 1
            if token.pos_ in counts: # gets universal POS tag
                counts[token.pos_] += 1
            else:
                counts["OTHER"] += 1  # catch any unclassified tokens

    # compute POS ratios
    pos_ratios = {pos: (counts[pos] / total_words if total_words > 0 else None) for pos in counts}

    # compute structural linguistic ratios
    noun_verb_ratio = counts["NOUN"] / counts["VERB"] if counts["VERB"] > 0 else None
    pronoun_noun_ratio = counts["PRON"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    determiner_noun_ratio = counts["DET"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    aux_verb_ratio = counts["AUX"] / counts["VERB"] if counts["VERB"] > 0 else None # how often auxiliary verbs support full verbs (for grammatical complexity)
    information_words_ratio = information_words / total_words if information_words > 0 else None

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
        "INFORMATION_WORDS": information_words_ratio
    })

    return pos_ratios
