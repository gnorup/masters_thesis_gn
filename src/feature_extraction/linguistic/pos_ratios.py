import spacy

from feature_extraction.linguistic import clean_text

# load spaCy model (install with `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

def pos_ratios_spacy(text):
    """
    Compute the following POS ratios using spaCy:
    - Ratios for all POS categories (based on Lindsay et al., 2021; Fraser et al., 2016; Huang et al., 2024),
        with strong evidence for pronoun-ratio (Robin et al., 2023; Slegers et al., 2018; Fraser et al., 2014)
    - Structural ratios:
        NOUN/VERB (based on Lindsay et al., 2021; Fraser et al., 2014, 2016; Vincze et al., 2022)
        PRON/NOUN (based on Lindsay et al., 2021; Slegers et al., 2018; Fraser et al., 2016)
        DET/NOUN (based on Lindsay et al., 2021)
        AUX/VERB (based on Hernández-Domínguez et al., 2018)
    - Content density: open-to-closed class ratio (based on Lindsay et al., 2021; Beltrami et al., 2018)
    - Information words ratio (based on Huang et al., 2024)
    """

    text = clean_text(text)
    doc = nlp(text)

    # initialize POS counts
    counts = {
        "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0,
        "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0,
        "SCONJ": 0, "VERB": 0, "OTHER": 0
    }

    total_words = 0
    information_words = 0

    for token in doc:
        if token.is_alpha:  # only count actual words (ignore punctuation, numbers, etc.)
            total_words += 1
            if token.pos_ in ["NOUN", "VERB", "ADJ", "NUM"]:
                information_words += 1
            if token.pos_ in counts: # universal POS tag
                counts[token.pos_] += 1
            else:
                counts["OTHER"] += 1  # catch any unclassified tokens

    # compute POS ratios
    pos_ratios = {pos: (counts[pos] / total_words if total_words > 0 else None) for pos in counts}

    # compute structural linguistic ratios
    noun_verb_ratio = counts["NOUN"] / counts["VERB"] if counts["VERB"] > 0 else None
    pronoun_noun_ratio = counts["PRON"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    determiner_noun_ratio = counts["DET"] / counts["NOUN"] if counts["NOUN"] > 0 else None
    aux_verb_ratio = counts["AUX"] / counts["VERB"] if counts["VERB"] > 0 else None
    information_words_ratio = information_words / total_words if total_words > 0 else None

    # open-to-closed class ratio
    open_class_count = sum(counts[pos] for pos in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"])
    closed_class_count = sum(counts[pos] for pos in ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"])
    open_closed_ratio = open_class_count / closed_class_count if closed_class_count > 0 else None

    # combine all ratios in a single dictionary
    pos_ratios.update({
        "noun_verb_ratio": noun_verb_ratio,
        "pronoun_noun_ratio": pronoun_noun_ratio,
        "determiner_noun_ratio": determiner_noun_ratio,
        "aux_verb_ratio": aux_verb_ratio,
        "content_density": open_closed_ratio,
        "information_words_ratio": information_words_ratio
    })

    return pos_ratios