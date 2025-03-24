import spacy # nlp library

# load spaCy model, includes POS tagger
nlp = spacy.load("en_core_web_sm")

def count_pos_spacy(text):
    """
    Tokenize text and compute POS counts using spaCy.
    The following tags are included:
    ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN,
    PUNCT, SCONJ, SYM, VERB, and OTHER (X).
    """

    if isinstance(text, float):  # handle NaN values
        text = ""

    doc = nlp(text) # pass text into spacy pipeline

    # initialize POS counts
    pos_counts = {
        "ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0,
        "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0,
        "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0, "OTHER": 0
    }

    # count occurrences of each POS -> loop through each token in text
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        else:
            pos_counts["OTHER"] += 1  # everything else goes into 'OTHER'

    return pos_counts
