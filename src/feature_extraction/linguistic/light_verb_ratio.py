import spacy
from feature_extraction.linguistic import clean_text

nlp = spacy.load("en_core_web_sm")

# list of light verbs based on Breedin et al. (1998)
LIGHT_VERBS = {"be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put"}

def light_verb_ratio(text):
    """
    Calculate the proportion of light verbs (Breedin et al., 1998) among all verbs in the transcript.
    Feature definition based on Fraser et al. (2014).
    """
    text = clean_text(text)
    doc = nlp(text)

    total_words = 0
    verb_count = 0
    light_verb_count = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "VERB":
                verb_count += 1
                if token.lemma_.lower() in LIGHT_VERBS:
                    light_verb_count += 1

    if total_words == 0:
        return None

    if verb_count > 0:
        return light_verb_count / verb_count
    else:
        return 0