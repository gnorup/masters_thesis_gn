# content words / all words -> measure how information-rich language is

import spacy
nlp = spacy.load("en_core_web_sm")

def lexical_density(text):
    doc = nlp(text) # passes input text through spacy pipeline -> doc contains tokenized and POS-tagged version of text
    content_words = [t for t in doc if t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and t.is_alpha] # content words from POS tags; ignoring numbers, punctuation etc.
    total_words = len([t for t in doc if t.is_alpha]) # counts all alphabetic tokens
    return len(content_words) / total_words if total_words > 0 else None # divides number of content words by total words
