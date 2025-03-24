# underestimates sentence-count due to missing punctuation

import stanza # stanza library for tokenization, sentence splitting, POS tagging etc.

# load Stanza pipeline for sentence segmentation
stanza.download("en", processors="tokenize") # only for first time using it
nlp = stanza.Pipeline(lang="en", processors="tokenize", tokenize_pretokenized=False, tokenize_no_ssplit=False) # activate tokenizer module, allows sentence-splitting

def n_sentences(text):
    """Calculate the number of sentences in a given text using Stanza."""
    if isinstance(text, float):  # handle NaN values
        text = ""

    doc = nlp(text) # passes text to Stanza pipeline, returns document object
    return len(doc.sentences) # counts number of sentences stanza found in doc
