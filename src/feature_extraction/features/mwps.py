import sys # for file-paths and directories
import os # for file-paths and directories
import re  # for text preprocessing using regular expressions
import stanza  # for NLP processing, including sentence segmentation

# add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # folder of current script -> go to parent directory -> get full path

from feature_extraction.features.n_words import n_words
from feature_extraction.features.n_sentences import n_sentences

# load Stanza pipeline with sentence segmentation and tokenization
stanza.download("en")
nlp = stanza.Pipeline(lang="en", processors="tokenize", tokenize_pretokenized=False, tokenize_no_ssplit=False)

# Mean Words Per Sentence Analysis
def mwps(text):
    """Calculate mean words per sentence (MWPS)."""
    if isinstance(text, float):  # handle NaN values
        text = ""

    words = n_words(text)  # get word count
    sentences = n_sentences(text)  # get sentence count

    return round(words / sentences, 2) if sentences > 0 else 0  # calculate MWPS, round to 2 decimal places (+ prevents division by zero)
