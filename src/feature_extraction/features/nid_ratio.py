# source: Fraser et al. (2014)
# frequency of words not in dictionary (using NLTK)

import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('words')
from nltk.corpus import words as nltk_words

from feature_extraction.features import tokenize, clean_text

# load english dictionary once
english_vocab = set(w.lower() for w in nltk_words.words())

def nid_ratio(text):
    text = clean_text(text)
    words = tokenize(text)
    total_words = len(words)

    if total_words == 0:
        return None

    doc = nlp(" ". join(words))
    lemmatized_words = [token.lemma_.lower() for token in doc]

    not_in_dict = sum(1 for w in lemmatized_words if w not in english_vocab)
    return not_in_dict / total_words

    # for debugging

    # not_in_dict_words = [w for w in words if w not in english_vocab]
    # nid = len(not_in_dict_words) / total_words

    # print("words not in dictionary:", not_in_dict_words)
    # print(f"NID ratio: {nid}")

    # return nid



