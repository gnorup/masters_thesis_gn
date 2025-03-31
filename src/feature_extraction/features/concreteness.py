# feature idea: Biran et al. (2023)

# concreteness score = average concreteness of the words used, based on human-rated norms
# Brysbaert lexicon: abstract words (1) to concrete words (5)

import os
import pandas as pd # to load csv lexicon

from config.constants import GIT_DIRECTORY
from feature_extraction.features import n_words


def load_concreteness_lexicon():
    path = os.path.join(GIT_DIRECTORY, "resources/Concreteness_Ratings_Brysbaert_et_al.xlsx")
    df = pd.read_excel(path) # loads concreteness ratings from given path
    return dict(zip(df["Word"].str.lower(), df["Conc.M"])) # converts data frame into dictionary with lowercase words & values

def concreteness_score(text, lexicon): # function with text and lexicon (dictionary created) as arguments
    """
    Calculates the average concreteness score of the words in the text,
    using the Brysbaert concreteness lexicon.
    """
    words = n_words(text, return_words=True)
    scores = [lexicon[w] for w in words if w in lexicon] # if word exists in lexicon -> take score
    return sum(scores) / len(scores) if scores else None # returns average of scores
