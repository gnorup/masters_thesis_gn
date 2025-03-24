# concreteness score = average concreteness of the words used, based on human-rated norms
# Brysbaert lexicon: abstract words (1) to concrete words (5)

import re
import os
import pandas as pd # to load csv lexicon

from config.constants import GIT_DIRECTORY


def load_concreteness_lexicon():
    path = os.path.join(GIT_DIRECTORY, "resources/Concreteness_Ratings_Brysbaert_et_al.xlsx")
    df = pd.read_excel(path) # loads concreteness ratings from given path
    return dict(zip(df["Word"].str.lower(), df["Conc.M"])) # converts data frame into dictionary with lowercase words & values

def concreteness_score(text, lexicon): # function with text and lexicon (dictionary created) as arguments
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)] # extracts words and converts them to lowercase
    scores = [lexicon[w] for w in words if w in lexicon] # if word exists in lexicon -> take score
    return sum(scores) / len(scores) if scores else None # returns average of scores
