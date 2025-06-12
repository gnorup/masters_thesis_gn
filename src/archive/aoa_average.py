# feature idea: Fraser et al. (2014, 2016); Forbes-McKay (2004)

# Age of Acquisition (AoA) average (average age when people typically learn a word), defined by Kuperman et al. (2012)) Lexicon
# -> idea that more complex vocabulary if higher

import os
import pandas as pd # to load AoA dataset

from config.constants import GIT_DIRECTORY
from feature_extraction.features import tokenize


def load_aoa_lexicon():
    path = os.path.join(GIT_DIRECTORY, "resources/AoA_ratings_Kuperman_et_al_BRM.xlsx")
    df = pd.read_excel(path) # load AoA lexicon
    return dict(zip(df["Word"].str.lower(), df["Rating.Mean"])) # creates dictionary for words and their average AoA rating

def aoa_average(text, lexicon):
    words = tokenize(text)
    scores = [lexicon[w] for w in words if w in lexicon] # loops through all words in the text -> if word exists in lexicon, AoA score is included
    return sum(scores) / len(scores) if scores else None # returns average AoA score
