# age of acquisition (AoA) average (average age when people typically learn a word) -> idea that more complex vocabulary if higher

import re
import os
import pandas as pd # to load AoA dataset

from config.constants import GIT_DIRECTORY


def load_aoa_lexicon():
    path = os.path.join(GIT_DIRECTORY, "src/files/AoA_ratings_Kuperman_et_al_BRM.xlsx")
    df = pd.read_excel(path) # load AoA lexicon
    return dict(zip(df["Word"].str.lower(), df["Rating.Mean"])) # creates dictionary for words and their average AoA rating

def aoa_average(text, lexicon): # define scoring function
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)] # extract words and ignores punctuation etc., converts it to lowercase
    scores = [lexicon[w] for w in words if w in lexicon] # loops through all words in the text -> if word exists in lexicon, AoA score is included
    return sum(scores) / len(scores) if scores else None # returns average AoA score
