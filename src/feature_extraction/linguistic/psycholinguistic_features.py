# Age of Acquisition (AoA), Frequency, Concreteness

import os
import pandas as pd
import spacy

from config.constants import GIT_DIRECTORY
from feature_extraction.linguistic import clean_text

nlp = spacy.load("en_core_web_sm")

def compute_avg_by_pos(text, lexicon, pos_tags=None):
    """
    Calculate the average value for different POS tags based on lexicon.
    Feature definition based on Fraser et al. (2014) and Mahajan & Baths (2021).
    """
    text = clean_text(text)
    doc = nlp(text)

    scores = []
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            if (pos_tags is None or token.pos_ in pos_tags) and lemma in lexicon:
                scores.append(lexicon[lemma])

    return sum(scores) / len(scores) if scores else None

def load_aoa_lexicon():
    """
    Load Age of Acquisition (AoA) ratings by Kuperman et al. (2012).
    """
    path = os.path.join(GIT_DIRECTORY, "resources/AoA_ratings_Kuperman_et_al_BRM.xlsx")
    df = pd.read_excel(path, engine="openpyxl") # load lexicon
    return dict(zip(df["Word"].str.lower(), df["Rating.Mean"])) # create dictionary for words and their average rating

def load_frequency_norms():
    """
    Load SUBTL word frequency norms by Brysbaert & New (2009).
    """
    path = os.path.join(GIT_DIRECTORY, "resources/SUBTLEXusfrequencyabove1.xls")
    df = pd.read_excel(path, engine="xlrd")
    return dict(zip(df["Word"].str.lower(), df["SUBTLWF"]))

def load_concreteness_lexicon():
    """
    Load concreteness ratings by Brysbaert et al. (2013).
    """
    path = os.path.join(GIT_DIRECTORY, "resources/Concreteness_Ratings_Brysbaert_et_al.xlsx")
    df = pd.read_excel(path, engine="openpyxl")
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))