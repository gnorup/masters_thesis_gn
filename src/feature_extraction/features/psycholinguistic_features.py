# source: Fraser et al. (2014)
# Age of Acquisition (AoA), Familiarity, Imageability, Frequency

# Age of Acquisition (AoA; average age when people typically learn a word): ratings of 30'121 English words by Kuperman et al. (2012) (feature idea: Fraser et al. (2014, 2016); Forbes-McKay (2004))
# Familiarity: combined Bristol/Gilhooly & Logie norms by Stadthagen-Gonzalez & Davis (2006)
# Imageability (how easy a word elicits mental images): combined Bristol/Gilhooly & Logie norms by Stadthagen-Gonzalez & Davis (2006)
# Frequency: SUBTL word frequency norms of 60'400 English words by Brysbaert & New, 2009
# Concreteness: based on human-rated norms of abstract (1) to concrete (5) words by Brysbaert et al., 2013


import os
import pandas as pd
import spacy

from config.constants import GIT_DIRECTORY
from feature_extraction.features import clean_text

nlp = spacy.load("en_core_web_sm")

def compute_avg_by_pos(text, lexicon, pos_tags=None):
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
    path = os.path.join(GIT_DIRECTORY, "resources/AoA_ratings_Kuperman_et_al_BRM.xlsx")
    df = pd.read_excel(path) # load AoA lexicon
    return dict(zip(df["Word"].str.lower(), df["Rating.Mean"])) # creates dictionary for words and their average AoA rating

def load_imageability_norms():
    path = os.path.join(GIT_DIRECTORY, "resources/StadthagenDavis2006BristolNorms.txt")
    df = pd.read_csv(path, delimiter="\t")
    return dict(zip(df["WORD"].str.lower(), df["IMG"]))

def load_familiarity_norms():
    path = os.path.join(GIT_DIRECTORY, "resources/StadthagenDavis2006BristolNorms.txt")
    df = pd.read_csv(path, delimiter="\t")
    return dict(zip(df["WORD"].str.lower(), df["FAM"]))

def load_frequency_norms():
    path = os.path.join(GIT_DIRECTORY, "resources/SUBTLEXusfrequencyabove1.xls")
    df = pd.read_excel(path)
    return dict(zip(df["Word"].str.lower(), df["SUBTLWF"]))

def load_concreteness_lexicon():
    path = os.path.join(GIT_DIRECTORY, "resources/Concreteness_Ratings_Brysbaert_et_al.xlsx")
    df = pd.read_excel(path)
    return dict(zip(df["Word"].str.lower(), df["Conc.M"]))
