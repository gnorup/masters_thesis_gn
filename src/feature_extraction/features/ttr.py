# Type-Token Ratio -> lexical diversity; number of unique words / total number of words

import re # regex module to extract word tokens from text

def ttr(text):
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)] # finds all word-like sequences, bounded by word-boundaries; converts all to lowercase -> list of words
    return len(set(words)) / len(words) if words else None # set(words) keeps only unique words, len(words) total number of words -> TTR = types / tokens
