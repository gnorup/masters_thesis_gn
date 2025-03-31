# feature idea: Lindsay et al. (2021)

import re  # regular expressions module, clean and manipulate text

def clean_text(text):
    if not isinstance(text, str):  # handle NaN or unexpected types
        return ""
    text = text.strip()  # remove leading/trailing spaces
    text = re.sub(r'^"|"$', '', text)  # remove leading/trailing quotation marks (^" for start, "$ for end)
    text = re.sub(r'[–—]', ' ', text)  # replace en/em dashes with space
    text = re.sub(r'[^\w\s\'-]', '', text)  # remove punctuation, keep apostrophes and hyphens (and word characters \w, whitespace characters \s)
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with single space
    return text.lower()


def n_words(text, return_words=False):
    cleaned = clean_text(text)
    words = cleaned.split()  # split text into list of words by whitespace
    return words if return_words else len(words)  # return the number of words in the list (the length of the list)

# note: contractions are retained to preserve the natural structure of spontaneous speech