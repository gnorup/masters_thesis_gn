import re

def clean_text(text):
    """Clean transcriptions before feature calculation."""
    if not isinstance(text, str):  # handle NaN or unexpected types
        return ""
    text = text.strip()  # remove leading/trailing spaces
    text = re.sub(r'^"|"$', '', text)  # remove leading/trailing quotation marks (^" for start, "$ for end)
    text = re.sub(r'[–—]', ' ', text)  # replace en/em dashes with space
    text = re.sub(r'[^\w\s\'-]', '', text)  # remove punctuation, keep apostrophes and hyphens (and word characters \w, whitespace characters \s)
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with single space
    return text.lower()

def tokenize(text):
    """Tokenize words in transcript."""
    cleaned = clean_text(text)
    return cleaned.split() # split text into list of words by whitespace

def n_words(text):
    """
    Calculate the number of words in the transcript.
    Feature definition based on Lindsay et al. (2021) and Hamrick et al. (2023).
    """
    words = tokenize(text)
    return len(words)

# Note: contractions are retained to preserve the natural structure of spontaneous speech