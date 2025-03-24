import re  # regular expressions module, clean and manipulate text

def n_words(text): # define function to count words, takes one parameter
    text = text.strip()  # remove leading/trailing spaces
    text = re.sub(r'^"|"$', '', text)  # remove leading/trailing quotation marks (^" for start, "$ for end)
    # text = re.sub(r'\b(um|uh)\b', '', text)  # remove filler words 'um' and 'uh' (\b for word boundary)
    text = re.sub(r'[–—]', ' ', text)  # replace en/em dashes with space
    text = re.sub(r'[^\w\s\'-]', '', text)  # remove punctuation, keep apostrophes and hyphens (and word characters \w, whitespace characters \s)
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with single space
    words = text.split()  # split text into list of words by whitespace
    return len(words)  # return the number of words in the list (the length of the list)
