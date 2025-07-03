def simple_stemmer(word):
    suffixes = ["ing", "ly", "ed", "ious", "ies", "ive", "es", "s", "ment"]
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)] # Remove the matched suffix.
    return word

# # Example usage
# words = ["running", "happily", "tried", "faster", "cats"]
# stemmed_words = [simple_stemmer(word) for word in words]
# print(stemmed_words)