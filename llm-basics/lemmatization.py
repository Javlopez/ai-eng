def simple_lemmatize(word):
    # A minimal dictionary for known irregular forms.
    irregular_lemmas = {
        "running": "run",
        "happily": "happy",
        "ran": "run",
        "better": "good",
        "faster": "fast",
        "cats": "cat",
        "dogs": "dog",
        "are": "be",
        "is": "be",
        "have": "have",
        "dancing": "dance"
    }
    return irregular_lemmas.get(word, word)

# words = ["running", "happily", "ran", "better", "faster", "cats"]
# lemmatized_words = [simple_lemmatize(word) for word in words]
# print(lemmatized_words)