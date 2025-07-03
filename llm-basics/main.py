from stemming import simple_stemmer
from tokenizer import simple_tokenizer
from lemmatization import simple_lemmatize

def process(text):
    tokens = simple_tokenizer(text)
    print("Tokens:")
    print(tokens)
    stemmed_words = [simple_stemmer(token) for token in tokens]
    print("Stemmed words:")
    print(stemmed_words)
    lemmatized_words = [simple_lemmatize(word) for word in stemmed_words]
    print("Lemmatized words:")
    print(lemmatized_words)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text = "This is a text while I'm dancing"
    process(text)

