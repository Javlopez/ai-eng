import numpy as np

from nlp.cbow import context_indices


def softmax(x):
    """
    Compute the softmax of vector x.
    We subtract the maximum value for numerical stability.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# -------------------------
# Step 1: Define the Corpus
# -------------------------
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying"
]

print("Original Corpus:")
for sentence in corpus:
    print(" -", sentence)

# ------------------------------------
# Step 2: Preprocess the Corpus
# Lowercase and tokenize each sentence.
# ------------------------------------
sentences = [sentence.lower().split() for sentence in corpus]
print("\nTokenized Sentences:")
for sentence in sentences:
    print(" -", sentence)

# -----------------------------------------
# Step 3: Build the Vocabulary and Mappings
# -----------------------------------------
vocab = list(set(word for sentence in sentences for word in sentence))

# Create dictionaries to map words to indices and vice-versa.
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

print("\nVocabulary (word to index mapping):")
for word, idx in word2idx.items():
    print(f" {word}: {idx}")


# -------------------------------------------------------
# Step 4: Generate Training Data (Skip-gram Pairs)
# -------------------------------------------------------
# For each word in a sentence, use a window of size 1 to collect context words.
window_size = 1
training_pairs = []  # will store tuples of (center_word_idx, context_word_idx)

for sentence in sentences:
    for idx, word in enumerate(sentence):
        center_word_idx = word2idx[word]
        # Determine the indices for the context window
        context_indices = list(range(max(0, idx - window_size), idx)) + \
                          list(range(idx + 1, min(len(sentence), idx + window_size + 1)))

        for context_idx in context_indices:
            context_word_idx = word2idx[sentence[context_idx]]
            training_pairs.append((center_word_idx, context_word_idx))


print("\nTraining Pairs (center word index, context word index):")
for center, context in training_pairs:
    print(f" Center: {idx2word[center]} ({center}), Context: {idx2word[context]} ({context})")


