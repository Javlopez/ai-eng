# =============================================================================
# IMPLEMENTATION 4: YOUR ORIGINAL CODE IMPROVED
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class RAGTfIdf:
    def __init__(self, documents):
        """Your original TF-IDF implementation"""
        print("\n🔧 Initializing RAG with TF-IDF...")
        self.documents = documents
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            lowercase=True,
            sublinear_tf=True,
        )

        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print(f"✅ TF-IDF matrix created: {self.tfidf_matrix.shape}")

        # Create index
        self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.index.fit(self.tfidf_matrix)
        print("✅ TF-IDF index created")

    def query(self, query):
        """Query with TF-IDF """
        print(f"\n🔍 TF-IDF Query: '{query}'")
        print("-" * 60)

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Search similar documents
        distances, indices = self.index.kneighbors(query_vec)

        print("📋 Documents found:")
        context = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            similarity = 1 - dist
            print(f"{i + 1}. [Similarity: {similarity:.3f}]")
            print(f"   {self.documents[idx]}")
            print()
            context.append(self.documents[idx])

        return context
