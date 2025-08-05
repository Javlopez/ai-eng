# =============================================================================
# IMPLEMENTATION 1: RAG with Sentence-BERT (LOCAL - FREE)
# =============================================================================
from sklearn.neighbors import NearestNeighbors


class RAGSentenceBERT:

    def __init__(self, documents):
        """RAG using Sentence-BERT for embeddings"""
        print("\n🔧 Initializing RAG with Sentence-BERT...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-BERT model loaded")
        except ImportError:
            print("❌ Error: Install sentence-transformers: pip install sentence-transformers")
            return

        self.documents = documents

        # Create REAL embeddings
        print("📊 Creating embeddings for documents...")
        self.document_embeddings = self.model.encode(self.documents)
        print(f"✅ Embeddings created: {self.document_embeddings.shape}")

        # Create search index
        self.index  = NearestNeighbors(n_neighbors=3, metric='cosine').fit(self.document_embeddings)
        print("✅ Search index created")

    def search_documents(self, query, top_k=3):
        """Find documents most similar to query"""
        # Convert query to embedding
        query_embeddings = self.model.encode([query])

        # Search for similar documents
        distance, indices = self.index.kneighbors(query_embeddings, n_neighbors=top_k)
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distance[0])):
            similarity = 1 - dist # Convert cosine distance to similarity
            results.append({
                'document': self.documents[idx],
                'similarity': similarity,
                'index': idx,
            })

        return results

    def query(self, query):
        """Complete query: search documents and return organized results"""
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        # Search relevant documents
        results = self.search_documents(query)

        print("📋 Most relevant documents:")
        context = []
        for i, result in enumerate(results):
            print(f"{i + 1}. [Similarity: {result['similarity']:.3f}]")
            print(f"   {result['document']}")
            print()
            context.append(result['document'])

        return context
