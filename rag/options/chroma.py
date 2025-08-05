# =============================================================================
# IMPLEMENTATION 2: RAG with ChromaDB (REAL VECTOR DATABASE)
# =============================================================================

class RAGChromaDB:
    def __init__(self, documents):
        """RAG using ChromaDB as vector database"""
        print("\n🔧 Initializing RAG with ChromaDB...")
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("❌ Error: Install dependencies:")
            print("   pip install chromadb sentence-transformers")
            return

        # Initialize embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # create chroma client
        self.client = chromadb.Client()

        # Create collection (delete if exists)
        try:
            self.client.delete_collection("documents")
        except:
            pass

        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space":"cosine"}
        )

        # Insert documents
        print("📊 Inserting documents into ChromaDB...")
        embeddings = self.model.encode(documents).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

        print(f"✅ {len(documents)} documents inserted into ChromaDB")

    def query(self, query, top_k=3):
        """Query using ChromaDB"""
        print(f"\n🔍 ChromaDB Query: '{query}'")
        print("-" * 60)

        # Create query embedding
        query_embeddings = self.model.encode([query]).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )

        print("📋 Documents found by ChromaDB:")
        context = []

        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance
            print(f"{i + 1}. [Similarity: {similarity:.3f}]")
            print(f"   {doc}")
            print()
            context.append(doc)

        return context
