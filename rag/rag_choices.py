import numpy as np

# Sample document knowledge base
DOCS = [
    "Apple reported record revenue of $94.8 billion in Q1 2024, beating Wall Street expectations by $3 billion.",
    "Tesla delivered 484,507 electric vehicles in Q4 2023, setting a new quarterly record for the company.",
    "Microsoft Azure experienced 30% year-over-year growth in the latest quarter, driven by cloud services demand.",
    "Google announced layoffs of 12,000 employees in January 2023 as part of cost-reduction restructuring.",
    "Amazon Prime reached 200 million subscribers worldwide, solidifying its e-commerce dominance.",
    "Apple launched Vision Pro at $3,499 in February 2024, marking its entry into the mixed reality market.",
    "Tesla reduced Model Y prices by 20% to stimulate demand in competitive markets like China.",
    "Microsoft invested an additional $10 billion in OpenAI in January 2023, strengthening its AI position.",
    "Google Bard was launched to compete directly with OpenAI's ChatGPT in the AI chatbot market.",
    "Amazon Web Services maintains cloud computing leadership with a 32% market share."
]

print("="*80)
print("REAL RAG IMPLEMENTATIONS")
print("="*80)
# =============================================================================
# IMPLEMENTATION 1: RAG with Sentence-BERT (LOCAL - FREE)
# =============================================================================
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




def test_tfidf_only():
    """Test only TF-IDF implementation"""
    print("Testing TF-IDF RAG Implementation...")
    rag = RAGTfIdfImproved(DOCUMENTS)

    queries = ["Apple revenue", "Tesla deliveries", "Microsoft cloud"]
    for query in queries:
        result = rag.query(query)
        print(f"Query: {query}")
        print(f"Top result: {result[0][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    print_installation_guide()

    print("\nChoose test option:")
    print("1. Full comparative demo")
    print("2. Test TF-IDF only")
    print("3. Test Sentence-BERT only")
    print("4. Test ChromaDB only")
    print("5. Test OpenAI only")
    print("\nEnter choice (1-5): ", end="")

    choice = input().strip()

    if choice == "1":
        test_tfidf_only()
        # run_comparative_demo()
    # elif choice == "2":
        # test_tfidf_only()
    # elif choice == "3":
    #     test_sentence_bert_only()
    # elif choice == "4":
    #     test_chromadb_only()
    # elif choice == "5":
        # test_openai_only()
    else:
        print("Invalid choice. Run individual functions as needed.")
        print("Available functions:")
        print("- test_tfidf_only()")
        print("- test_sentence_bert_only()")
        print("- test_chromadb_only()")
        print("- test_openai_only()")
        print("- run_comparative_demo()")