# =============================================================================
# INDIVIDUAL TESTING FUNCTIONS
# =============================================================================
from rag.options.bert import RAGSentenceBERT
from rag.options.chroma import RAGChromaDB
from rag.options.rag_openai import RAGOpenAI
from rag.options.tfidf import RAGTfIdf

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


def test_tfidf_only():
    """Test only TF-IDF implementation"""
    print("Testing TF-IDF RAG Implementation...")
    rag = RAGTfIdf(DOCS)

    queries = ["Apple revenue", "Tesla deliveries", "Microsoft cloud"]
    for query in queries:
        result = rag.query(query)
        print(f"Query: {query}")
        print(f"Top result: {result[0][:100]}...")
        print("-" * 50)


def test_sentence_bert_only():
    """Test only Sentence-BERT implementation"""
    print("Testing Sentence-BERT RAG Implementation...")
    try:
        rag = RAGSentenceBERT(DOCS)

        queries = ["company earnings", "vehicle production", "cloud services"]
        for query in queries:
            result = rag.query(query)
            print(f"Query: {query}")
            print(f"Top result: {result[0][:100]}...")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install sentence-transformers")


def test_chromadb_only():
    """Test only ChromaDB implementation"""
    print("Testing ChromaDB RAG Implementation...")
    try:
        rag = RAGChromaDB(DOCS)

        queries = ["financial performance", "electric vehicles", "layoffs"]
        for query in queries:
            result = rag.query(query)
            print(f"Query: {query}")
            print(f"Top result: {result[0][:100]}...")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install chromadb sentence-transformers")

def test_openai_only():
    """Test only OpenAI implementation"""
    print("Testing OpenAI RAG Implementation...")
    try:
        rag = RAGOpenAI(DOCS)

        queries = ["What are the latest Apple results?", "How is Tesla performing?"]
        for query in queries:
            result = rag.query(query)
            print(f"Query: {query}")
            print(f"Generated answer: {result}")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install OpenAI and set API key")