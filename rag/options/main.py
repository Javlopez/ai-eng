# =============================================================================
# INSTALLATION AND USAGE INSTRUCTIONS
# =============================================================================
from rag.options.rag_tests import test_tfidf_only, test_sentence_bert_only, test_chromadb_only, test_openai_only


def print_installation_guide():
    """Print installation instructions"""

    print("""
INSTALLATION INSTRUCTIONS:
=========================

1. BASIC REQUIREMENTS:
   pip install scikit-learn pandas numpy

2. FOR SENTENCE-BERT:
   pip install sentence-transformers

3. FOR CHROMADB:
   pip install chromadb

4. FOR OPENAI:
   pip install openai
   export OPENAI_API_KEY='your-api-key-here'

5. INSTALL ALL:
   pip install scikit-learn pandas numpy sentence-transformers chromadb openai

USAGE:
======
python rag_implementations.py

APPROXIMATE COSTS (OpenAI):
==========================
- Embeddings: $0.0001 per 1K tokens
- GPT-3.5-turbo: $0.0015 per 1K input tokens
- For this demo: ~$0.05 total

PERFORMANCE COMPARISON:
======================
- TF-IDF: Fast, free, keyword-based matching
- Sentence-BERT: Semantic understanding, free, local
- ChromaDB: Production-ready vector database
- OpenAI: Best quality, paid service, internet required
""")

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
    elif choice == "2":
        test_sentence_bert_only()
    elif choice == "3":
        test_chromadb_only()
    elif choice == "4":
        test_openai_only()
    else:
        print("Invalid choice. Run individual functions as needed.")
        print("Available functions:")
        print("- test_tfidf_only()")
        print("- test_sentence_bert_only()")
        print("- test_chromadb_only()")
        print("- test_openai_only()")
        print("- run_comparative_demo()")