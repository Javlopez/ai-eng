import os

import numpy as np
from sklearn.neighbors import NearestNeighbors


class RAGOpenAI:
    def __init__(self, documents):
        """RAG using OpenAI embeddings and GPT"""
        print("\n🔧 Initializing RAG with OpenAI...")

        # Initialize attributes first to avoid AttributeError
        self.client = None
        self.documents = documents
        self.document_embeddings = None
        self.index = None

        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Error: Install OpenAI: pip install openai>=1.0.0")
            raise ImportError("OpenAI package not found")

        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: Set OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY='your-api-key'")
            print("   Or in Python: os.environ['OPENAI_API_KEY'] = 'your-key'")
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized")
        except Exception as e:
            print(f"❌ Error initializing OpenAI client: {e}")
            raise

        # Test API connection with a simple call
        try:
            print("🔍 Testing OpenAI connection...")
            test_response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input="test"
            )
            print("✅ OpenAI connection successful")
        except Exception as e:
            print(f"❌ Error connecting to OpenAI: {e}")
            print("   Check your API key and internet connection")
            raise

        self.client = OpenAI(api_key=api_key)
        self.documents = documents

        # Create embeddings with OpenAI
        print("📊 Creating embeddings with OpenAI...")
        self.document_embeddings = []

        for i, doc in enumerate(self.documents):
            print(f"   Processing document {i+1}/{len(documents)}")
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc
            )
            self.document_embeddings.append(response.data[0].embedding)


        self.document_embeddings = np.array(self.document_embeddings)
        print(f"✅ OpenAI embeddings created: {self.document_embeddings.shape}")

        # Create index
        self.index = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.index.fit(self.document_embeddings)

    def search_documents(self, query, top_k=3):
        """Search documents with OpenAI embeddings"""
        # Create query embedding
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )

        query_embedding = np.array([response.data[0].embedding])

        # Search similar documents
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=top_k)

        relevant_documents = []
        for idx, distance in zip(indices[0], distances[0]):
            relevant_documents.append({
                'document': self.documents[idx],
                'similarity': 1 - distance,
                'index': idx
            })

        return relevant_documents

    def generate_response(self, query, context):
        """Generate response using GPT with context"""
        prompt = f"""Based solely on the following context, answer the question clearly and concisely.

        CONTEXT:
        {chr(10).join([f"- {doc}" for doc in context])}

        QUESTION: {query}

        ANSWER:"""

        print("\n<UNK> Generating response from prompt...")
        print(prompt)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that answers questions based solely on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=200,
            temperature=0.1,
        )

        return response.choices[0].message.content

    def query(self, query):
        """Complete RAG with OpenAI"""
        print(f"\n🔍 OpenAI RAG Query: '{query}'")
        print("-" * 60)

        # 1. Search relevant documents
        results = self.search_documents(query)

        print("📋 Documents found:")
        context = []
        for i, result in enumerate(results):
            print(f"{i + 1}. [Similarity: {result['similarity']:.3f}]")
            print(f"   {result['document'][:100]}...")
            context.append(result['document'])

        # 2. Generate response with GPT
        response = self.generate_response(query, context)
        print(f"\n✅ FINAL ANSWER:")
        print(f"   {response}")

        return response
