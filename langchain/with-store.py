from langchain_core.documents import Document
from langchain_core.vectorstores import  InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

documents = [document_1, document_2, document_3]
uuids = [str(uuid4()) for _ in range(len(documents))]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(documents=documents, ids=uuids)

query = "What's the weather going to be like tomorrow?"
results = vector_store.similarity_search(query, k=1)  # e.g., top 3 matches
print(results)