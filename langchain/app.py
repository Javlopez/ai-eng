from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter  import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

with open("the_brain.txt") as f:
    the_brain = f.read()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=600,
#     chunk_overlap=0,
#     length_function=len,
# )
#
# naive_chunks = text_splitter.split_text(the_brain)
#
# for chunk in naive_chunks[40:55]:
#     print(chunk + "\n")

local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # o 'cuda' si tienes GPU
)

# Esto será 10x más rápido
semantic_chunker = SemanticChunker(
    local_embeddings,
    breakpoint_threshold_type="percentile",
)
# semantic_chunker = SemanticChunker(
#     OpenAIEmbeddings(
#         model='text-embedding-3-large',
#     ),
#     breakpoint_threshold_type="percentile",
# )
#
semantic_chunks = semantic_chunker.create_documents([the_brain])

for semantic_chunk in semantic_chunks:
    if "MDT is associated with the basic" in semantic_chunk.page_content:
        print(semantic_chunk.page_content)
        print(len(semantic_chunk.page_content))