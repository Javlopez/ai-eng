import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
import pandas as pd

model_name = "ViT-B-32"
embedding_function = OpenCLIPEmbeddingFunction(model_name=model_name)
data_loader = ImageLoader()

client = chromadb.Client()
collection_name = "multimodal_embeddings_collection"
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
    data_loader=data_loader
)

csv_file = '/usr/local/datasetsDir/images-and-descriptions/data/image_descriptions.csv'
df = pd.read_csv(csv_file)
images_folder = '/usr/local/datasetsDir/images-and-descriptions/data/images'

image_paths = []
image_Ids = []
descriptions = []
description_Ids=[]

for  _, row in df.iterrows():
    des_id = str(row[0])  # Ensure the ID is a string for matching
    description_Ids.append(des_id)
    description = str(row[1])
    # Find the image file corresponding to the image_id
    for file_name in os.listdir(images_folder):
        if file_name.startswith(f"{des_id}_") and file_name.endswith(".png"):
            image_path = os.path.join(images_folder, file_name)
            image_paths.append(image_path)
            image_Ids.append(des_id)
            descriptions.append(description)
            break

for img_id, img_path, desc_id, desc in zip(image_Ids, image_paths, description_Ids, descriptions):
    collection.add(
        ids=[img_id],
        uris=[img_path],
        metadatas=[{"image_uri": img_path, "description": desc}]
    )
    collection.add(
        ids=[desc_id],
        documents=[desc],
        metadatas=[{"image_uri": img_path, "description": desc}]

    )

# Query by text
query_text= "vitamic C fruits"
text_query_results = collection.query(
    query_texts=[query_text],
    n_results=5
)

# Query by image
query_images=[]
query_image_path = '/usr/local/datasetsDir/images-and-descriptions/queries/Test_girlwithorangesliceoneyes.jpg'
query_images.append(query_image_path)
image_query_results = collection.query(
    query_uris=query_images,
    n_results=5
)
print(text_query_results)
print(image_query_results)

client.delete_collection(name=collection_name)







