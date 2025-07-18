import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Inicializar ChromaDB
client = chromadb.Client()

# Crear colección con configuración HNSW
collection = client.create_collection(
    name="peliculas_hnsw",
    metadata={
        "hnsw:space": "cosine",        # Métrica de distancia
        # "hnsw:M": 16,                  # Conexiones por nodo en cada capa
        # "hnsw:ef_construction": 200,   # Tamaño de búsqueda durante construcción
        # "hnsw:ef": 100,         # Tamaño de búsqueda durante query
        # "hnsw:max_elements": 10000     # Máximo elementos
    }
)

# Modelo para generar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Datos de ejemplo - películas
peliculas = [
    "Avengers superhéroes Marvel acción",
    "Iron Man superhéroe Marvel Tony Stark",
    "Thor superhéroe Marvel Asgard",
    "Spider-Man superhéroe Marvel Peter Parker",
    "Matrix ciencia ficción Neo realidad virtual",
    "Terminator robots futuro Arnold Schwarzenegger",
    "Blade Runner androides futuro Philip K Dick",
    "Titanic romance drama Leonardo DiCaprio",
    "Casablanca romance clásico Humphrey Bogart",
    "El Padrino crimen familia mafia",
    "Scarface crimen drogas Al Pacino",
    "Goodfellas mafia crimen Martin Scorsese"
]

# Generar embeddings
embeddings = model.encode(peliculas)
print(f"Dimensión de embeddings: {embeddings.shape}")

# Agregar a ChromaDB
ids = [f"pelicula_{i}" for i in range(len(peliculas))]
collection.add(
    embeddings=embeddings.tolist(),
    documents=peliculas,
    ids=ids,
    metadatas=[{"genero": "accion" if "superhér" in p or "Matrix" in p or "Terminator" in p
                else "romance" if "romance" in p or "Titanic" in p
                else "crimen"} for p in peliculas]
)

print("=== BASE DE DATOS CONSTRUIDA ===")
print(f"Total películas: {len(peliculas)}")

# Simular cómo HNSW organiza en capas conceptualmente
print("\n=== SIMULACIÓN DE CAPAS HNSW ===")
print("CAPA 2 (Superior - Conexiones largas):")
print("  Matrix ←→ Avengers ←→ Titanic ←→ El Padrino")
print("  (Representantes de cada género)")

print("\nCAPA 1 (Intermedia - Conexiones medias):")
print("  Matrix ←→ Terminator ←→ Blade Runner")
print("  Avengers ←→ Iron Man ←→ Thor ←→ Spider-Man")
print("  Titanic ←→ Casablanca")
print("  El Padrino ←→ Scarface ←→ Goodfellas")

print("\nCAPA 0 (Base - Todas las conexiones precisas):")
print("  Todas las películas con conexiones detalladas")

# Ejemplo de búsqueda
print("\n=== BÚSQUEDA SIMILAR A 'AVENGERS' ===")
query = "superhéroes Marvel equipo"
query_embedding = model.encode([query])

# Buscar similares
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5,
    include=['documents', 'distances', 'metadatas']
)

print(f"Query: '{query}'")
print("\nResultados (simulando navegación por capas):")
for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
)):
    print(f"{i+1}. {doc}")
    print(f"   Distancia: {distance:.4f}")
    print(f"   Género: {metadata['genero']}")
    print(f"   {'→ Encontrado en capa 2' if 'Avengers' in doc else '→ Encontrado en capa 1' if any(x in doc for x in ['Iron Man', 'Thor', 'Spider-Man']) else '→ Encontrado en capa 0'}")
    print()

# Ejemplo con filtros (aprovechando metadatos)
print("=== BÚSQUEDA FILTRADA POR GÉNERO ===")
results_filtered = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
    where={"genero": "accion"},
    include=['documents', 'distances', 'metadatas']
)

print("Solo películas de acción:")
for doc, distance in zip(results_filtered['documents'][0], results_filtered['distances'][0]):
    print(f"- {doc} (distancia: {distance:.4f})")

# Demostrar eficiencia
print("\n=== EFICIENCIA HNSW ===")
print("Sin HNSW (búsqueda lineal):")
print(f"  - Comparaciones necesarias: {len(peliculas)} (todas las películas)")
print(f"  - Complejidad: O(n)")

print("\nCon HNSW (multicapa):")
print(f"  - Comparaciones estimadas: ~{int(np.log2(len(peliculas)) * 3)} (navegación por capas)")
print(f"  - Complejidad: O(log n)")
print(f"  - Mejora: {len(peliculas) / max(1, int(np.log2(len(peliculas)) * 3)):.1f}x más rápido")

# Configuración avanzada
print("\n=== PARÁMETROS HNSW EXPLICADOS ===")
print("hnsw:M = 16:")
print("  - Cada nodo se conecta a máximo 16 otros nodos")
print("  - Más conexiones = mejor recall, pero más memoria")

print("\nhnsw:ef_construction = 200:")
print("  - Durante construcción, explora 200 candidatos")
print("  - Valor alto = índice más preciso pero construcción más lenta")

print("\nhnsw:ef_search = 100:")
print("  - Durante búsqueda, explora 100 candidatos")
print("  - Valor alto = mejores resultados pero búsqueda más lenta")

# Ejemplo de cómo se vería la navegación paso a paso
print("\n=== SIMULACIÓN DE NAVEGACIÓN POR CAPAS ===")
print("Buscando similares a 'Avengers':")
print("1. CAPA 2: Empieza en Matrix → salta a Avengers (más similar)")
print("2. CAPA 1: Desde Avengers → explora Iron Man, Thor")
print("3. CAPA 0: Desde Iron Man/Thor → encuentra Spider-Man")
print("4. RESULTADO: [Iron Man, Thor, Spider-Man, Avengers]")

print("\n=== VENTAJAS DE CHROMADB CON HNSW ===")
print("✓ Búsqueda ultrarrápida en millones de vectores")
print("✓ Construcción automática del índice multicapa")
print("✓ Persistencia automática")
print("✓ Filtros por metadatos")
print("✓ Diferentes métricas de distancia (cosine, euclidean, etc.)")
print("✓ Configuración ajustable según necesidades")