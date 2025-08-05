# RAG EN PYTHON - EJEMPLOS COMPLETOS
# ===================================

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Simulamos una base de documentos empresarial
DOCUMENTOS = [
    "Apple reportó ingresos de $94.8B en Q1 2024, superando expectativas",
    "Tesla entregó 484,507 vehículos en Q4 2023, un récord trimestral",
    "Microsoft Azure creció 30% año tras año en el último trimestre",
    "Google anunció despidos de 12,000 empleados en enero 2023",
    "Amazon Prime tiene ahora 200 millones de suscriptores globalmente",
    "Apple lanzó Vision Pro por $3,499 en febrero 2024",
    "Tesla redujo precios del Model Y en 20% para aumentar demanda",
    "Microsoft invirtió $10B adicionales en OpenAI en enero 2023",
    "Google Bard compite directamente con ChatGPT de OpenAI",
    "Amazon Web Services sigue siendo el líder en cloud computing"
]

print("=" * 80)
print("COMPARACIÓN: LLM Solo vs Vector DB Solo vs RAG Completo")
print("=" * 80)


# =============================================================================
# MÉTODO 1: SOLO LLM (Sin acceso a documentos)
# =============================================================================
def solo_llm(pregunta):
    """Simula un LLM sin acceso a información actualizada"""
    print(f"\n🤖 SOLO LLM:")
    print(f"Pregunta: {pregunta}")
    print("Respuesta: Lo siento, no tengo información actualizada sobre")
    print("           los resultados financieros más recientes de estas empresas.")
    print("           Mi conocimiento está limitado a mi fecha de entrenamiento.")


# =============================================================================
# MÉTODO 2: SOLO VECTOR DB (Sin procesamiento inteligente)
# =============================================================================
def solo_vector_db(pregunta):
    """Vector DB que solo devuelve documentos crudos"""
    print(f"\n📊 SOLO VECTOR DB:")
    print(f"Pregunta: {pregunta}")

    # Simulación simple de búsqueda por palabras clave
    palabras = pregunta.lower().split()
    documentos_encontrados = []

    for i, doc in enumerate(DOCUMENTOS):
        score = 0
        for palabra in palabras:
            if palabra in doc.lower():
                score += 1
        if score > 0:
            documentos_encontrados.append((score, doc))

    # Ordenar por relevancia
    documentos_encontrados.sort(reverse=True, key=lambda x: x[0])

    print("Documentos encontrados (texto crudo):")
    for i, (score, doc) in enumerate(documentos_encontrados[:3]):
        print(f"  {i + 1}. [Score: {score}] {doc}")

    print("\n❌ PROBLEMA: Solo texto crudo, sin respuesta procesada")


# =============================================================================
# MÉTODO 3: RAG CON SENTENCE-BERT (Vector DB + LLM)
# =============================================================================
class RAGConSentenceBERT:
    def __init__(self, documentos):
        print("\n🔧 Inicializando RAG con Sentence-BERT...")

        # Simulamos embeddings de Sentence-BERT (en realidad serían de 384 dimensiones)
        self.documentos = documentos
        self.embeddings = self._crear_embeddings_simulados()

        # Crear índice de búsqueda
        self.index = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.index.fit(self.embeddings)

        print(f"✅ Vector DB creada con {len(documentos)} documentos")

    def _crear_embeddings_simulados(self):
        """Simula embeddings de Sentence-BERT"""
        # En realidad usarías:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return model.encode(self.documentos)

        np.random.seed(42)  # Para resultados reproducibles
        return np.random.random((len(self.documentos), 10))  # 10D en lugar de 384D

    def _buscar_documentos(self, pregunta):
        """PASO 1: Vector DB busca documentos relevantes"""
        print(f"\n🔍 PASO 1 - Vector DB busca documentos...")

        # Simular embedding de la pregunta
        np.random.seed(hash(pregunta) % 1000)
        pregunta_embedding = np.random.random((1, 10))

        # Buscar documentos similares
        distancias, indices = self.index.kneighbors(pregunta_embedding)

        documentos_relevantes = []
        print("   Documentos encontrados:")
        for i, (idx, dist) in enumerate(zip(indices[0], distancias[0])):
            doc = self.documentos[idx]
            documentos_relevantes.append(doc)
            print(f"   {i + 1}. [Similitud: {1 - dist:.3f}] {doc[:60]}...")

        return documentos_relevantes

    def _generar_respuesta(self, pregunta, contexto):
        """PASO 2: LLM procesa contexto y genera respuesta"""
        print(f"\n🤖 PASO 2 - LLM procesa contexto...")

        # Simular el prompt que se enviaría al LLM
        prompt = f"""
CONTEXTO:
{chr(10).join([f"- {doc}" for doc in contexto])}

PREGUNTA: {pregunta}

INSTRUCCIÓN: Responde basándote únicamente en el contexto proporcionado.
"""

        print("   Prompt enviado al LLM:")
        print("   " + "=" * 50)
        print("   " + prompt.replace("\n", "\n   "))
        print("   " + "=" * 50)

        # Simular respuesta del LLM (en realidad sería llamada a OpenAI/etc)
        respuesta = self._simular_respuesta_llm(pregunta, contexto)

        return respuesta

    def _simular_respuesta_llm(self, pregunta, contexto):
        """Simula lo que respondería un LLM real"""
        # Análisis simple para simular respuesta inteligente
        empresas_mencionadas = []
        for doc in contexto:
            for empresa in ["Apple", "Tesla", "Microsoft", "Google", "Amazon"]:
                if empresa in doc:
                    empresas_mencionadas.append(empresa)

        if "resultados" in pregunta.lower() or "ingresos" in pregunta.lower():
            return f"Según la información más reciente, {empresas_mencionadas[0] if empresas_mencionadas else 'las empresas'} han reportado resultados sólidos. Específicamente, basándome en los documentos: {contexto[0]}"

        elif "precios" in pregunta.lower():
            return f"En cuanto a precios, encontré información relevante: {contexto[0]}"

        else:
            return f"Basándome en los documentos disponibles, puedo confirmar que {contexto[0]}"

    def consultar(self, pregunta):
        """RAG completo: Retrieval + Augmented Generation"""
        print(f"\n🚀 RAG COMPLETO:")
        print(f"Pregunta: {pregunta}")

        # PASO 1: Buscar documentos relevantes
        documentos_relevantes = self._buscar_documentos(pregunta)

        # PASO 2: Generar respuesta con LLM
        respuesta = self._generar_respuesta(pregunta, documentos_relevantes)

        print(f"\n✅ RESPUESTA FINAL:")
        print(f"   {respuesta}")

        return respuesta


# =============================================================================
# MÉTODO 4: RAG CON OPENAI (Más realista)
# =============================================================================
class RAGConOpenAI:
    def __init__(self, documentos):
        """RAG usando OpenAI embeddings y GPT"""
        print("\n🔧 Inicializando RAG con OpenAI...")

        self.documentos = documentos
        # En un caso real:
        # import openai
        # self.client = openai.OpenAI(api_key="tu-key")
        # self.embeddings = self._crear_embeddings_openai()

        # Para demo, usamos embeddings simulados
        self.embeddings = self._crear_embeddings_simulados()
        self.index = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.index.fit(self.embeddings)

        print(f"✅ Vector DB con OpenAI embeddings creada")

    def _crear_embeddings_simulados(self):
        """En realidad sería llamada a OpenAI API"""
        # Código real sería:
        # embeddings = []
        # for doc in self.documentos:
        #     response = self.client.embeddings.create(
        #         model="text-embedding-ada-002",
        #         input=doc
        #     )
        #     embeddings.append(response.data[0].embedding)
        # return np.array(embeddings)

        np.random.seed(42)
        return np.random.random((len(self.documentos), 1536))  # OpenAI ada-002 = 1536D

    def consultar(self, pregunta):
        """RAG con OpenAI real"""
        print(f"\n🚀 RAG CON OPENAI:")
        print(f"Pregunta: {pregunta}")

        # 1. Buscar documentos
        np.random.seed(hash(pregunta) % 1000)
        pregunta_embedding = np.random.random((1, 1536))
        distancias, indices = self.index.kneighbors(pregunta_embedding)

        contexto = [self.documentos[idx] for idx in indices[0]]

        print("\n📋 Documentos recuperados:")
        for i, doc in enumerate(contexto):
            print(f"   {i + 1}. {doc}")

        # 2. Llamada a GPT (simulada)
        prompt = f"""Basándote en el siguiente contexto, responde a la pregunta:

CONTEXTO:
{chr(10).join(contexto)}

PREGUNTA: {pregunta}

RESPUESTA:"""

        print(f"\n🤖 Llamada a GPT-3.5-turbo:")
        print("   (Simulada - en realidad sería llamada a OpenAI API)")

        # Respuesta simulada
        respuesta = f"Según los datos más recientes, {contexto[0][:50]}... Esta información indica tendencias positivas en el sector."

        print(f"\n✅ RESPUESTA:")
        print(f"   {respuesta}")

        return respuesta


# =============================================================================
# DEMO COMPARATIVA
# =============================================================================
def demo_comparativa():
    pregunta = "¿Cuáles fueron los últimos resultados financieros de Apple?"

    print("PREGUNTA DE PRUEBA:")
    print(f"'{pregunta}'")
    print("\n" + "=" * 80)

    # 1. Solo LLM
    solo_llm(pregunta)

    print("\n" + "-" * 80)

    # 2. Solo Vector DB
    solo_vector_db(pregunta)

    print("\n" + "-" * 80)

    # 3. RAG con Sentence-BERT
    rag_bert = RAGConSentenceBERT(DOCUMENTOS)
    rag_bert.consultar(pregunta)

    print("\n" + "-" * 80)

    # 4. RAG con OpenAI
    rag_openai = RAGConOpenAI(DOCUMENTOS)
    rag_openai.consultar(pregunta)


# =============================================================================
# CÓDIGO REAL CON APIS (Para referencia)
# =============================================================================
def ejemplo_codigo_real():
    """Cómo se vería con APIs reales"""

    codigo_real = '''
# IMPLEMENTACIÓN REAL CON APIS
# ============================

from sentence_transformers import SentenceTransformer
import openai
import chromadb

# Opción 1: Con Sentence-BERT + OpenAI
def rag_real_sentencebert():
    # 1. Crear embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documentos)

    # 2. Vector DB
    client = chromadb.Client()
    collection = client.create_collection("docs")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documentos,
        ids=[f"doc_{i}" for i in range(len(documentos))]
    )

    # 3. Query
    query_embedding = model.encode([pregunta])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    # 4. LLM
    contexto = "\\n".join(results['documents'][0])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Contexto: {contexto}\\n\\nPregunta: {pregunta}"
        }]
    )

    return response.choices[0].message.content

# Opción 2: Todo OpenAI
def rag_real_openai():
    # 1. Embeddings
    embeddings = []
    for doc in documentos:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=doc
        )
        embeddings.append(response['data'][0]['embedding'])

    # 2. Buscar similares (con numpy o vector DB)
    # ... código de búsqueda ...

    # 3. Generar respuesta
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
'''

    print("\n" + "=" * 80)
    print("CÓDIGO REAL CON APIS:")
    print("=" * 80)
    print(codigo_real)


# Ejecutar demo
if __name__ == "__main__":
    demo_comparativa()
    ejemplo_codigo_real()

    print("\n" + "=" * 80)
    print("RESUMEN:")
    print("=" * 80)
    print("1. ❌ SOLO LLM: No tiene info actualizada")
    print("2. ❌ SOLO VECTOR DB: Solo devuelve docs crudos")
    print("3. ✅ RAG: Vector DB encuentra + LLM procesa = Respuesta inteligente")
    print("4. 🚀 APIs REALES: Sentence-BERT/OpenAI + GPT para producción")