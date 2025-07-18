# Simulación para entender StrOutputParser

# Simulamos la respuesta completa del LLM
class MockAIMessage:
    def __init__(self, content):
        self.content = content
        self.additional_kwargs = {}
        self.response_metadata = {
            'token_usage': {'completion_tokens': 1, 'prompt_tokens': 15, 'total_tokens': 16},
            'model_name': 'llama-3.1-8b-instant',
            'finish_reason': 'stop'
        }
        self.id = 'run-12345-abcd'
        self.usage_metadata = {'input_tokens': 15, 'output_tokens': 1, 'total_tokens': 16}

    def __str__(self):
        return f"AIMessage(content='{self.content}', additional_kwargs={self.additional_kwargs}, response_metadata={self.response_metadata}, id='{self.id}', usage_metadata={self.usage_metadata})"


# Simulamos StrOutputParser
class MockStrOutputParser:
    def invoke(self, ai_message):
        return ai_message.content

    def __str__(self):
        return "StrOutputParser()"


print("🔍 QUÉ HACE StrOutputParser")
print("=" * 50)

# Simulamos la salida del LLM
llm_response = MockAIMessage("Negative")
parser = MockStrOutputParser()

print("📤 SALIDA DEL LLM (objeto completo):")
print("-" * 30)
print(f"Tipo: {type(llm_response)}")
print(f"Contenido completo:\n{llm_response}")
print(f"Solo el texto: '{llm_response.content}'")

print("\n🔧 APLICANDO StrOutputParser:")
print("-" * 30)
parsed_result = parser.invoke(llm_response)
print(f"Tipo resultado: {type(parsed_result)}")
print(f"Contenido: '{parsed_result}'")

print("\n📊 COMPARACIÓN:")
print("-" * 30)
print(f"Sin parser: {llm_response}")
print(f"Con parser: '{parsed_result}'")

print("\n🎯 EN TU CADENA:")
print("-" * 30)
print("sentiment_template | llm | StrOutputParser()")
print("                     ↓")
print("AIMessage(content='Negative', metadata=...) → 'Negative'")

print("\n💡 OTROS PARSERS DISPONIBLES:")
print("-" * 30)
print("• StrOutputParser() → Extrae solo el texto")
print("• JsonOutputParser() → Parsea JSON del texto")
print("• PydanticOutputParser() → Valida con esquemas Pydantic")
print("• OutputFixingParser() → Corrige errores automáticamente")

print("\n🚀 EJEMPLO PRÁCTICO:")
print("-" * 30)

# Simulamos diferentes tipos de respuestas del LLM
responses = [
    MockAIMessage("Positive"),
    MockAIMessage("The sentiment is clearly negative based on the feedback."),
    MockAIMessage('{"sentiment": "neutral", "confidence": 0.85}')
]

for i, response in enumerate(responses, 1):
    parsed = parser.invoke(response)
    print(f"Ejemplo {i}:")
    print(f"  LLM devuelve: {response.content}")
    print(f"  Parser extrae: '{parsed}'")
    print()

print("🔗 POR QUÉ ES NECESARIO:")
print("-" * 30)
print("• El LLM siempre devuelve objetos con metadata")
print("• Tu aplicación normalmente solo necesita el texto")
print("• StrOutputParser hace la extracción automáticamente")
print("• Es el último paso típico en muchas cadenas")