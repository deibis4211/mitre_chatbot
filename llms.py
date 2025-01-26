import json
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Configurar la clave de API de OpenAI
os.environ['OPENAI_API_KEY'] = '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'

# Leer el fichero JSON
with open("techniques_enterprise_attack.json", "r") as f:
    techniques = json.load(f)

# Crear embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Crear documentos con contenido y metadatos
documents = []
for technique in techniques:
    page_content = technique.get("description", "")
    mitigation_methods = "\n".join(
        [f"{m['name']}: {m['description']}" for m in technique.get("mitigation_methods", [])]
    )
    metadata = {
        "id": technique.get("id", ""),
        "name": technique.get("name", ""),
        "detection": technique.get("detection", ""),
        "mitigation_methods": mitigation_methods
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

# Crear el vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Definir el prompt específico
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
Eres un asistente experto en ciberseguridad utilizando la base de datos MITRE ATT&CK.

Historial de la conversación:
{history}

Pregunta del usuario:
{input}

Respuesta:
"""
)

# Crear el modelo de chat
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Crear la memoria de la conversación
memory = ConversationBufferMemory(memory_key="history")

# Crear la cadena de conversación
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    prompt=prompt_template
)

# Bucle principal para leer la entrada del usuario y generar respuestas
while True:
    user_input = input("Usuario: ")
    if user_input.lower() in [":salir", ":exit", ":terminar"]:
        break

    # Generar la respuesta del modelo
    ai_response = conversation.predict(input=user_input)

    # Mostrar la respuesta del modelo
    print(f"\nChatBot: {ai_response}\n")