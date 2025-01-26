import json
import os
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Configurar la clave de API de OpenAI
load_dotenv()
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

# Crear el vector store usando Chroma
vectorstore = Chroma(
    collection_name="mitre_techniques",
    embedding_function=embeddings
)

# Agregar los documentos al vector store
ids = [doc.metadata["id"] for doc in documents]
vectorstore.add_documents(documents=documents, ids=ids)

# Crear el modelo de chat
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

# Definir un nuevo grafo
workflow = StateGraph(state_schema=MessagesState)

# Función para generar el contexto
def generate_context(query):
    results = vectorstore.similarity_search(query, k=5)
    context = ""
    for result in results:
        context += f"Descripción: {result.page_content}\n"
        context += f"Detección: {result.metadata['detection']}\n"
        context += f"Mitigaciones: {result.metadata['mitigation_methods']}\n\n"
    return context

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    # Actualizar el historial de mensajes con la respuesta
    return {"messages": response}

# Añadir nodo al grafo
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Añadir memoria
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Contexto para la conversación actual (thread_id=1111)
config = {"configurable": {"thread_id": "1111"}}

# System prompt
texto = ("Eres un asistente experto en ciberseguridad utilizando la base de datos MITRE ATT&CK. Si el usuario no pregunta direcatamente sobre un ataque, no respondas con información sobre un ataque. Usa el contexto de otros ataques proporcionados para comentar la deteccion y mitigacion del ataque que se te pregunte")
prompt_base = SystemMessage(texto)
# Carga del system prompt inicial en la memoria
output = app.invoke({"messages": [prompt_base]}, config)

print("CHATBOT con historia.\nLo sé todo sobre ciberseguridad, ¡pregúntame!.\nFinalizar sesión con los comandos :salir, :exit o :terminar\n")
while True:
    query = input("\n>> ")
    if query.lower() in [":salir", ":exit", ":terminar"]:
        print("Gracias por hablar conmigo!!")
        sys.exit(0)

    # Crear el mensaje de entrada con el contexto adicional
    input_messages = [HumanMessage(f"{query}\n\nAtaques similares:\n{generate_context(query)}")]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()