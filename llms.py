import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

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
prompt_template = """
Eres un asistente experto en ciberseguridad utilizando la base de datos MITRE ATT&CK.

Historial de la conversación:
{history}

Contexto relevante:
{context}

Pregunta del usuario:
{query}

Respuesta:
"""

# Función para generar el contexto
def generate_context(query):
    results = vectorstore.similarity_search(query, k=5)
    context = ""
    for result in results:
        context += f"Descripción: {result.page_content}\n"
        context += f"Detección: {result.metadata['detection']}\n"
        context += f"Mitigaciones: {result.metadata['mitigation_methods']}\n\n"
    return context

# Implementar el Chatbot con memoria
class ConversationalChatbot:
    def __init__(self, history_file="history.json"):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error al cargar el historial, iniciando con un historial vacío.")
            return []  # Si no existe o hay error, se inicializa vacío

    def save_history(self):
        # Guardar historial en archivo
        with open(self.history_file, "w") as f:
            json.dump(self.history, f)

    def chatbot(self, query):
        # Generar contexto desde un vector store
        context = generate_context(query)
        full_history = "\n".join(self.history)  # Concatenar historial
        prompt = prompt_template.format(history=full_history, context=context, query=query)
        
        # Usar modelo LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.predict(prompt)
        
        # Actualizar historial y guardar
        self.history.append(f"Usuario: {query}")
        self.history.append(f"Chatbot: {response}")
        self.save_history()
        
        return response

# Ejemplo de uso del Chatbot
chatbot_instance = ConversationalChatbot()
query = "An administrator noticed that a user account, which belongs to an employee on leave, was used to access sensitive files at odd hours"
response = chatbot_instance.chatbot(query)
print(response)