from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

PERSONALITY = (
    """
# Role
You are an empathetic assistant specialized in wellness, your goal is to help users improve their mental health and emotional well-being through understanding its goals, lifestyle, and challenges.

# Instructions

# Rules

"""  )

# Modelo local (llama3:8b ejecutándose vía Ollama en localhost)
LANGUAGE_MODEL = ChatOllama(
    model="llama3:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

VECTOR_DB = []

dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



class AgentState(dict):
    """Estructura que viaja entre nodos del grafo."""
    msg: str
    history: list  # lista de tuplas (role, content)

# Prompt template con personalidad + memoria
template = ChatPromptTemplate.from_messages([
    ("system", PERSONALITY),
    ("system", "{memory}"),
    ("human", "{input}"),
])

MAX_TURNS = 4  # Número de pares usuario/asistente que se conservan

def chat_node(state: AgentState) -> AgentState:
    """Nodo principal: compone el prompt, invoca el modelo y actualiza la memoria."""
    # a) Combina la historia en texto
    mem_text = "\n".join(f"{role}: {content}" for role, content in state["history"])
    # b) Crea prompt y llama al LLM
    prompt = template.format(memory=mem_text, input=state["msg"])
    reply = llm.invoke(prompt)
    print("\n🤖", reply.content, "\n")

    # c) Actualiza la historia manteniendo solo los últimos MAX_TURNS pares
    new_hist = state["history"] + [
        ("user", state["msg"]),
        ("assistant", reply.content),
    ]
    state["history"] = new_hist[-MAX_TURNS*2:]
    return state

# Construcción del grafo mínimo
g = StateGraph(AgentState)
g.add_node("CHAT", chat_node)
g.add_edge(START, "CHAT")
g.add_edge("CHAT", END)

bot = g.compile()

if __name__ == "__main__":
    history = []
    while True:
        user = input("Tú > ")
        if user.lower() in {"salir", "exit"}:
            break
        bot.invoke({"msg": user, "history": history})
