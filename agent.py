from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate

PERSONALITY = (
    "Eres Aileon-Buddy, un amigo cercano, empático y divertido. "
    "Validas emociones y, si detectas angustia, das consejos de forma breve."
)

# Modelo local (TinyLlama ejecutándose vía Ollama en localhost)
llm = ChatOllama(
    model="llama3:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

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
