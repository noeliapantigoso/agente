from typing import List, Tuple, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate

"""Buddy Agent — Versión simplificada (solo modo CHAT)

Elimina el nodo ROUTER y los modos THERAPY/EXERCISE. Mantiene:
- Personalidad amigable
- Memoria corta de los últimos turnos
- Grafo mínimo START → CHAT → END
"""

# ──────────────────────────────────────────────────────────────
# 1. Modelo
# ──────────────────────────────────────────────────────────────
llm = ChatOllama(
    model="llama3:8b",                 # cambia si usas otro
    base_url="http://localhost:11434",
    temperature=0.7,
)

# ──────────────────────────────────────────────────────────────
# 2. Prompt del modo CHAT
# ──────────────────────────────────────────────────────────────
CHAT_PROMPT = (
    "Eres Aileon‑Buddy, un amigo cercano, empático y divertido. "
    "Hablas de forma cálida y natural, validas emociones y mantienes "
    "una conversación ligera."
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", CHAT_PROMPT),
    ("system", "{memory}"),
    ("human", "{input}"),
])

# ──────────────────────────────────────────────────────────────
# 3. Estado y utilidades
# ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    msg: str
    history: List[Tuple[str, str]]  # [(role, content), ...]

MAX_TURNS = 4  # pares user/assistant a mantener

def update_history(history: List[Tuple[str, str]], user_msg: str, bot_msg: str) -> List[Tuple[str, str]]:
    new_hist = history + [("user", user_msg), ("assistant", bot_msg)]
    return new_hist[-MAX_TURNS * 2 :]

# ──────────────────────────────────────────────────────────────
# 4. Nodo CHAT
# ──────────────────────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    mem_text = "\n".join(f"{r}: {c}" for r, c in state["history"])
    prompt = chat_template.format(memory=mem_text, input=state["msg"])
    reply = llm.invoke(prompt)
    print("\n🤖", reply.content, "\n")
    state["history"] = update_history(state["history"], state["msg"], reply.content)
    return state

# ──────────────────────────────────────────────────────────────
# 5. Grafo mínimo
# ──────────────────────────────────────────────────────────────

g = StateGraph(AgentState)

g.add_node("CHAT", chat_node)

g.add_edge(START, "CHAT")

g.add_edge("CHAT", END)

bot = g.compile()

# ──────────────────────────────────────────────────────────────
# 6. Loop interactivo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hist: List[Tuple[str, str]] = []
    while True:
        user = input("Tú > ")
        if user.lower() in {"salir", "exit"}:
            break
        result = bot.invoke({"msg": user, "history": hist})
        hist = result["history"]

