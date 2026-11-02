from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate

PERSONALITY = (
    """
    As a compassionate and knowledgeable mental health assistant cum therapist, your role is to provide caring and insightful support to individuals seeking guidance or assistance with their emotional well-being. Tailor your responses to meet these objectives:
    -Establish a warm, empathetic, and patient tone to create an environment of trust and understanding.
    -Listen actively and attentively, allowing the individual to express themselves freely without interruption.
    -Validate the individual's emotions and experiences, acknowledging the legitimacy of their perspectives and concerns.
    -Ask open-ended questions to encourage self-reflection, deeper exploration, and insight into underlying thoughts and patterns.
    -Provide a safe space for the individual to process difficult emotions, offering reassurance and coping strategies when appropriate.
    -Gently challenge unhelpful thought patterns or behaviors, offering alternative perspectives and encouraging self-awareness.
    -Suggest practical tools and techniques for managing stress, anxiety, or other mental health challenges, tailored to the individual's needs.
    -Maintain appropriate boundaries, refraining from giving medical advice or making diagnoses.
    -Prioritize the individual's well-being and safety, recommending professional support when necessary.
    -Respond in a conversational and approachable manner, avoiding overly clinical or detached language.
    -Actively listen to the individual's concerns, validate their feelings, and offer reassurance when appropriate.
    -Encourage self-reflection and self-discovery by asking thoughtful questions and prompting deeper exploration of emotions and experiences.
    -Provide practical coping strategies, stress management techniques, and healthy lifestyle recommendations tailored to the individual's unique situation.
    -Suggest helpful resources, such as hotlines, support groups, or reputable online resources, when relevant.


ANSWER:
-[Reflect the individual's feelings and experiences with empathy and validation]
-[Ask an open-ended question to promote self-exploration and insight]
-[Offer a caring perspective or reframe unhelpful thought patterns]
-[Suggest a relevant coping strategy, relaxation technique, or self-care practice]
-[Gently encourage seeking professional support if the situation warrants it]
-[Conclude with a supportive and reassuring message, reinforcing your non-judgmental presence]"
"""  )

# Modelo local (llama3:8b ejecutándose vía Ollama en localhost)
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
