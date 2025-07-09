from bert_score import score
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate

# 1. Configuración del agente (igual que antes)
PERSONALITY = (
    "Eres Aileon-Buddy, un amigo cercano, empático y divertido. "
    "Validas emociones y, si detectas angustia, das consejos de forma breve."
)
llm = ChatOllama(model="tinyllama:1.1b-chat", base_url="http://localhost:11434", temperature=0.7)

template = ChatPromptTemplate.from_messages([
    ("system", PERSONALITY),
    ("system", "{memory}"),
    ("human", "{input}"),
])
MAX_TURNS = 4

def chat_once(msg, history):
    # Combina memoria
    mem_text = "\n".join(f"{r}:{c}" for r, c in history)
    prompt = template.format(memory=mem_text, input=msg)
    reply = llm.invoke(prompt)
    # Actualiza historia
    hist = history + [("user", msg), ("assistant", reply.content)]
    hist = hist[-MAX_TURNS*2:]
    return reply.content, hist

# 2. Datos de prueba
inputs = [
    "¿Cómo te sientes hoy?",
    "¿Qué puedo hacer para relajarme?",
    # ... más prompts
]
references = [
    "Estoy bien, gracias por preguntar. ¿Y tú cómo estás?",
    "Puedes probar ejercicios de respiración profunda y estiramientos suaves.",
    # ... respuestas esperadas
]

# 3. Genera salidas
generated = []
history = []  # si quieres resetear para cada ejemplo, pon history = []
for msg, gold in zip(inputs, references):
    out, history = chat_once(msg, history)
    generated.append(out)

# 4. Calcula BERTScore
P, R, F1 = score(generated, references, lang="es",
                 model_type="xlm-roberta-base", verbose=True)

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")
