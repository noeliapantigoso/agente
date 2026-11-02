from typing import Any, Dict
from langgraph_structure import GraphState
from data import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents from vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}