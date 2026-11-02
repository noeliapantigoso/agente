from typing import Any, Dict
from src.generation import generation_chain
from langgraph_structure import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using documents and question."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
