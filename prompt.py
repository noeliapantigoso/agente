from langgraph.graph import StateGraph, Node
from langgraph.agent import Agent
from langchain_community.llms import Ollama

# Define a simple node that uses Ollama to generate a response
class OllamaNode(Node):
    def __init__(self, model_name="llama2"):
        self.llm = Ollama(model=model_name)

    def run(self, state):
        prompt = state.get("prompt", "")
        response = self.llm(prompt)
        return {"response": response}

# Create the graph with one node
graph = StateGraph()
graph.add_node("ollama_node", OllamaNode())

# Set the entry point
graph.set_entry_point("ollama_node")

# Create the agent
agent = Agent(graph)

# Example usage
if __name__ == "__main__":
    input_state = {"prompt": "Hello, how are you?"}
    output = agent.run(input_state)
    print(output["response"])