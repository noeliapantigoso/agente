from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List

#State schema
class State(TypedDict):
    number1 : int
    number2: int
    number3: int
    number4: int
    operation: str
    operation2: str
    final1: int
    final2: int

def router1(state:State) -> State:
    if state["operation"] == "+":
        return("addition_operation")
    elif state["operation"] == "-":
        return("substraction_operation")

def add_node(state: State) -> State:
    state["final1"]= state["number1"] + state["number2"]
    return state

def substract_node(state: State) -> State:
    state["final1"]= state["number1"] - state["number2"]
    return state

def router2(state:State) -> State:
    if state["operation2"] == "+":
        return("addition_operation2")
    elif state["operation2"] == "-":
        return("substraction_operation2")

def add_node2(state:State) -> State:
    state["final2"]= state["number3"] + state["number4"]
    return state

def substract_node2(state:State) -> State:
    state["final2"]= state["number3"] - state["number4"]
    return state

graph = StateGraph(State)
graph.add_node(add_node)
graph.add_node(add_node2)
graph.add_node(substract_node)
graph.add_node(substract_node2)
graph.add_node("router1", lambda state:state)
graph.add_node("router2", lambda state:state)

graph.add_edge(START, "router1")
graph.add_conditional_edges(
    "router1",
    router1,
    {#Edge:Node
        "addition_operation":"add_node",
        "substraction_operation":"substract_node"}
)
graph.add_edge("add_node", "router2")
graph.add_edge("substract_node", "router2")
graph.add_conditional_edges(
    "router2",
    router2,
    {#Edge:Node
        "addition_operation2":"add_node2",
        "substraction_operation2":"substract_node2"}
)
graph.add_edge("add_node2", END)
graph.add_edge("substract_node2", END)

app = graph.compile()

print(app.invoke({"number1":10,"operation": "-", "number2": 5, "number3":7, "number4":2, "operation2" :"+", "final1":0 , "final2": 0})["final1"])

