import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3:8b")

def process(state: AgentState) -> AgentState:
    """This node will solve the user's input"""
    response = llm.invoke(state["messages"])

    #response converted to AI message and appended
    state["messages"].append(AIMessage(content=response.content))

    print(f"\n AI:{response.content}")

    return(state)

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history=[]

user_input = input("you:")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    conversation_history= result["messages"]
    user_input= input("you:")


