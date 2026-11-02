"""
Functions:

- 
- Response Generation
- 

- Retrieve context from previous conversations

Nodes:
- Node to decide if context is needed
- Node to retrieve context
- Node to decide if a tool is needed
- Node to grade retrieved context
- Node to generate response


Tools:
- Sentiment Analysis
- Goal Tracking
- Personality analysis
- Smartwatch Data Retrieval
- Memory Selector
"""


from typing import TypedDict, Callable, Dict, Any, Union, List
import os
from langgraph.graph import StateGraph, Node, START
from langgraph.agent import Agent
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from src.constants import DATASEARCH, GENERATE, TOOLS, GRADE, RETRIEVE

load_dotenv()

class State(TypedDict):
    """State schema"""
    query: str
    response: str
    data_search: bool
    data: List[str]


llm = Ollama(model="llama3:8b",
             temperature=0.7,
             base_url="http://localhost:11434")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def decide_to_generate(State):
    "Route to context retrieval or response generation"
    print("Assessing if context is needed...")
    return DATASEARCH if State["data_search"] else GENERATE

def decide_to_use_tools(State):
    "Route to tool usage or response generation"
    print("Assessing if tools are needed...")
    return TOOLS if State["data"] else GENERATE

def retrieve_context(State):
    "Route to context retrieval or response generation"
    print("Retrieving context...")
    query = State["query"]