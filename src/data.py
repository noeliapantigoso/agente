import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langgraph_structure import eembedding_model


def vector_store() -> Chroma:
    """Initialize and return a Chroma vector store."""
    path_chroma = './db'
    if os.path.exists(path_chroma):
        print("Loading existing vector store")
        vector_store = Chroma(persist_directory=path_chroma,
                              embedding_function=embedding_model,
                              collection_name="rag_collection")
        return vector_store.as_retriever()
    
    print("Creating directory.")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, 
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)


    vector_store = Chroma.from_documents(  
                          documents=doc_splits,
                          embedding_function=embedding_model,
                          collection_name="rag_collection",
                          persist_directory=path_chroma)
    
    print("Vector store created")
    return vector_store.as_retriever()

retriever = vector_store()



