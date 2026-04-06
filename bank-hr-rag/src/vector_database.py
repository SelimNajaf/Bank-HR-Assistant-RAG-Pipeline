"""
Module to initialize and retrieve the Chroma Vector Database.
"""

import os
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from data_processing import get_chunks

def get_retriever(k: int = 4) -> Any:
    """
    Initializes the vector database and returns a retriever interface.
    If the database does not exist, it processes the text chunks and builds it.

    Args:
        k (int): Number of nearest neighbor documents to retrieve.

    Returns:
        Any: A LangChain vector store retriever instance.
    """
    base_dir = Path(__file__).resolve().parent.parent
    default_chroma_path = str(base_dir / 'data' / 'chroma_db')
    
    chroma_file_path = Path(os.getenv('bank-hr-rag', default_chroma_path))
    
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    # Optimization: Only generate chunks if the database doesn't already exist
    if chroma_file_path.exists():
        print("Existing vector database found. Loading...")
        vector_db = Chroma(
            persist_directory=str(chroma_file_path), 
            embedding_function=embedding_model
        )
    else:
        print("Vector database not found. Creating a new one...")
        chunks = get_chunks()
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model, 
            persist_directory=str(chroma_file_path)
        )

    return vector_db.as_retriever(search_kwargs={"k": k})