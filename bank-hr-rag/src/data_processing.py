"""
Module to handle data ingestion and text chunking for the RAG pipeline.
"""

import os
import json
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_chunks() -> List[Document]:
    """
    Reads the internal rules JSON data, creates Document objects, and splits them into manageable chunks.

    Returns:
        List[Document]: A list of chunked LangChain Document objects ready for vectorization.
    """
    base_dir = Path(__file__).resolve().parent.parent
    
    # Safely handle the environment variable fallback for the file path
    default_path = str(base_dir / 'data' / 'daxili_qaydalar_toy_data.json')
    file_path = Path(os.getenv('bank-hr-rag', default_path))

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    documents =[]

    # Map the JSON data (kept in original language keys) to LangChain Documents
    for section in data.get('bolmeler',[]):
        doc = Document(
            page_content=section.get('mezmun', ''),
            metadata={'section': section.get('bolme_adi', 'Unknown')}
        )
        documents.append(doc)

    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks