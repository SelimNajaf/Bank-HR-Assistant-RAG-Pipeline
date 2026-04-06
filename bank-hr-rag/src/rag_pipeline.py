"""
Module defining the main Retrieval-Augmented Generation (RAG) pipeline for the HR Assistant.
"""

import os
from typing import List

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from vector_database import get_retriever

def format_docs(docs: List[Document]) -> str:
    """
    Formats a list of LangChain Document objects into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline() -> any:
    """
    Initializes the LLM, retriever, and prompt template to create the RAG execution chain.

    Returns:
        Any: A runnable LangChain sequence for the RAG pipeline.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            'HF_TOKEN not found. Please set it in your terminal: export HF_TOKEN="hf_..."'
        )

    # Initialize base LLM endpoint
    llm_base = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        provider="featherless-ai",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        temperature=0.1,
    )
    
    llm = ChatHuggingFace(llm=llm_base)

    # Define the prompt template (kept in original language for business context)
    template_str = """
    Sən Kapital Bankın HR köməkçisisən. Aşağıdakı məlumatlara əsasən suala peşəkar və qısa cavab ver. 
    Əgər cavab məlumatda yoxdursa, "Bu barədə daxili qaydalarda məlumat tapılmadı" de.

    Məlumat: {context}

    Sual: {input}
    """
    
    prompt = PromptTemplate.from_template(template_str)
    retriever = get_retriever()

    # Construct the chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print("\n--- Initializing Pipeline... ---\n")
    try:
        pipeline = setup_rag_pipeline()
        
        test_query = "Xəstəlik vərəqəsi neçə gün ərzində HR-a təqdim edilməlidir?"
        print(f"USER QUERY: {test_query}\n")
        print("--- Sending Query... ---\n")
        
        response = pipeline.invoke(test_query)
        
        print("BANK ASSISTANT:")
        print(response)
        
    except Exception as error:
        print(f"An error occurred: {error}")