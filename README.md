# 🏦 Bank HR Assistant: RAG Pipeline

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Production-green.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Mistral_7B-yellow.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-lightgrey.svg)

## 📖 Business Context & Objective

At large financial institutions like **Bank**, the Human Resources department handles hundreds of routine employee queries daily—ranging from lunch break schedules to sick leave policies. 

**Objective:** This project aims to automate and streamline internal HR support by implementing a **Retrieval-Augmented Generation (RAG)** system. By embedding Bank's internal rules and policies into a vector database, this pipeline empowers an AI assistant to provide instant, highly accurate, and context-aware answers to employee questions in Azerbaijani, significantly reducing the HR team's manual workload.

## 🛠️ Tech Stack

This project leverages modern, open-source Generative AI tools to ensure data privacy and system efficiency:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | LangChain | Framework for connecting the LLM, prompts, and vector store. |
| **Embeddings** | Hugging Face | `all-MiniLM-L6-v2` for generating dense vector representations of text. |
| **Vector Database** | ChromaDB | Local, lightweight vector store for efficient document retrieval. |
| **LLM** | Mistral AI | `Mistral-7B-Instruct-v0.2` (via Hugging Face Endpoint) for natural language generation. |

## 📂 Directory Structure

```text
bank-hr-rag/
│
├── data/
│   ├── daxili_qaydalar_toy_data.json   # Internal HR rules and policies (Source Data)
│   └── chroma_db/                      # Persisted local Chroma vector database
│
├── embedding_intro.py                  # Standalone script demonstrating embedding generation
├── data_processing.py                  # Ingestion, parsing, and chunking logic
├── vector_database.py                  # ChromaDB initialization and retriever setup
├── rag_pipeline.py                     # Main execution script linking LLM, Prompt, and DB
│
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

## ⚙️ Installation Instructions

To replicate this project on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/SelimNajaf/Bank-HR-Assistant-RAG-Pipeline.git
cd bank-hr-rag
```

**2. Create and activate a virtual environment:**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
*(Note: Ensure libraries like `langchain`, `langchain-huggingface`, `langchain-chroma`, `transformers`, `torch`, and `pandas` are in your requirements.txt)*

**4. Set up Environment Variables:**
You will need a Hugging Face API token to query the Mistral model. 
```bash
export HF_TOKEN="your_huggingface_api_token_here"
```

## 🚀 Usage

### Running the RAG Pipeline
To test the HR Assistant and query the internal rules, run the main pipeline script. The script will automatically check for an existing vector database, create one if it is missing, and execute a sample query.

```bash
python rag_pipeline.py
```

### Testing the Embeddings
If you want to inspect how the Hugging Face `MiniLM` model tokenizes and embeds text into multi-dimensional vectors, run the standalone intro script:

```bash
python embedding_intro.py
```

## 📊 Key Results & Visuals

The system successfully grounds the LLM in internal documents, preventing hallucination. When prompted with a question, the chain fetches the top 4 most relevant document chunks and synthesizes an answer.

**Example Execution:**

```text
--- Initializing Pipeline... ---
Existing vector database found. Loading...

USER QUERY: Xəstəlik vərəqəsi neçə gün ərzində HR-a təqdim edilməlidir?

--- Sending Query... ---

BANK ASSISTANT:
Xəstəlik vərəqəsi işə çıxdığınız ilk gün ərzində HR departamentinə təqdim edilməlidir.
```

*If queried about a topic outside the provided JSON policies, the model correctly triggers its fallback instruction:*
> "Bu barədə daxili qaydalarda məlumat tapılmadı." (No information was found in the internal rules regarding this.)

## 🔮 Future Enhancements
* **FastAPI Integration:** Wrap the LangChain logic in a RESTful API to connect with a frontend interface.
* **Memory Management:** Add `ConversationBufferMemory` to allow the assistant to handle multi-turn conversations and follow-up questions.
* **Document Expandability:** Extend `data_processing.py` to ingest PDFs and Word documents alongside JSON.

---
*Built with 💻 and ☕ by [Selim Najaf / [LinkedIn](https://www.linkedin.com/in/selimnajaf/)]*
