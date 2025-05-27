# 📚 RAG System for Company Annual Reports (API-Based)

## 🔍 Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to query information from multiple years of company annual reports. It provides a FastAPI-based REST API backed by a local or hosted ChromaDB vector store.

---

## 🧱 Architecture Overview

### 🔹 Input Documents
- **Source**: Annual Reports in PDF format
- **Companies**: Multiple tickers supported
- **Years**: Multiple annual versions per company

---

### 🛠️ Preprocessing Pipeline

1. **Table of Contents Extraction**  
   - Parses the index section of each PDF to define semantic sections.

2. **Section-Based Splitting**
   - Text is split per section, not per page, to retain context.
   - Uses `PyPDFLoader` to load and segment content accordingly.

3. **Text Chunking**
   - Uses `RecursiveCharacterTextSplitter` with overlap:
     - `chunk_size=1500`, `chunk_overlap=250`
   - Keeps chunks semantically meaningful.

4. **Embeddings**
   - Uses `HuggingFaceEmbeddings` from `sentence-transformers/all-MiniLM-L6-v2`
   - Embeds each chunk into dense vector space.

5. **Vector Store**
   - Uses `ChromaDB` for local persistence (with option to host on GCP or Render).
   - Each document has metadata:
     - `year`, `ticker`, `section_title`, `page_start`, `page_end`, `source`

---

## 🚀 API Functionality (FastAPI)

### Features

- **Query Endpoint**
  - Allows semantic querying of the vector DB via REST.
  - Returns relevant sections with metadata.

- **Filtering**
  - Supports filters like `ticker`, `year`, `section_title`.

- **Model-Aware Chunking (Optional)**
  - Potential to implement token-based chunking using model-specific tokenizers.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| `FastAPI` | RESTful API server |
| `ChromaDB` | Vector DB backend |
| `HuggingFace Transformers` | Embeddings & tokenizers |
| `Docker` | Containerization |
| `GCP Cloud Run (optional)` | Deployment |
| `Pandas` | Data processing |
| `Loguru`, `dotenv` | Logging & config |

---

## 📦 Folder Structure (Suggested)

```
project-root/
├── api/
│   ├── chromadb/  
│   ├── main.py # FastAPI app
│   ├── models.py
    ├── Dockerfile

├── database/
│   └── __init__.py
│   └── documents.py 
│   └── generate_vdb.py 
│   └── pdf_extractor.py
            
├── data/
│   └── annual_reports/           

├── requirements.txt
├── .env
└── README.md
```
