from fastapi import FastAPI
from models import Query, QueryResponse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Initialize vector store
persist_directory = "./chorma_db"
collection_name = "Indian_Companies_Annual_Reports"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)

@app.get("/")
async def read_root():
    return {"message": "Vector DB query API is live."}

@app.post("/query", response_model=QueryResponse)
async def query_vector_db(query: Query):
    # Build Chroma filter using $and
    filter_conditions = [
        {
            "ticker": 
            {"$eq": query.ticker}     
        },
        {
            "year": 
            {"$eq": query.year}
        }
    ]
    
    if query.section:
        filter_conditions.append({
            "section_title": {"$eq": query.section}
            })
    
    # Create the filter with proper ChromaDB structure
    filters = {"$and": filter_conditions}

    # Perform the vector similarity search
    matched_docs = vector_db.similarity_search(
        query=query.question,
        k=query.k,
        filter=filters
    )

    return QueryResponse(
        chunks=[doc.page_content for doc in matched_docs],
        source_documents=list(set([doc.metadata.get("source", "") for doc in matched_docs ])),
        source_sections=list(set([doc.metadata.get("section_title", "") for doc in matched_docs])),
    )

