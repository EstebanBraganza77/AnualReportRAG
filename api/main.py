import os
from fastapi import FastAPI
from models import Query, QueryResponse, ExtractQuery, ExtractQueryResponse, ExtractSections, SectionsResponse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Initialize vector store
persist_directory = "./chroma_db"
collection_name = "Indian_Companies_Annual_Reports"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 

vector_db = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)

@app.get("/")
async def read_root():
    """
    Root endpoint to check if the API is live.

    Returns:
        dict: A simple message indicating that the Vector DB query API is live.
    """
    return {"message": "Vector DB query API is live."}

@app.post("/query", response_model=QueryResponse)
async def query_vector_db(query: Query):
    """
    Perform a vector similarity search on the document database with optional filters.

    Args:
        query (Query): Query model containing the question, ticker, year, section (optional), 
                       and number of results (k).

    Returns:
        QueryResponse: Contains matched document chunks, source document names, and source sections.
    """
    # Build Chroma filter using $and
    filter_conditions = [
        {"ticker": {"$eq": query.ticker}},
        {"year": {"$eq": query.year}}
    ]
    
    if query.section:
        filter_conditions.append({"section_title": {"$eq": query.section}})
    
    filters = {"$and": filter_conditions}

    matched_docs = vector_db.similarity_search(
        query=query.question,
        k=query.k,
        filter=filters
    )

    return QueryResponse(
        chunks=[doc.page_content for doc in matched_docs],
        source_documents=list(set([doc.metadata.get("source", "") for doc in matched_docs])),
        source_sections=list(set([doc.metadata.get("section_title", "") for doc in matched_docs])),
    )

@app.post("/extract_section", response_model=ExtractQueryResponse)
async def extract_section_text(query: ExtractQuery):
    """
    Extract the full text content of a specific section for a given ticker and year.

    Args:
        query (ExtractQuery): Model containing ticker, year, and section title.

    Returns:
        ExtractQueryResponse: The requested section title and concatenated text content, or a
                              message if no content is found.
    """
    all_data = vector_db._collection.get(include=["documents", "metadatas"])
    filtered = [
        doc for doc, meta in zip(all_data['documents'], all_data['metadatas'])
        if meta.get("ticker") == query.ticker and meta.get("year") == query.year and meta.get('section_title') == query.section
    ]
    
    if not filtered:
        return ExtractQueryResponse(section_title=query.section, text="No content found for the given query.")

    return ExtractQueryResponse(section_title=query.section, text='\n'.join(filtered))

@app.post("/extract_section_names", response_model=SectionsResponse)
async def extract_sections(data: ExtractSections):
    """
    Retrieve all available section names for a given ticker and year.

    Args:
        data (ExtractSections): Model containing ticker and year.

    Returns:
        SectionsResponse: Contains the ticker, year, and a list of section names found in the documents.
    """
    all_data = vector_db._collection.get(include=["metadatas"])  # Only metadata
    
    filtered_metas = [
        meta for meta in all_data["metadatas"]
        if meta.get("ticker") == data.ticker and meta.get("year") == data.year
    ]
    
    sections = list({meta.get("section_title") for meta in filtered_metas if meta.get("section_title")})

    return SectionsResponse(year=data.year, ticker=data.ticker, section_names=sections)
