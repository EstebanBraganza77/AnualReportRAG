from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pdf_extractors import extractors  # Assuming this is the correct import path for DMartARExtractor
import traceback
import os
import re


def generate_vdb(pdf_path, year, ticker) -> None:
    """
    Inserts a vectorized pdf into a vector database from the given PDF document.

    Args:
        pdf_path (str): Path to the PDF file.
        year (int): Year of the report.
        ticker (str): Ticker symbol of the company.
        persist_directory (str): Directory to persist the vector database.

    Returns:
        Chroma: The vector store containing the embedded documents.
    """

    # Extract section-wise content from the PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file {pdf_path} does not exist.")
    
    extractor = extractors[ticker](pdf_path, year, ticker)
    section_docs = extractor.extract_section_documents()

    # Upload vdb and add documents to the vector store
    if not section_docs:
        print(f"No sections found in {pdf_path}. Skipping...")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    split_docs = text_splitter.split_documents(section_docs)

    vector_db.add_documents(split_docs)
    print(f"Added {len(split_docs)} chunks to the vector store for {ticker} {year}.")
     
    
if __name__ == "__main__":

    persist_directory = "../api/chorma_db"
    pdf_directory = "../data/anual_reports"
    tickers = ["DMART"]
    collection_name = "Indian_Companies_Annual_Reports"

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize the Chroma vector store (if exists, load; else create empty)
    if os.path.exists(persist_directory):
        vector_db = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    else:
        # If no DB exists, initialize an empty one for now
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name)
    

    for ticker in tickers:
        pdf_directory = os.path.join(pdf_directory, ticker)
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                match = re.search(r'\d{4}', pdf_file)
                if not match:
                    print(f"Skipping {pdf_file}: No year found.")
                    continue
                year = int(match.group())
                pdf_path = os.path.join(pdf_directory, pdf_file)
                print(f"Processing {pdf_file} for year {year}...")
                generate_vdb(pdf_path, year, ticker)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                traceback.print_exc()

    # Persist the vector store
    vector_db.persist()
    