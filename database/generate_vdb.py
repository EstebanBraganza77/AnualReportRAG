from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .pdf_extractors import extractors  # Assuming this is the correct import path for DMartARExtractor
from dotenv import load_dotenv
load_dotenv()
import traceback
import os
import re


def generate_vdb(pdf_path, year, ticker, index_page: int, sections:list = None) -> None:
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
    
    extractor = extractors[ticker](pdf_path, year, ticker, api_key=os.environ.get("OPENAI_API_KEY"), index_page=index_page)
    extractor.sections = sections if sections else extractor.clean_and_extract_index_mrf()

    section_docs = extractor.extract_section_documents()


    # Upload vdb and add documents to the vector store
    if not section_docs:
        print(f"No sections found in {pdf_path}. Skipping...")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    split_docs = text_splitter.split_documents(section_docs)

    vector_db.add_documents(split_docs)
    print(f"Added {len(split_docs)} chunks to the vector store for {ticker} {year}.")

def extract_list_with_brackets(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
     
    
if __name__ == "__main__":

    persist_directory = "../api/chroma_db"
    pdf_directory = "../data/anual_reports"
    tickers = ["TATAMOTORS"]
    collection_name = "Indian_Companies_Annual_Reports"

    # Initialize HuggingFace embeddings
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 

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
        pdf_path = os.path.join(pdf_directory, ticker)
        pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files for {ticker} in {pdf_path}.")
        
        for pdf_file in pdf_files:
            try:
                match = re.search(r'\d{4}', pdf_file)
                if not match:
                    print(f"Skipping {pdf_file}: No year found.")
                    continue
                year = int(match.group())
                pdf_file_path = os.path.join(pdf_path, pdf_file)
                print(f"Processing {pdf_file} for year {year}...")
                generate_vdb(pdf_file_path, year, ticker)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                traceback.print_exc()

    