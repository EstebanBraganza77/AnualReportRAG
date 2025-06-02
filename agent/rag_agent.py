from typing import Callable, Dict, List, Tuple, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import os
from openai import OpenAI
import json


# Load environment variables
load_dotenv()
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# === Pydantic Models ===
class Query(BaseModel):
    year: int
    ticker: str
    k: int = 5
    question: str
    section: Optional[str] = None


class QueryResponse(BaseModel):
    chunks: List[str]
    source_documents: List[str]
    source_sections: List[str]


# === Step 1: Build prompt to choose relevant sections ===
def build_section_selection_prompt(section_descriptions: Dict[str, str], sections_to_select: int = 2) -> str:
    sections_text = "\n".join([f"- {key}: {desc}" for key, desc in section_descriptions.items()])
    return  f"""
You are an assistant specialized in analyzing companies' annual reports. Given a user's question and a list of available sections (with descriptions), select the {sections_to_select} most relevant sections to answer the question.

### Available Sections:
{sections_text}

Respond with a JSON list of section names. Example:
["Section A", "Section B"]
If no relevant section is found, respond with: []
"""
# === Step 2: Use OpenAI to rephrase the question to a more relevant one ===
def rephrase_question(question: str, ticker, year) -> str:
    prompt = prompt = f"""
    You are an expert financial analyst.
    Given a user's question about a company's annual report, rephrase the question into a clear and concise standalone query that could be used as a semantic search input in a vector database.
    Preserve the core intent, and include relevant context (e.g., company name, year, topics like risk, strategy, financials, etc.). Avoid vague phrasing or conversational tone.
    ### Company Ticker: {ticker}
    ### Year: {year}"""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=prompt,
        input=question,
        temperature=0
    )
    return response.output[0].content[0].text.strip()

# === Step 3: Use OpenAI to choose sections ===
def select_sections_with_llm(question: str, section_descriptions: Dict[str, str]) -> List[str]:
    prompt = build_section_selection_prompt(section_descriptions)
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=prompt,
        input=question,
        temperature=0
    )
     
    content = response.output[0].content[0].text.strip()
    try:
        selected_sections = json.loads(content)
        if isinstance(selected_sections, list):
            return selected_sections
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
    return []


# === Step 4: Call your RAG API ===
def ask_sections(
    question: str,
    ticker: str,
    year: int,
    selected_sections: List[str],
    k: int = 5
) -> List[Tuple[str, QueryResponse]]:
    results = []
    for section in selected_sections:
        query = Query(year=year, ticker=ticker, question=question, section=section, k=k)
        response = api_call(query)
        results.append((section, response))
    return results


# === Step 4: Build final response ===
def build_final_answer(question: str, results: List[Tuple[str, QueryResponse]], ticker: str, year: int) -> str:
    chunks_by_section = []

    for section, result in results:
        documents = result.source_documents[0] if result.source_documents else "N/A"
        for chunk in result.chunks:
            chunks_by_section.append(f"Section: {section} Document: {documents} \nContent: {chunk}")

    instructions = f"""You are an expert financial analyst. Given a user's question about a company's, answer the question using the information provided below, which has been extracted from the annual report of the company. Provide where did you find the information after each idea.
    ### Company Ticker: {ticker}
    ### Year: {year}
    ### Context:
    {"\n\n".join(chunks_by_section)}"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=question,
        temperature=0.5
    )
    return response.output[0].content[0].text.strip()

# === Orchestration function ===
def handle_question(
    question: str,
    ticker: str,
    year: int,
    section_descriptions: Dict[str, str],
    k: int = 5
) -> str:
    # Step 1: Rephrase the question
    #question = rephrase_question(question, ticker, year)
    #print(f"Rephrased Question: {question}")
    selected_sections = select_sections_with_llm(question, section_descriptions)
    print(f"Selected Sections: {selected_sections}")
    results = ask_sections(question, ticker, year, selected_sections, k=k)
    print(f"Final response:")
    return build_final_answer(question, results, ticker, year)


# === Main test driver ===
def api_call(query: Query) -> QueryResponse:
    url = os.getenv("RAG_API_URL")
    if not url:
        raise ValueError("RAG_API_URL environment variable is not set.")
    response = requests.post(url, json=query.dict())
    response.raise_for_status()
    data = response.json()

    return QueryResponse(
        chunks=data.get("chunks", []),
        source_documents=data.get("source_documents", []),  
        source_sections=[query.section or "unknown"]
    )


def main():
    section_descriptions = {
        "Management ": "Includes revenues, expenses, assets, and liabilities.",
        "risk": "Details key risks such as financial, operational, and regulatory.",
        "governance": "Covers the board of directors and corporate governance policies."
    }

    ticker = "DMART"
    year = 2024
    question = f"What risks an concerns where mentioned in {ticker}'s {year} annual report?"

    answer = handle_question(
        question=question,
        ticker=ticker,
        year=year,
        section_descriptions=section_descriptions
    )

    print(answer)


if __name__ == "__main__":
    main()
