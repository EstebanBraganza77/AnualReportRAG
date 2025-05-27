from pydantic import BaseModel

class Query(BaseModel):
    year: int
    ticker: str
    k: int = 5
    question: str
    section: str | None
    
class QueryResponse(BaseModel):
    chunks: list[str]
    source_documents: list[str]
    source_sections: list[str]

