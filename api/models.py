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

class ExtractQuery(BaseModel):
    year: int
    ticker: str
    section: str

class ExtractQueryResponse(BaseModel):
    section_title: str
    text: str

class ExtractSections(BaseModel):
    year: int
    ticker: str

class SectionsResponse(BaseModel):
    year: int
    ticker: str
    section_names: list

