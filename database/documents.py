import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from openai import OpenAI
from typing import List, Dict, Optional
from pydantic import BaseModel


class DMartARExtractor:

    """
    Extracts and processes section-wise content from DMart Annual Reports in PDF format.

    Attributes:
        pdf_path (str): Path to the PDF file.
        year (int): Financial year of the report.
        ticker (str): Ticker symbol of the company.
        docs (list): List of Document objects representing pages of the PDF.
    """

    def __init__(self, pdf_path, year, ticker, api_key: str):
        """
        Initializes the extractor with the PDF file, year, and ticker.

        Args:
            pdf_path (str): Path to the PDF file.
            year (int): Financial year of the report.
            ticker (str): Ticker symbol of the company.
        """
        self.pdf_path = pdf_path
        self.year = year
        self.ticker = ticker
        self.docs = PyPDFLoader(pdf_path).load()
        self.section_descriptions_path = "./section_descriptions.json"
        self.client = OpenAI(api_key=api_key)
 


    def extract_index(self):
        """
        Extracts the content of the first page (assumed to be the index page).

        Returns:
            str: Text content of the index page.
        """
        for doc in self.docs:
            if doc.metadata.get("page") == 1:
                return doc.page_content

    def _extract_raw_text(self):
        """
        Helper method to extract the first page's text content.
        """
        return self.extract_index()

    def clean_and_extract_index_dmart(self):
        """
        Parses the index page to extract section titles and their starting page numbers.

        Returns:
            list[dict]: List of dictionaries with 'page_start' and 'title' keys.
        """
        raw_text = self._extract_raw_text()
        lines = raw_text.strip().splitlines()

        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "contents")
        except StopIteration:
            raise ValueError("No line labeled 'Contents' was found.")

        lines = lines[start_idx + 1:]

        entries = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(r"^(0*\d{1,3})(\s+.+)?$", line)

            if match:
                page = match.group(1)
                title = match.group(2).strip() if match.group(2) else ""

                # Concatenate continuation lines (not starting with a number)
                i += 1
                while i < len(lines) and not re.match(r"^\d{1,3}(\s|$)", lines[i].strip()):
                    title += " " + lines[i].strip()
                    i += 1

                page = str(int(page))
                title = ' '.join(title.strip().split()[0:6])

                entries.append({"page_start": page, "title": title.strip()})
            else:
                i += 1

        return entries

    def assign_end_pages_dmart(self):
        """
        Assigns ending page numbers to each section based on the start of the next section.
        Stores the result in self.sections.
        """
        sections = self.clean_and_extract_index_dmart()
        result = []

        for i, section in enumerate(sections):
            start = int(section["page_start"])
            end = int(sections[i + 1]["page_start"]) - 1 if i + 1 < len(sections) else None

            result.append({
                "title": section["title"].strip(),
                "page_start": start,
                "page_end": end
            })

        self.sections = result

    def extract_section_documents(self):
        """
        Extracts and creates Document objects for each report section.
        Stores them in self.section_documents.

        Returns:
            list[Document]: A list of LangChain Document objects for each section.
        """
        self.assign_end_pages_dmart()
        self.section_documents = []

        for section in self.sections:
            start = int(section["page_start"])
            end = section["page_end"]

            pages_in_section = [
                doc for doc in self.docs  # ✅ FIXED: `docs` → `self.docs`
                if doc.metadata.get("page_label") is not None and doc.metadata["page_label"].isdigit()
                and start <= int(doc.metadata["page_label"]) <= (end if end is not None else start)
            ]

            section_text = "\n\n".join([page.page_content for page in pages_in_section])
            try:
                section_description = self.generate_section_description(section["title"], section_text)
                added_description = self.add_new_section_description(section_description)
            except Exception as e:
                added_description = None
                print(f"Error generating description for section: {section['title']} - {e}")


            section_doc = Document(
                page_content=section_text,
                metadata={
                    "year": self.year,
                    "ticker": self.ticker,
                    "section_title": section["title"],  # ✅ FIXED: `self.section["title"]` → `section["title"]`
                    "page_start": start,
                    "page_end": end,
                    "source": os.path.basename(self.pdf_path),
                }
            )
            self.section_documents.append(section_doc)

        return self.section_documents

    def generate_section_description(self, section: str, context: str) -> dict:
 
        prompt = f"""
        You are an expert financial analyst. Given a section title and its content, generate a clear and informative description that outlines what type of information this section contains.

        Do not summarize every detail. Instead, highlight the general purpose and scope of the section so that another language model can determine whether the section is relevant to a given question.

        The description should be no more than 200 words and should convey what kind of content the section includes (e.g., financial performance, governance policies, risk disclosures, operational metrics, etc.).

        ### Section Title: {section}
    """

        response = self.client.responses.create(
            model="gpt-4o-mini",
            instructions=prompt,
            input=context,
            temperature=0
        )
        description = response.output[0].content[0].text.strip()

        return {
            "section": section,
            "description": description
        }

    def add_new_section_description(self, description_dict: dict) -> None:
        """
        Adds a new section description to the section descriptions file.

        Args:
            description_dict (dict): Dictionary with one section-title: description pair
        """

        file_path = self.section_descriptions_path

        # Asegúrate de que el archivo existe y no está vacío
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, "w") as file:
                json.dump({}, file)

        # Leer el contenido existente
        with open(file_path, "r") as file:
            data = json.load(file)

        # Inicializar estructura si no existe
        if self.ticker not in data:
            data[self.ticker] = {}

        if str(self.year) not in data[self.ticker]:
            data[self.ticker][str(self.year)] = {}

        # Agregar descripción al diccionario
        
        data[self.ticker][str(self.year)][description_dict.get('section', '')] = description_dict.get('description', '')

        # Escribir el nuevo contenido
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(data, file, indent=4)


class MRFARVectorizer:
    pass
