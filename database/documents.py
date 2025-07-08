import os
import re
import json
import ast
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from openai import OpenAI
from typing import List, Dict, Optional
from pydantic import BaseModel
import tempfile
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract



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
            print(section['title'])
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
                self.add_new_section_description(section_description)
            except Exception as e:
                print(f"Error generating description for section: {section['title']} - {e}")
                pass

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
            print('Correctly created document', section['title'])

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
            input=context[:5000],
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
        with open(file_path, "r", encoding='utf-8') as file:
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


class MRFExtractor:
    def __init__(self, pdf_path: str, year: int, ticker: str, api_key: str, index_page: int, tesseract_path: str = None):
        """
        Initialize the MRF Annual Report Extractor
        
        Args:
            pdf_path: Path to the PDF file
            year: Year of the annual report
            ticker: Company ticker symbol
            api_key: OpenAI API key
            tesseract_path: Optional path to Tesseract OCR executable
        """
        self.pdf_path = pdf_path
        self.year = year
        self.ticker = ticker
        self.docs = PyPDFLoader(pdf_path).load()
        self.section_descriptions_path = "./section_descriptions.json"
        self.client = OpenAI(api_key=api_key)
        self.index_page = index_page  # Page number for the table of contents
        
        # Configure Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def _is_toc_image(self, page_content: str) -> bool:
        """Check if the table of contents appears to be an image"""
        # Heuristics for detecting image-based TOC
        return (len(page_content.strip()) < 100 or 
                "[Image]" in page_content or 
                "..." in page_content or  # Common OCR artifact
                sum(c.isalpha() for c in page_content) / len(page_content) < 0.5)  # Low text ratio

    def _extract_text_from_image_page(self, page_number: int) -> str:
        """Extract text from a specific page using OCR"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert the specific page to image with high DPI
                images = convert_from_path(
                    self.pdf_path, 
                    first_page=page_number,
                    last_page=page_number,
                    dpi=300,
                    poppler_path='C:/poppler/Release-24.08.0-0/poppler-24.08.0/Library/bin' # Add path if needed: r'C:\path\to\poppler-xx\bin'
                )
                
                if not images:
                    return ""
                
                # Save temp image and perform OCR
                temp_img_path = os.path.join(temp_dir, "toc_page.png")
                images[0].save(temp_img_path, 'PNG')
                
                # Custom Tesseract config for tables/contents
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(
                    Image.open(temp_img_path), 
                    config=custom_config,
                    lang='eng'
                )
                return text
        except Exception as e:
            print(f"OCR failed for page {page_number}: {str(e)}")
            return ""

    def extract_index(self) -> str:
        """Extract table of contents either from text or via OCR if it's an image"""
        try:
            first_page = next(doc for doc in self.docs if doc.metadata.get("page") == self.index_page)
            
            if self._is_toc_image(first_page.page_content):
                print("Detected image-based TOC, using OCR...")
                ocr_text = self._extract_text_from_image_page(2)
                return ocr_text if ocr_text else first_page.page_content
            return first_page.page_content
        except StopIteration:
            print("Could not find first page")
            return ""

    def _extract_raw_text(self) -> str:
        """Get raw text from the table of contents"""
        return self.extract_index()

    def clean_and_extract_index_mrf(self) -> None:
        """Parse and clean the MRF table of contents"""
        raw_text = self._extract_raw_text()
        print(f"Raw TOC text: {raw_text}")  # Debugging output

        prompt = f"""
        You are a Python function. Convert the following Table of Contents into a JSON-like list of dictionaries.

        Each dictionary must have:
        - "title": section title
        - "page_start": starting page number (integer)
        - "page_end": ending page number (integer)

        If "page_end" is not provided in the text, infer it by using the next section's "page_start" minus 1. For the last item, use null as "page_end".

        ⚠️ Return only the list. No explanation, no extra text, no formatting outside Python list syntax.

        Table of Contents:
        {raw_text.strip()}
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        print(f"OpenAI TOC response: {content}")  # Debugging output
        content = self.parse_openai_toc_response(content)
        self.sections = content
        print(f"Extracted sections: {self.sections}")

    def parse_openai_toc_response(self, content: str):
        # 1. Quita bloques de markdown
        content = re.sub(r"```(json)?", "", content).strip()

        # 2. Reemplaza comillas “inteligentes” por comillas normales
        content = content.replace("’", "'").replace("“", '"').replace("”", '"')

        # 3. Intenta parsear con json
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass  # Falló el JSON, intenta con eval seguro

        # 4. Si JSON falla, intenta con ast.literal_eval
        try:
            return ast.literal_eval(content)
        except Exception as e:
            print("Error evaluando el contenido:", e)
            raise ValueError("No se pudo parsear el contenido como lista de diccionarios.")

    def generate_section_description(self, section: str, context: str) -> dict:
        """Generate a description of the section using OpenAI"""
        prompt = f"""
        You are an expert financial analyst analyzing MRF's annual report. 
        Given a section title and content sample, create a concise description that:
        1. Identifies the type of information contained
        2. Explains the section's purpose
        3. Highlights key data points typically found here
        
        Section Title: {section}
        
        Content Sample:
        {context[0:5000]}...
        
        Respond with just the description (no headings or labels).
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256
            )
            description = response.choices[0].message.content.strip()
            return {"section": section, "description": description}
        except Exception as e:
            print(f"Error generating description for {section}: {str(e)}")
            return {"section": section, "description": "Content section"}

    def add_new_section_description(self, description_dict: dict) -> None:
        """Store section descriptions in JSON file"""
        try:
            if not os.path.exists(self.section_descriptions_path):
                with open(self.section_descriptions_path, "w") as file:
                    json.dump({}, file)

            with open(self.section_descriptions_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if self.ticker not in data:
                data[self.ticker] = {}
            if str(self.year) not in data[self.ticker]:
                data[self.ticker][str(self.year)] = {}

            data[self.ticker][str(self.year)][description_dict["section"]] = description_dict["description"]

            with open(self.section_descriptions_path, "w", encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving section description: {str(e)}")

    def extract_section_documents(self) -> List[Document]:
        """Extract all sections from the annual report as documents"""
        self.section_documents = []

        for section in self.sections:
            start = section["page_start"] 
            end = section["page_end"]  

            # Handle cases where page numbers might be missing
            if start == 0:
                continue

            pages_in_section = [
                doc for doc in self.docs
                if doc.metadata.get("page_label") and (doc.metadata["page_label"].isdigit() or isinstance(doc.metadata["page_label"], int))
                and (start) <= int(doc.metadata["page_label"]) <= ((end) if end is not None else (start -1) )
            ]

            if not pages_in_section:
                print(f"No pages found for section: {section['title']} (pages {start}-{end})")
                continue

            section_text = "\n\n".join([page.page_content for page in pages_in_section])


            if section_text.strip():
            
                # Generate and store description
                #try:
                #    section_description = self.generate_section_description(section["title"], section_text)
                #    self.add_new_section_description(section_description)
                #except Exception as e:
                #    print(f"Error processing section {section['title']}: {str(e)}")
                #    raise e

                # Create document with metadata
                section_doc = Document(
                    page_content=section_text,
                    metadata={
                        "year": self.year,
                        "ticker": self.ticker,
                        "section_title": section["title"],
                        "page_start": start,
                        "page_end": end,
                        "source": os.path.basename(self.pdf_path),
                        "report_type": "annual"
                    }
                )
                self.section_documents.append(section_doc)

            else:
                print(f"Section {section['title']} is empty or contains no valid text.")

        return self.section_documents

    def save_sections_to_json(self, output_path: str = None) -> None:
        """Save extracted sections to a JSON file"""
        if not output_path:
            output_path = f"mrf_{self.year}_sections.json"
        
        sections_data = []
        for doc in self.section_documents:
            sections_data.append({
                "title": doc.metadata["section_title"],
                "pages": f"{doc.metadata['page_start']}-{doc.metadata['page_end']}",
                "content_sample": doc.page_content[:500] + "...",
                "metadata": doc.metadata
            })
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(sections_data, f, indent=4, ensure_ascii=False)


class TATAExtractor:
    def __init__(self, pdf_path: str, year: int, ticker: str, api_key: str):
        """
        Initialize the TATA Annual Report Extractor
        
        Args:
            pdf_path: Path to the PDF file
            year: Year of the annual report
            ticker: Company ticker symbol
            api_key: OpenAI API key
        """
        self.pdf_path = pdf_path
        self.year = year
        self.ticker = ticker
        self.docs = PyPDFLoader(pdf_path).load()
        self.section_descriptions_path = "./section_descriptions.json"
        self.client = OpenAI(api_key=api_key)