from bs4 import BeautifulSoup
import os
from langchain_core.documents import Document as LangchainDocument
from app.services.document_processor import DocumentProcessor

def process_single_file(file_path, selectedOptions):
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
        content = content[content.find("<body>")+6:content.find("</body>")]
        soup = BeautifulSoup(content, "html.parser")

        parts = []
        if "text" in selectedOptions:
            parts.append(DocumentProcessor.extract_text(soup))
        if "table" in selectedOptions:
            parts.append(DocumentProcessor.extract_table_as_text_block(soup, file_path))
        if "list" in selectedOptions:
            parts.append(DocumentProcessor.extract_list(soup))

        # Flatten, filter non-string, and ensure all items are stringified
        flattened_parts = []

        for part in parts:
            if isinstance(part, list):
                for item in part:
                    if item:  # not None or empty
                        flattened_parts.append(str(item))
            elif part:
                flattened_parts.append(str(part))

        combined_text = "\n".join(flattened_parts)

        return LangchainDocument(
            page_content=combined_text,
            metadata={
                'links': [a['href'] for a in soup.find_all('a', href=True)],
                'page_name': os.path.basename(file_path)
            }
        )
        