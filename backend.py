from bs4 import BeautifulSoup
import os

website_extensions = {".html", ".htm"}

def extract_text_from_html(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(tuple(website_extensions)):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator=' ')
                non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
                filtered_text = '\n'.join(non_empty_lines)
                data.append({"filename": filename, "content": filtered_text})
    return data