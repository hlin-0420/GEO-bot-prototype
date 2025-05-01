from config import *
from bs4 import BeautifulSoup

def load_htm_file_content():
    # 📥 Load and clean HTML content
    try:
        with open(HTM_FILE_PATH, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            raw_text = soup.get_text(separator="\n").strip()
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            
            # ❌ Remove known boilerplate patterns
            remove_phrases = {
                "Click here to see this page in full context",
                "*Maximize screen to view table of contents*",
                "Back", "Forward",
                "ODF Template File (ODT)"  # May appear at top & again
            }
            clean_lines = [line for line in lines if line not in remove_phrases]

            # Join cleaned lines
            odf_text = "\n".join(clean_lines)
    except Exception as e:
        print(f"⚠️ Failed to load or parse HTML file: {e}")
        odf_text = ""
        
    return odf_text