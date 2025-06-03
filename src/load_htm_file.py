from config import *
try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    from html.parser import HTMLParser

    class _FallbackParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    def BeautifulSoup(html, parser="html.parser"):
        p = _FallbackParser()
        p.feed(html)

        class _Doc:
            def get_text(self, separator="\n"):
                return separator.join(p.parts)

        return _Doc()
import re

def select_relevant_sentences(text, question, top_k=5):
    # Tokenise
    question_words = set(re.findall(r'\w+', question.lower()))
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Score each sentence by keyword overlap
    scored = []
    for sentence in sentences:
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = question_words & sentence_words
        scored.append((len(overlap), sentence))

    # Sort and return top-k scoring sentences
    top_sentences = [s for _, s in sorted(scored, reverse=True)[:top_k]]
    return "\n".join(top_sentences)

def load_all_htm_files_content():
    """Recursively loads and compresses content from all .htm files under HTM_FOLDER_PATH and its subdirectories."""
    all_texts = []
    file_count = 0

    if not os.path.exists(HTM_FOLDER_PATH):
        print(f"⚠️ Folder not found: {HTM_FOLDER_PATH}")
        return ""

    for root, _, files in os.walk(HTM_FOLDER_PATH):
        for filename in files:
            if filename.endswith(".htm"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "html.parser")
                        raw_text = soup.get_text(separator="\n").strip()
                        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

                        # Remove known boilerplate patterns
                        remove_phrases = {
                            "Click here to see this page in full context",
                            "*Maximize screen to view table of contents*",
                            "Back", "Forward",
                            "ODF Template File (ODT)"
                        }
                        clean_lines = [line for line in lines if line not in remove_phrases]
                        all_texts.append("\n".join(clean_lines))
                        file_count += 1
                        print(f"✅ Loaded: {filepath}")
                except Exception as e:
                    print(f"⚠️ Failed to load {filepath}: {e}")

    print(f"\n📊 Total .htm files loaded: {file_count}")
    return "\n".join(all_texts)

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
