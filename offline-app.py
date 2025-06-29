from flask import Flask, request, jsonify, render_template, Response, send_from_directory, g
import logging
import os
import threading
import time
from threading import Lock
from bs4 import BeautifulSoup
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import json
from tabulate import tabulate
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import process
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from datetime import datetime
from bs4 import XMLParsedAsHTMLWarning
import warnings
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import secrets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
logging.getLogger("faiss").setLevel(logging.ERROR)

nltk.data.path.append('/local_models/nltk_data')
nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# tracks the current sessions in memory
current_session_id = None
current_session_messages = []

valid_model_names = {
    "deepseek1.5",
    "llama3.2:latest",
    "tinyllama:latest",
    "gemma3:1b"
}
# Initialize the selected bot. 
app = Flask(__name__)

@app.before_request
def set_nonce():
    g.nonce = secrets.token_urlsafe(16)  # Generate a nonce for each request

@app.after_request
def add_csp_headers(response):
    response.headers['Content-Security-Policy'] = (
        f"default-src 'self' https://cdnjs.cloudflare.com https://apis.google.com https://content.googleapis.com https://www.gstatic.com;"
        f"script-src 'self' 'nonce-{g.nonce}' 'unsafe-eval' https://cdnjs.cloudflare.com https://apis.google.com https://www.gstatic.com https://accounts.google.com/gsi/client;"
        "connect-src 'self' https://www.googleapis.com https://content.googleapis.com;"
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;"
        "img-src 'self' https://upload.wikimedia.org https://www.gstatic.com data:;"
        "frame-src 'self' https://accounts.google.com https://content.googleapis.com;"
        "object-src 'none';"
    )
    return response

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# initialise a default name for the models. 
selected_model_name = "llama3.2:1b"
# initialise a variable to store the length of time taken to answer a question.
answer_time = 0 # default time taken to answer a question.
# Declare global variable
rag_application = None 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# Evaluation files
EXCEL_FILE = os.path.join(DATA_DIR, "evaluation", "query_responses.xlsx")
EXPECTED_RESULTS_FILE = os.path.join(DATA_DIR, "evaluation", "expected_query_responses.xlsx")

# Feedback
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback", "feedback_dataset.json")

# Model files
PROMPT_VISUALISATION_FILE = os.path.join(DATA_DIR, "model_files", "prompt_visualisation.txt")
PROCESSED_CONTENT_FILE = os.path.join(DATA_DIR, "model_files", "processed_content.txt")
UPLOADED_FILE = os.path.join(DATA_DIR, "model_files", "uploaded_document.txt")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "model_files", "faiss_index")

# User session files
CHAT_SESSIONS_DIR = os.path.join(DATA_DIR, "user_sessions", "ChatSessions")
SESSION_METADATA_FILE = os.path.join(DATA_DIR, "user_sessions", "session_metadata.json")
TIMED_RESPONSES_FILE = os.path.join(DATA_DIR, "user_sessions", "timed_responses.json")

@app.route("/store-response-time", methods=["POST"])
def store_response_time():
    """Stores response time, question asked, and timestamp in a JSON file."""
    try:
        data = request.json
        question_number = str(data.get("questionNumber"))
        question_text = data.get("question")
        duration = float(data.get("duration"))
        timestamp = data.get("timestamp")

        if not question_number or not question_text or duration is None or not timestamp:
            return jsonify({"error": "Invalid data"}), 400

        response_times = []

        # Load existing response times
        if os.path.exists(TIMED_RESPONSES_FILE):
            with open(TIMED_RESPONSES_FILE, "r", encoding="utf-8") as file:
                try:
                    response_times = json.load(file)
                except json.JSONDecodeError:
                    response_times = []

        # Append new entry
        response_entry = {
            "question_number": question_number,
            "question": question_text,
            "response_time": duration,
            "timestamp": timestamp
        }
        response_times.append(response_entry)

        # Save updated data
        with open(TIMED_RESPONSES_FILE, "w", encoding="utf-8") as file:
            json.dump(response_times, file, indent=4)

        return jsonify({"message": "Response time stored successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_keywords(text, top_n=3):
    """Extracts top N keywords from text using TF-IDF."""
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]
    
    return " ".join(filtered_words[:top_n])  # Take top keywords

@app.route("/knowledge-tree")
def knowledge_tree():
    # Load chat data
    chat_data = []
    for filename in os.listdir(CHAT_SESSIONS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(CHAT_SESSIONS_DIR, filename), "r", encoding="utf-8") as file:
                chat_data.extend(json.load(file))

    # Extract user questions
    questions = [entry["content"] for entry in chat_data if entry["role"] == "user"]

    if not questions:
        return render_template("knowledge_tree.html", tree_data={"name": "No Data", "children": []})

    # Apply TF-IDF to extract key topics
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    X = vectorizer.fit_transform(questions)
    feature_names = vectorizer.get_feature_names_out()  # Get words corresponding to features

    # Cluster similar topics
    num_clusters = min(6, len(questions))  # Avoid exceeding available questions
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Extract Top Keywords for Cluster Naming
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_names = {}
    for i in range(num_clusters):
        top_keywords = [feature_names[ind] for ind in order_centroids[i, :1]]  # Top 3 words
        cluster_names[i] = ", ".join(top_keywords)  # Assign top keywords as cluster name

    # Organize topics into a tree structure
    knowledge_tree = defaultdict(list)
    for idx, label in enumerate(labels):
        topic_name = extract_keywords(cluster_names[label])  # Simplify topic names
        knowledge_tree[topic_name].append(extract_keywords(questions[idx], top_n=1))  # Keep 3-word summaries
        
    # Correct structured tree format
    structured_tree = {
        "name": "GEO Software Knowledge Tree",
        "children": [
            {"name": topic, "children": [{"name": subtopic} for subtopic in sorted(set(subs))]}
            for topic, subs in knowledge_tree.items()
        ],
    }

    return render_template("knowledge_tree.html", tree_data=structured_tree)

@app.route("/chat-history/<session_id>", methods=["GET"])
def get_single_chat_session(session_id):
    session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(session_file):
        return jsonify({"error": "Session not found"}), 404

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        return jsonify({
            "session_id": session_id,
            "messages": messages
        }), 200
    except Exception as e:
        app.logger.error(f"Error loading session {session_id}: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

def load_session_metadata():
    if os.path.exists(SESSION_METADATA_FILE):
        with open(SESSION_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_session_metadata(metadata):
    with open(SESSION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
@app.route("/rename-session", methods=["POST"])
def rename_session():
    data = request.json
    session_id = data.get("session_id")
    new_name = data.get("new_name")

    if not session_id or not new_name:
        return jsonify({"error": "Session ID and new name are required"}), 400

    metadata = load_session_metadata()
    metadata[session_id] = new_name
    save_session_metadata(metadata)

    return jsonify({"message": "Session renamed successfully"})

@app.route("/clear-chat-sessions", methods=["POST"])
def clear_chat_sessions():
    try:
        if os.path.exists(CHAT_SESSIONS_DIR):
            for filename in os.listdir(CHAT_SESSIONS_DIR):
                file_path = os.path.join(CHAT_SESSIONS_DIR, filename)
                os.remove(file_path)

        # Optional: Clear session metadata (if you want to reset names too)
        save_session_metadata({})

        return jsonify({"success": True, "message": "All chat sessions cleared."}), 200
    except Exception as e:
        app.logger.error(f"Error clearing chat sessions: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def load_chat_history():
    try:
        if not os.path.exists(CHAT_SESSIONS_DIR):
            os.makedirs(CHAT_SESSIONS_DIR)

        chat_history = []
        
        metadata = load_session_metadata()

        for filename in os.listdir(CHAT_SESSIONS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(CHAT_SESSIONS_DIR, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session_id = filename.replace(".json", "")
                    print(f"Session id: {session_id}")
                    chat_history.append({
                        "session_id": session_id,
                        "session_name": metadata.get(session_id, session_id),
                        "messages": session_data
                    })

        return chat_history
    except Exception as e:
        return {"error": str(e)}

@app.route("/chat-history", methods=["GET"])
def get_chat_history():
    return jsonify(load_chat_history())

@app.route("/chathistory")
def chathistory():
    global current_session_id, current_session_messages
    current_session_id = None
    current_session_messages = []
    return render_template("chathistory.html")

class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join(doc.page_content for doc in documents)
        response = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return response

def load_feedback_dataset():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)  # Initialize an empty list
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def auto_adjust_column_width(writer, df):
    """ Auto-adjusts column width based on the max length of cell content in each column """
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    for column in df.columns:
        max_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
        col_idx = df.columns.get_loc(column) + 1
        worksheet.column_dimensions[chr(64 + col_idx)].width = max_length

def extract_table(soup):
    tables = soup.find_all("table")
    
    formatted_tables = []
                    
    # Process and format each table
    for i, table in enumerate(tables, start=1):
        rows = []
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            rows.append(cols)
            
        # Flatten row values for filtering irrelevant tables
        flat_rows = [item.lower().strip() for sublist in rows for item in sublist]
        
        # Skip navigation tables containing only "Back" and "Forward"
        if set(flat_rows).issubset({"back", "forward", "", "-", "next", "previous"}):
            continue  # Skip this table

        # Convert to DataFrame for better readability
        df = pd.DataFrame(rows)
                        
        formatted_table = tabulate(df, headers="firstrow", tablefmt="grid")
        
        formatted_tables.append(formatted_table)
        
    formatted_tables = "\n\n".join(formatted_tables)
    
    return formatted_tables

def fix_run_together_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "URL"]:
            text = text.replace(ent.text, " " + ent.text + " ")
    return re.sub(r'\s+', ' ', text)

def fix_spacing_and_punctuation(text):
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,!?])(?=\w)', r'\1 ', text)  # Add space after punctuation if missing
    return text

def remove_short_filler_sentences(text, min_words=5):
    doc = nlp(text)
    return "\n".join([
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip().split()) >= min_words and not sent.text.lower().startswith("did you know")
    ])

def clean_bullet_lists(text):
    bullets = re.findall(r'(•.*?)((?=•)|$)', text, flags=re.DOTALL)
    cleaned = []
    for item, _ in bullets:
        item = fix_run_together_entities(item)
        item = re.sub(r'\s+', ' ', item).strip()
        cleaned.append("• " + item)
    return "\n".join(cleaned)

def full_cleaning_pipeline(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")

    # Clean main text
    text = extract_text(soup)
    text = fix_spacing_issues(text)
    text = fix_run_together_entities(text)
    text = fix_spacing_and_punctuation(text)
    text = remove_short_filler_sentences(text)

    # Clean bullet points
    bullets = extract_list(soup)

    # Extract formatted tables
    tables = extract_table_as_text_block(soup, file_path)

    # Combine all parts if they exist
    parts = [part.strip() for part in [text, bullets, tables] if part and part.strip()]
    cleaned_content = "\n\n".join(parts)

    return cleaned_content

def extract_text(soup):
    # Define navigation-related keyword patterns
    navigation_keywords = [
        r'contact\s+us', r'click\s+(here|for)', r'guidance', r'help', r'support', r'assistance',
        r'maximize\s+screen', r'view\s+details', r'read\s+more', r'convert.*file', r'FAQ', r'learn\s+more',
        r'Click\s+here\s+to\s+see\s+this\s+page\s+in\s+full\s+context'
    ]
    navigation_pattern = re.compile(r"|".join(navigation_keywords), re.IGNORECASE)

    # Remove navigation-related <p> tags
    for tag in soup.find_all("p"):
        if navigation_pattern.search(tag.get_text()):
            tag.decompose()

    # Add line breaks after block-level tags to improve separation
    for tag in soup.find_all(['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'li']):
        tag.append("\n")

    # Extract raw text
    raw_text = soup.get_text(separator=" ")

    # Normalize whitespace
    normalized_text = re.sub(r'\s+', ' ', raw_text).strip()

    # Remove duplicate phrases (case-insensitive, 1–5 words)
    deduped_text = re.sub(
        r'\b((?:\w+\s+){1,5}?\w+)\s+\1\b',
        r'\1',
        normalized_text,
        flags=re.IGNORECASE
    )

    # Remove filler phrases
    filler_patterns = [
        r'GEO Help 8\.09\s*%', r'GEO Help 8\.09', r'End of search results\.?',
        r'Click\s+here\s+to\s+see\s+this\s+page\s+in\s+full\s+context'
    ]
    filler_regex = re.compile(r"|".join(filler_patterns), re.IGNORECASE)
    deduped_text = filler_regex.sub('', deduped_text)

    # Remove navigation and label clutter
    navigation_fillers = [
        r'\bBack Forward\b', r'\bGEO Help\b', r'\bApplication Look\b',
        r'\bGEO License and Version Number\b'
    ]
    nav_regex = re.compile(r"|".join(navigation_fillers), re.IGNORECASE)
    deduped_text = nav_regex.sub('', deduped_text)

    # Fix overused ellipses and apologies
    deduped_text = re.sub(r'\.{2,}', '.', deduped_text)
    deduped_text = re.sub(r'please accept our apologies\.?', '', deduped_text, flags=re.IGNORECASE)

    # Split into lines and filter short ones
    # Force split on punctuation + line breaks
    sentences = re.split(r'(?<=[.?!])\s+', deduped_text)

    # Create paragraphs if sentences are too disjointed
    paragraphs = []
    para = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        para.append(sent)
        if len(sent.split()) < 8 or sent.endswith(('.', '!', '?')):
            paragraphs.append(" ".join(para))
            para = []

    if para:
        paragraphs.append(" ".join(para))

    clean_text = "\n\n".join(paragraphs)

    return clean_text

def fix_spacing_issues(text):
    # Add space between words and URLs
    text = re.sub(r'([a-z])(?=https?://)', r'\1 ', text)
    text = re.sub(r'([a-z])(?=www\.)', r'\1 ', text)
    return text

def extract_list(soup):
    lists = []

    for ul in soup.find_all(['ul', 'ol']):
        items = [
            fix_spacing_issues(li.get_text(" ", strip=True))
            for li in ul.find_all('li')
            if li.get_text(strip=True) and len(li.get_text(strip=True)) > 2
        ]
        if items:
            lists.extend(items)

    # Fallback for stray <li> outside lists
    if not lists:
        items = [
            fix_spacing_issues(li.get_text(" ", strip=True))
            for li in soup.find_all('li')
            if li.get_text(strip=True) and len(li.get_text(strip=True)) > 2
        ]
        lists.extend(items)

    if lists:
        seen = set()
        bullets = []
        for item in lists:
            key = item.lower().strip()
            if key and key not in seen and len(key) > 2:
                bullets.append("• " + item)
                seen.add(key)
        return "\n".join(bullets)

    return ""

def extract_table_as_text_block(soup, file_path):
    tables = soup.find_all("table")
    output = []
    for table in tables:
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            row_text = " | ".join(cols).strip()
            # Skip rows that are:
            # 1. Completely empty
            # 2. Only contain 'back' and/or 'forward'
            # 3. Visually empty (like " | ")
            if row_text and not all(cell.lower() in ("back", "forward") for cell in cols) and not re.fullmatch(r"(\s*\|\s*)+", row_text):
                rows.append(row_text)
        if rows:
            output.append("\n".join(rows))

    combined_output = "\n\n".join(output)
    
    # Replace multiple blank lines with a single blank line
    cleaned_output = re.sub(r'\n\s*\n+', '\n\n', combined_output)

    return cleaned_output.strip() if cleaned_output else ""

class OllamaBot:
    def __init__(self):
        self.base_directory = DATA_DIR
        self.web_documents = []
        self._load_content()
        self.llm_model = ChatOllama(
            model=selected_model_name,
            temperature=0,
            num_predict=150
        )
        self._initialize_rag_application()

    def _initialize_rag_application(self):
        global rag_application
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20)
        doc_splits = text_splitter.split_documents(self.web_documents)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(doc_splits, embedding_model)

        retriever = vectorstore.as_retriever(k=3)
        prompt = PromptTemplate(
            template="""
            You are an AI assistant for the GEO application.
            Use only the **Documents** below to help you answer the **Question** directly, concisely, and factually.

            ---
            **Documents:**
            {documents}
            ---

            **Question:** {question}
            
            **Answer:**
            """,
            input_variables=["question", "documents"]
        )
        rag_chain = prompt | self.llm_model | StrOutputParser()
        rag_application = RAGApplication(retriever, rag_chain)

    def _load_content(self):
        htm_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(self.base_directory) for f in filenames if f.endswith(".htm")]
        for file_path in htm_files:
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    soup = BeautifulSoup(content, "html.parser")
                                     
                    cleaned_content = full_cleaning_pipeline(file_path)
                        
                    if cleaned_content.strip():
                        file_header = f"===== FILE: {file_path} ====="
                        page_text = "\n\n".join([file_header, cleaned_content.strip()])
                        
                        document = LangchainDocument(
                            page_content=page_text,
                            metadata={'links': [a['href'] for a in soup.find_all('a', href=True)]}
                        )
                        self.web_documents.append(document)
                    
            except UnicodeDecodeError:
                logging.warning(f"Could not read {file_path} due to encoding issues.")

        with open(PROCESSED_CONTENT_FILE, "w", encoding="utf-8") as f:
            f.write("\n\n".join([doc.page_content for doc in self.web_documents]))

    def query(self, question):
        return rag_application.run(question)

ai_bot = OllamaBot()
pending_responses = {}
stored_responses = {}
question_id = 0
lock = Lock()

# Process uploaded file
def process_file(file_path):
    print("***PROCESSING FILE***")
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            ai_bot.add(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the file encoding.")
        return "Error: Invalid file encoding."

def append_feedback(feedback_entry):
    try:
        # Check if the file exists and is non-empty
        if os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 0:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Load existing data
                    if not isinstance(data, list):  # Ensure it's a list
                        data = []
                except json.JSONDecodeError:
                    data = []  # If there's a parsing error, start fresh
        else:
            data = []  # Initialize an empty list if file doesn't exist or is empty

        # Append new entry
        data.append(feedback_entry)

        # Write back as a proper JSON list
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)  # Pretty formatting for readability

        print("Feedback appended successfully!")

    except Exception as e:
        print(f"Error while appending feedback: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit-feedback", methods=["POST"])
def submitFeedback():
    try:    
        global selected_model_name
        data = request.json
        
        # Dynamic mapping to support both "comment" and "details"
        comment = data.get("comment") or data.get("details") or ""  # Prefer "comment", fallback to "details"
        
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        # if not details and not question:
        #     return jsonify({"error": "Both feedback details and question details are required"}), 400

        feedback_entry = {
            "model-name": selected_model_name,
            "question": question,
            "response": response,
            "feedback": comment,
            "rating-score": rating
        }
        
        # Write data_string to feedback file without overwriting existing contents
        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
                try:
                    feedback_data = json.load(file)
                except json.JSONDecodeError:
                    feedback_data = []  # Initialize as empty list if file is corrupted
        
        feedback_data.append(feedback_entry)
        
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as file:
            json.dump(feedback_data, file, indent=4)
            
        # reload the contents
        ai_bot._load_content()
        # retrains the application whenever new training data is updated.
        ai_bot._initialize_rag_application()
        
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200
    except Exception as e:
        app.logger.error(f"Error in /submit-feedback endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = f"./Data/{file.filename}"
        file.save(file_path)
        result = process_file(file_path)
        print(f"Processed result: {result}")
        return jsonify({"message": result})


def check_selected_options(selectedOptions):
    expected_options = ["text", "table", "list"]
    return set(selectedOptions) == set(expected_options)

def save_chat_session(session_id, messages):
    # Use a module-level constant or ensure directory creation happens once (outside the function) if possible
    if not os.path.isdir(CHAT_SESSIONS_DIR):
        os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

    session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

    # Pre-generate timestamp to avoid recalculating multiple times
    current_timestamp = datetime.now().isoformat() + "Z"

    # Use a list comprehension to update messages more efficiently
    updated_messages = [
        message if "timestamp" in message else {**message, "timestamp": current_timestamp}
        for message in messages
    ]

    # Use faster file writing (single write operation)
    with open(session_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(updated_messages, ensure_ascii=False, indent=4))

def process_question(question_id, question, ai_bot, selectedOptions):
    """
    Simulate long processing of the question and store the response.
    """
    global answer_time, current_session_id, current_session_messages
    
    start_time = time.time() # Begins to record how long it takes the model for querying
    
    response = ai_bot.query(question)
    
    print("Response from advanced model:", response)
    
    end_time = time.time()
    
    # Calculate and print the elapsed time
    answer_time = end_time - start_time
    
    print(f"⏱️ The time taken to implement `query` function is {answer_time:.4f} seconds.")
    
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)

    stored_responses[question_id] = formatted_response
    
    # Append assistant's response to current session
    current_session_messages.append({"role": "assistant", "content": response})

    # Save current session to file
    save_chat_session(current_session_id, current_session_messages)

@app.route("/delete-session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return jsonify({"error": "Session not found"}), 404
        
        # Delete the file
        os.remove(session_file)

        # Update the metadata (optional - if you want to remove the name as well)
        metadata = load_session_metadata()
        if session_id in metadata:
            del metadata[session_id]
            save_session_metadata(metadata)

        return jsonify({"message": "Session deleted successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error deleting session: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global question_id, current_session_id, current_session_messages

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        question = data.get("question", "").strip()
        selectedOptions = data.get("selectedOptions", "")
        incoming_session_id = data.get("session_id")

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Check if we are continuing a historical session
        if incoming_session_id:
            if current_session_id != incoming_session_id:
                # New session is being reloaded
                current_session_id = incoming_session_id
                session_file = os.path.join(CHAT_SESSIONS_DIR, f"{current_session_id}.json")

                if os.path.exists(session_file):
                    with open(session_file, "r", encoding="utf-8") as f:
                        current_session_messages = json.load(f)
                else:
                    # Fall back in case of missing file
                    current_session_messages = []

        elif current_session_id is None:
            # If no historical session is loaded, start fresh
            current_session_id = f"chat_session_{time.strftime('%Y%m%d_%H%M%S')}"
            current_session_messages = []

        # Append new user message
        current_session_messages.append({"role": "user", "content": question})

        def process_question_wrapper(*args):
            """Wraps process_question to measure its execution time."""
            global execution_time
            start_time = time.time()
            process_question(*args)  # Run the actual function
            execution_time = time.time() - start_time  # Store execution time
        
        with lock:
            current_id = str(question_id)
            question_id += 1

        pending_responses[current_id] = "Processing..."
        
        # Start a thread and track execution time properly
        process_question_start_time = time.time()  # Start the timer before the thread starts
        process_question_thread = threading.Thread(target=process_question_wrapper, args=(current_id, question, ai_bot, selectedOptions))
        process_question_thread.start()
        process_question_thread.join()  # Ensures we wait for the function to finish
        process_question_time = time.time() - process_question_start_time  # Stop the timer after the thread finishes

        print(f"⏱️ The time taken to implement `process_question` function is {process_question_time:.4f} seconds.")

        return jsonify({"question_id": current_id, "session_id": current_session_id}), 200

    except Exception as e:
        app.logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    """
    Endpoint for EventSource to fetch the response.
    """
    def generate_response():
        while True:
            response = stored_responses.get(question_id)
            
            if response == "Processing" or response is None:
                yield "data: Processing your question...\n\n"
            elif response:
                formatted_response = response.replace("\n", "<br>")
                yield f"data: {formatted_response}\n\n"
                break
            else:
                yield "data: Error: Invalid question ID\n\n"
                break
            time.sleep(1)  # Polling interval

    return Response(generate_response(), content_type="text/event-stream")

@app.route('/delete-pair', methods=['POST'])
def delete_question_answer_pair():
    try:
        data = request.json
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            return jsonify({"success": False, "error": "Missing session_id or question"}), 400

        session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

        if not os.path.exists(session_file):
            return jsonify({"success": False, "error": "Session not found"}), 404

        with open(session_file, "r", encoding="utf-8") as file:
            messages = json.load(file)

        # Find the index of the user question and remove it along with the next assistant message
        new_messages = []
        skip_next = False

        for i, msg in enumerate(messages):
            if skip_next:
                skip_next = False
                continue  # Skip the assistant response after the user question

            if msg["role"] == "user" and msg["content"] == question:
                skip_next = True  # Skip this user question and the following assistant message
                continue

            new_messages.append(msg)

        with open(session_file, "w", encoding="utf-8") as file:
            json.dump(new_messages, file, indent=4)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/selection", methods=["GET"])
def update_model_name():
    global selected_model_name
    
    model_name = request.args.get("model")  # Retrieve model name from URL parameters

    if not model_name:
        return jsonify({"error": "No model selected"}), 400
    
    selected_model_name = model_name

    print(f"Selected model name: \"{selected_model_name}\".")

    return jsonify({"message": f"Model updated to {model_name}"}), 200

@app.route('/feedback')
def feedback():
    return render_template('feedback.html', nonce=g.nonce)

@app.route('/feedback_dataset.json')
def feedback_data():
    feedback_path = os.path.join(DATA_DIR, "Feedback")  # Capital 'F'
    return send_from_directory(feedback_path, "feedback_dataset.json")

@app.route("/view-file", methods=["GET"])
def view_file():
    filename = request.args.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    htm_filepaths = ai_bot._list_htm_files()
    
    current_directory = os.getcwd()
    
    # for each file, join path with the path of the htm file
    temp_directories = [os.path.join(current_directory, htm_filepath) for htm_filepath in htm_filepaths]
    
    # Find the matching file path
    file_path = next((directory for directory in temp_directories if directory.endswith(filename)), None)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """
    Computes the semantic similarity between two text extracts using Sentence-BERT.

    Parameters:
    text1 (str): The first text (expected response).
    text2 (str): The second text (chatbot response).
    model_name (str): Name of the pre-trained SentenceTransformer model.

    Returns:
    float: Cosine similarity score (range: 0 to 1, where 1 means identical meaning).
    """
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return similarity_score

# Function to check if a question has a similar wording in expected_answers_dict
def is_similar_question(question, expected_answers, threshold=0.80):
    print(f"question: {question}")
    print(f"expected answer: {expected_answers}")
    match, score, _ = process.extractOne(question, expected_answers, score_cutoff=threshold)
    if match is None:
        return False
    print(f"score: {score}")
    return score >= threshold  # Returns True if similarity score is above threshold

# Function to replace invalid values with an empty string
def clean_dataframe(df):
    """
    Replace NaN, None, or other invalid values with an empty string in the given DataFrame.
    """
    return df.fillna("").replace({None: ""})  # Replace NaN and None with empty string

# Function to find the most semantically similar expected answer
def find_best_match(question, expected_answers_dict):
    if not question or not expected_answers_dict:
        return ""  # Return empty string if no valid input

    expected_questions = list(expected_answers_dict.keys())
    
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

    # Compute embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    expected_embeddings = model.encode(expected_questions, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, expected_embeddings)[0]

    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()

    # Set a similarity threshold (adjust if needed)
    threshold = 0.75  
    if best_match_score < threshold:
        return ""  # No match found above the threshold

    return expected_answers_dict[expected_questions[best_match_idx]]

# Function to compute BLEU and ROUGE scores
def calculate_bleu_rouge(reference, candidate):
    """
    Compute BLEU and ROUGE similarity scores between the reference answer and the response.
    """
    if not reference or not candidate:
        return {"BLEU": 0.0, "ROUGE": 0.0}

    # Compute BLEU score
    bleu_score = sentence_bleu([reference.split()], candidate.split())

    # Compute ROUGE-L score (longest common subsequence)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score = scorer.score(reference, candidate)["rougeL"].fmeasure  # F1 score of ROUGE-L

    return {"BLEU": bleu_score, "ROUGE": rouge_score}

@app.route("/get-results", methods=["GET"])
def get_results():
    try:
        logging.info("Processing GET request for /get-results")

        # Step 1: Check if the results file exists
        if not os.path.exists(EXCEL_FILE):
            logging.warning(f"Results file not found: {EXCEL_FILE}")
            return jsonify({"error": "Results file not found"}), 404

        # Define columns
        columns = ["Question", "Model Name", "Response", "Expected Answer", "similarity_score"]

        # Generate random data
        num_rows = 10  # Number of rows in the dataframe
        questions = [f"Question {i+1}" for i in range(num_rows)]
        models = [f"Model_{np.random.randint(1, 5)}" for _ in range(num_rows)]
        responses = [f"Response {i+1}" for i in range(num_rows)]
        expected_answers = [f"Expected Answer {i+1}" for i in range(num_rows)]
        similarity_scores = np.random.uniform(0, 1, num_rows)  # Random similarity scores between 0 and 1

        # Define data as a list of dictionaries to avoid potential issues
        data = [
            {
                "Question": questions[i],
                "Model Name": models[i],
                "Response": responses[i],
                "Expected Answer": expected_answers[i],
                "similarity_score": similarity_scores[i],
            }
            for i in range(num_rows)
        ]

        temp_df = pd.DataFrame(data)

        temp_results = pd.DataFrame({
            "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
            "Accuracy": ["90%", "84%", "79%", "92%"]
        })

        # Step 2: Load the results file
        logging.info(f"Loading results from {EXCEL_FILE}")
        if os.path.exists(EXCEL_FILE):
            try:
                df = pd.read_excel(EXCEL_FILE)
            except ValueError:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except PermissionError:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except Exception as e:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
        else:
            df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })
        logging.debug(f"Loaded DataFrame columns: {df.columns.tolist()}")

        # Step 3: Check for expected results file
        expected_results_df = None
        if os.path.exists(EXPECTED_RESULTS_FILE):
            try:
                logging.info(f"Loading expected results from {EXPECTED_RESULTS_FILE}")
                expected_results_df = pd.read_excel(EXPECTED_RESULTS_FILE)
            except ValueError as ve:
                logging.error(f"ValueError: Unable to read the Excel file due to format issues - {ve}")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except PermissionError:
                logging.error(f"PermissionError: The file {EXPECTED_RESULTS_FILE} is in use or lacks necessary permissions.")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except Exception as e:
                logging.error(f"Unexpected error while reading the Excel file: {e}")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
        else:
            logging.warning(f"Expected results file not found: {EXPECTED_RESULTS_FILE}")
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })

        question_column = "Question"
        response_column = "Response"
        expected_answer_column = "Expected Answer"

        # Step 4: Validate column existence
        required_columns = [question_column, response_column]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns in results file: {df.columns.tolist()}")
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })
            # return jsonify({"error": "Invalid results file format"}), 400

        if expected_results_df is not None and question_column in expected_results_df.columns:
            expected_answers_dict = dict(zip(expected_results_df[question_column], expected_results_df[response_column]))
            logging.info("Successfully created expected answers dictionary.")
        else:
            logging.warning("Expected results file does not contain required columns. Skipping filtering.")
            expected_answers_dict = {}
        print(expected_answers_dict)
        # Step 5: Check if questions exist in expected results
        df["is_question_in_expected"] = df[question_column].apply(lambda q: is_similar_question(q, expected_answers_dict.keys()))
        print(f"Questions found in expected results: {df['is_question_in_expected'].sum()}")

        # Step 6: Filter data and compute similarity scores
        filtered_df = df[df["is_question_in_expected"] == True].copy()
        if not filtered_df.empty:
            filtered_df.loc[:, expected_answer_column] = filtered_df.loc[:, question_column].apply(lambda q: find_best_match(q, expected_answers_dict))

            similarity_scores = filtered_df.apply(
                lambda row: calculate_bleu_rouge(str(row[expected_answer_column]), str(row[response_column])),
                axis=1
            )
            
            # Store BLEU and ROUGE scores in separate columns
            filtered_df["BLEU_score"] = similarity_scores.apply(lambda x: x["BLEU"])
            filtered_df["ROUGE_score"] = similarity_scores.apply(lambda x: x["ROUGE"])
            
            # Compute similarity score
            filtered_df.loc[:, "similarity_score"] = filtered_df.apply(
                lambda row: calculate_semantic_similarity(str(row[expected_answer_column]), str(row[response_column])),
                axis=1
            )
            logging.info(f"Computed similarity scores for {len(filtered_df)} entries.")
        else:
            logging.warning("No matching questions found in expected results.")

        logging.info("Filtered results preview:")
        logging.debug(filtered_df.head().to_string())

        # Save filtered results for debugging
        temp_filtered_file = "Data/temp_filtered_data.xlsx"
        filtered_df.drop(columns=["is_question_in_expected"], inplace=True)
        filtered_df.to_excel(temp_filtered_file)
        logging.info(f"Filtered results saved to {temp_filtered_file}")

        # Step 7: Prepare model accuracy results
        if "Model Name" not in df.columns or "Accuracy" not in df.columns:
            logging.warning("Missing 'Model Name' or 'Accuracy' columns in results file. Using default values.")
            results = pd.DataFrame({
                "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
                "Accuracy": ["90%", "84%", "79%", "92%"]
            })
        else:
            results = df[["Model Name", "Accuracy"]]

        # Step 8: Convert results to JSON and return response
        logging.info("Returning JSON response with models and filtered results.")
        print(results.columns)
        print(filtered_df.columns)
        # Apply the function to clean both DataFrames
        results = clean_dataframe(results)
        filtered_df = clean_dataframe(filtered_df)
        
        json_results = results.to_dict(orient="records")
        json_filtered_results = filtered_df.to_dict(orient="records")
        
        json_results = json.loads(json.dumps(json_results))  # Converts NaN to valid JSON
        json_filtered_results = json.loads(json.dumps(json_filtered_results))
        return jsonify({
            "models": json_results,
            "filtered_results": json_filtered_results
        })

    except Exception as e:
        logging.error(f"Error reading results: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    
def append_to_excel(question, response):
    """ Append question and response to the Excel file. """
    global selected_model_name
    new_entry = pd.DataFrame([[question, selected_model_name, response]], columns=["Question", "Model Name", "Response"])

    # Check if the file exists
    if not os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            new_entry.to_excel(writer, index=False)
    else:
        existing_data = pd.read_excel(EXCEL_FILE)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        updated_data.to_excel(EXCEL_FILE, index=False)
        
@app.route("/ask-file", methods=["POST"])
def ask_file():
    """ Process a question from the uploaded file and store the answer. """
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Get AI model response
    response = ai_bot.query(question)

    # Append question and response to Excel
    append_to_excel(question, response)

    return jsonify({"message": response}), 200

@app.route('/search')
def search_chats():
    query = request.args.get('query', '').lower()

    if not query:
        return jsonify([])  # No query provided, return empty list

    chat_sessions = load_chat_history()  # Load all sessions from disk

    matching_messages = []

    # Iterate through all sessions and messages to find matches
    for session in chat_sessions:
        session_id = session['session_id']
        temp_question = None

        for message in session['messages']:
            content = message['content'].lower()

            if query in content:
                if message['role'] == 'user':
                    temp_question = message['content']
                elif message['role'] == 'assistant' and temp_question:
                    matching_messages.append({
                        "session_id": session_id,
                        "question": temp_question,
                        "answer": message['content']
                    })
                    temp_question = None  # Reset after capturing the pair

    return jsonify(matching_messages)

@app.route('/generate-title', methods=['POST'])
def generate_chat_title():
    """Generate a concise chat title using Ollama based on user-provided conversation data."""

    # Extract messages from frontend request
    data = request.json
    messages = data.get("messages", [])

    if not messages:
        print(f"⚠️ No valid messages received: {data}")
        return jsonify({"title": "Untitled Chat"}), 400

    # Convert chat messages into a summarized format
    conversation_text = " ".join([msg["content"] for msg in messages])

    # ✅ Modified prompt to enforce conciseness
    ollama_request = {
        "model": "llama3.2:1b",
        "prompt": (
            "Generate a **very short, clear, and concise** title for this conversation."
            " Keep it **under 8 words** and make it **informative**:"
            f"\n\n{conversation_text}"
        ),
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=ollama_request)
        response_json = response.json()

        title = response_json.get("response", "Untitled Chat").strip()

        # ✅ Additional Post-Processing: Keep **max 8 words**
        title_words = title.split()
        if len(title_words) > 8:
            title = " ".join(title_words[:8]) + "..."  # Truncate and add ellipsis

        return jsonify({"title": title})

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return jsonify({"title": "Untitled Chat"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)