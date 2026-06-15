import os

# Get absolute path to the app/ directory (where config.py is)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root (offline_app/)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

# Data Directory at the root level
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Session Directories
CHAT_SESSIONS_DIR = os.path.join(DATA_DIR, "user_sessions", "ChatSessions")
SESSION_METADATA_FILE = os.path.join(DATA_DIR, "user_sessions", "session_metadata.json")
TIMED_RESPONSES_FILE = os.path.join(DATA_DIR, "user_sessions", "timed_responses.json")

# Feedback File
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback", "feedback_dataset.json")

# Evaluation Files
EXCEL_FILE = os.path.join(DATA_DIR, "evaluation", "query_responses.xlsx")
EXPECTED_RESULTS_FILE = os.path.join(DATA_DIR, "evaluation", "expected_query_responses.xlsx")

# Model Files
PROMPT_VISUALISATION_FILE = os.path.join(DATA_DIR, "model_files", "prompt_visualisation.txt")
PROCESSED_CONTENT_FILE = os.path.join(DATA_DIR, "model_files", "processed_content.txt")
UPLOADED_FILE = os.path.join(DATA_DIR, "model_files", "uploaded_document.txt")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "model_files", "faiss_index")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

# Model Settings
DEFAULT_MODEL_NAME = "llama3.2:latest"
MODEL_OPTIONS = [
    {"value": "deepseek1.5", "label": "DeepSeek 1.5"},
    {"value": "llama3.2:latest", "label": "Llama 3.2"},
    {"value": "tinyllama:latest", "label": "Tiny Llama"},
    {"value": "gemma3:1b", "label": "Gemma 3"},
    {"value": "openai", "label": "OpenAI"},
]
VALID_MODEL_NAMES = {model["value"] for model in MODEL_OPTIONS}

# Upload Settings
ALLOWED_UPLOAD_EXTENSIONS = {"csv", "htm", "html", "json", "md", "txt"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Optional: Flask Settings
DEBUG_MODE = True
PORT = 5000

# Active model setting
selected_model_name = DEFAULT_MODEL_NAME
valid_model_names = VALID_MODEL_NAMES
