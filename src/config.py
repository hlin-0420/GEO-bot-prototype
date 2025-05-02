# src/config.py
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CYPHER_FILE_PATH = os.getenv("CYPHER_FILE_PATH")
HTM_FILE_PATH = os.getenv("HTM_FILE_PATH")
HTM_FOLDER_PATH = os.getenv("HTM_FOLDER_PATH")