import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")