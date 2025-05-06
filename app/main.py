print("[DEBUG] main.py loaded")

from flask import Flask
print("[DEBUG] Flask imported")

from app.routes.api import api_blueprint
from app.routes.ui import ui_blueprint
import os

# Define root-relative template path
TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
STATIC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

app = Flask(__name__, template_folder=TEMPLATE_PATH, static_folder=STATIC_PATH)
print("[DEBUG] Flask app instance created")

# Register blueprints
app.register_blueprint(api_blueprint)
app.register_blueprint(ui_blueprint)

# If you are loading any models or databases here, add timers:
import time

print("[DEBUG] Starting heavy component loading")
start = time.time()

# Example:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Simulated delay
# time.sleep(3)

print(f"[DEBUG] Component load complete in {time.time() - start:.2f} seconds")
print("[DEBUG] Routes registered")
