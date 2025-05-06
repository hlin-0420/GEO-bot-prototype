import os
import json
from app.config import FEEDBACK_FILE

def load_feedback_dataset():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)  # Initialize an empty list
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)