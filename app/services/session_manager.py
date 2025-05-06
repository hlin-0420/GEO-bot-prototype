import os, json
from datetime import datetime

SESSION_DIR = "Data/user_sessions/ChatSessions"
SESSION_METADATA_FILE = "Data/user_sessions/session_metadata.json"

from app.config import CHAT_SESSIONS_DIR

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