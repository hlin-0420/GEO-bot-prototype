import os, json
from datetime import datetime

from app.config import CHAT_SESSIONS_DIR, SESSION_METADATA_FILE

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
    
def load_session_metadata():
    if os.path.exists(SESSION_METADATA_FILE):
        with open(SESSION_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_session_metadata(metadata):
    with open(SESSION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)