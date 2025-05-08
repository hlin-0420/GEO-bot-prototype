from flask import Blueprint, request, jsonify
import os
from app.config import DATA_DIR, CHAT_SESSIONS_DIR
from app.services.session_manager import load_session_metadata, save_session_metadata, load_chat_history
import json
import requests

session_routes = Blueprint('session', __name__)

@session_routes.route("/rename-session", methods=["POST"])
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

@session_routes.route("/clear-chat-sessions", methods=["POST"])
def clear_chat_sessions():
    try:
        session_dir = os.path.join(DATA_DIR, "user_sessions", "ChatSessions")
        if os.path.exists(session_dir):
            for filename in os.listdir(session_dir):
                os.remove(os.path.join(session_dir, filename))
        save_session_metadata({})
        return jsonify({"success": True, "message": "All chat sessions cleared."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@session_routes.route("/delete-session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
        if not os.path.exists(session_file):
            return jsonify({"error": "Session not found"}), 404
        os.remove(session_file)
        metadata = load_session_metadata()
        if session_id in metadata:
            del metadata[session_id]
            save_session_metadata(metadata)
        return jsonify({"message": "Session deleted successfully"}), 200
    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500
    
@session_routes.route('/delete-pair', methods=['POST'])
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

        new_messages = []
        skip_next = False

        for msg in messages:
            if skip_next:
                skip_next = False
                continue
            if msg["role"] == "user" and msg["content"] == question:
                skip_next = True
                continue
            new_messages.append(msg)

        with open(session_file, "w", encoding="utf-8") as file:
            json.dump(new_messages, file, indent=4)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@session_routes.route('/search', methods=['GET'])
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

@session_routes.route('/generate-title', methods=['POST'])
def generate_chat_title():
    """Generate a concise chat title using Ollama based on user-provided conversation data."""

    data = request.json
    messages = data.get("messages", [])

    if not messages:
        print(f"⚠️ No valid messages received: {data}")
        return jsonify({"title": "Untitled Chat"}), 400

    conversation_text = " ".join([msg["content"] for msg in messages])

    ollama_request = {
        "model": "llama3.2:latest",
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

        title_words = title.split()
        if len(title_words) > 8:
            title = " ".join(title_words[:8]) + "..."

        return jsonify({"title": title})

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return jsonify({"title": "Untitled Chat"}), 500
