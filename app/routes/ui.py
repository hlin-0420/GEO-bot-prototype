from flask import Blueprint, render_template, request, jsonify
import os, json
from app.config import DATA_DIR, TIMED_RESPONSES_FILE, FEEDBACK_FILE, CHAT_SESSIONS_DIR, selected_model_name
from app.services.session_manager import load_chat_history, load_session_metadata, save_session_metadata
from app.services.ollama_bot import get_bot
from app.utils.file_helpers import process_file

ui_blueprint = Blueprint('ui', __name__)
ai_bot = get_bot()

@ui_blueprint.route('/')
def index():
    return render_template('index.html')

@ui_blueprint.route('/feedback')
def feedback():
    return render_template('feedback.html')

@ui_blueprint.route('/chathistory')
def chathistory():
    return render_template("chathistory.html")

@ui_blueprint.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify(load_chat_history())

@ui_blueprint.route('/chat-history/<session_id>', methods=['GET'])
def get_single_chat_session(session_id):
    session_file = os.path.join(DATA_DIR, "user_sessions", "ChatSessions", f"{session_id}.json")
    if not os.path.exists(session_file):
        return jsonify({"error": "Session not found"}), 404

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return jsonify({"session_id": session_id, "messages": messages}), 200
    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500