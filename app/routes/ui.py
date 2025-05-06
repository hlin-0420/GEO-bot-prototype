from flask import Blueprint, render_template, request, jsonify, send_from_directory
from app.services.session_manager import load_chat_history, load_session_metadata, save_session_metadata
import os, json
from app.config import DATA_DIR

ui_blueprint = Blueprint('ui', __name__)

@ui_blueprint.route('/')
def index():
    return render_template('index.html')

@ui_blueprint.route('/feedback')
def feedback():
    return render_template('feedback.html')

@ui_blueprint.route('/feedback_dataset.json')
def feedback_data():
    return send_from_directory(DATA_DIR, "feedback_dataset.json")

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

        return jsonify({
            "session_id": session_id,
            "messages": messages
        }), 200
    except Exception as e:
        return jsonify({"error": "Internal Server Error"}), 500

@ui_blueprint.route("/rename-session", methods=["POST"])
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

@ui_blueprint.route("/clear-chat-sessions", methods=["POST"])
def clear_chat_sessions():
    try:
        session_dir = os.path.join(DATA_DIR, "user_sessions", "ChatSessions")
        if os.path.exists(session_dir):
            for filename in os.listdir(session_dir):
                file_path = os.path.join(session_dir, filename)
                os.remove(file_path)
        save_session_metadata({})
        return jsonify({"success": True, "message": "All chat sessions cleared."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500