from flask import Blueprint, request, jsonify
import os
from app.config import DATA_DIR, CHAT_SESSIONS_DIR
from app.services.session_manager import load_session_metadata, save_session_metadata

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
