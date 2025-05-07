from flask import Blueprint, render_template, jsonify, request
import os
import json
from app.config import DATA_DIR, FEEDBACK_FILE
from app.services.session_manager import load_chat_history
from app.utils.file_helpers import process_file

ui_blueprint = Blueprint('ui', __name__)

def get_ai_bot():
    from app.services.ollama_bot import get_bot
    return get_bot()

@ui_blueprint.route('/')
def index():
    return render_template('index.html')

@ui_blueprint.route('/feedback')
def feedback():
    return render_template('feedback.html')

@ui_blueprint.route('/chathistory')
def chathistory():
    return render_template("chathistory.html")

@ui_blueprint.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.json
        comment = data.get("comment") or data.get("details") or ""
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        feedback_entry = {
            "question": question,
            "response": response,
            "feedback": comment,
            "rating-score": rating
        }

        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
                try:
                    feedback_data = json.load(file)
                except json.JSONDecodeError:
                    feedback_data = []

        feedback_data.append(feedback_entry)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as file:
            json.dump(feedback_data, file, indent=4)

        get_ai_bot().refresh()  # Safe call now
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200

    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500

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

@ui_blueprint.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    file_path = f"./Data/{file.filename}"
    file.save(file_path)
    result = process_file(file_path)
    return jsonify({"message": result})