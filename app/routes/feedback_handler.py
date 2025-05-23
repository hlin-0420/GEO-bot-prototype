from flask import Blueprint, request, jsonify, send_from_directory
import os, json
from app.config import FEEDBACK_FILE, selected_model_name, DATA_DIR

feedback_routes = Blueprint('feedback', __name__)
def get_ai_bot():
    from app.services.ollama_bot import get_bot
    return get_bot()

@feedback_routes.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.json
        comment = data.get("comment") or data.get("details") or ""
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        feedback_entry = {
            "model-name": selected_model_name,
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

        get_ai_bot().refresh()
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200

    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500

@feedback_routes.route('/feedback_dataset.json', methods=["GET"])
def feedback_data():
    return send_from_directory(DATA_DIR, "feedback_dataset.json")
