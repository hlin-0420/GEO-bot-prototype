import json
import os

from flask import Blueprint, jsonify, request, send_from_directory

from app import config

feedback_routes = Blueprint("feedback", __name__)


def get_ai_bot():
    from app.services.ollama_bot import get_bot
    return get_bot()


@feedback_routes.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json(silent=True) or {}
        comment = data.get("comment") or data.get("details") or ""
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        feedback_entry = {
            "model-name": config.selected_model_name,
            "session-id": data.get("session_id"),
            "question-number": data.get("question_number"),
            "question": question,
            "response": response,
            "feedback": comment,
            "rating-score": rating,
        }

        feedback_data = []
        if os.path.exists(config.FEEDBACK_FILE):
            with open(config.FEEDBACK_FILE, "r", encoding="utf-8") as file:
                try:
                    feedback_data = json.load(file)
                except json.JSONDecodeError:
                    feedback_data = []

        feedback_data.append(feedback_entry)
        os.makedirs(os.path.dirname(config.FEEDBACK_FILE), exist_ok=True)
        with open(config.FEEDBACK_FILE, "w", encoding="utf-8") as file:
            json.dump(feedback_data, file, indent=4)

        get_ai_bot().refresh()
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200

    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500


@feedback_routes.route("/feedback_dataset.json", methods=["GET"])
def feedback_data():
    return send_from_directory(
        os.path.dirname(config.FEEDBACK_FILE),
        os.path.basename(config.FEEDBACK_FILE),
    )
