from flask import Blueprint, request, jsonify
import os, json
from app.config import TIMED_RESPONSES_FILE

timing_routes = Blueprint('timing', __name__)

@timing_routes.route("/store-response-time", methods=["POST"])
def store_response_time():
    try:
        data = request.json
        question_number = str(data.get("questionNumber"))
        question_text = data.get("question")
        duration = float(data.get("duration"))
        timestamp = data.get("timestamp")

        if not question_number or not question_text or duration is None or not timestamp:
            return jsonify({"error": "Invalid data"}), 400

        response_times = []
        if os.path.exists(TIMED_RESPONSES_FILE):
            with open(TIMED_RESPONSES_FILE, "r", encoding="utf-8") as file:
                try:
                    response_times = json.load(file)
                except json.JSONDecodeError:
                    response_times = []

        response_times.append({
            "question_number": question_number,
            "question": question_text,
            "response_time": duration,
            "timestamp": timestamp
        })

        with open(TIMED_RESPONSES_FILE, "w", encoding="utf-8") as file:
            json.dump(response_times, file, indent=4)

        return jsonify({"message": "Response time stored successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500