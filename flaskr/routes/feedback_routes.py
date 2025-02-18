from flask import Blueprint, request, jsonify, render_template
from ..data_processing.feedback import append_feedback
from ..data_processing.data_loading import load_feedback_dataset
from ..models.ollama_bot import OllamaBot
import json

feedback_bp = Blueprint('feedback', __name__)

ai_bot = OllamaBot()

@feedback_bp.route("/submit-feedback", methods=["POST"])
def submitFeedback():
    try:    
        data = request.json
        details = data.get("details")
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        if not details and not question:
            return jsonify({"error": "Both feedback details and question details are required"}), 400

        feedback_entry = {
            "model-name": "llama3.2:latest",
            "question": question,
            "response": response,
            "feedback": details,
            "rating-score": rating
        }

        append_feedback(feedback_entry)
        
        feedback_data = load_feedback_dataset()
        
        for entry in feedback_data:
            ai_bot.update_training(json.dumps(entry, indent=4), True)  
        
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@feedback_bp.route("/feedback")
def feedback():
    return render_template('feedback.html')
