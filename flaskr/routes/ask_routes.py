from flask import Blueprint, request, jsonify, Response
import threading
import time
from ..models.ollama_bot import OllamaBot
from ..data_processing.file_processing import process_question
from threading import Lock

ask_bp = Blueprint('ask', __name__)

ai_bot = OllamaBot()
lock = Lock()
pending_responses = {}
question_id = 0

@ask_bp.route("/ask", methods=["POST"])
def ask():
    global question_id
    
    print("Asking a question...")

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400
        
        question = data.get("question", "").strip()
        selectedOptions = data.get("selectedOptions", "")

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        with lock:
            current_id = str(question_id)
            question_id += 1

        pending_responses[current_id] = "Processing..."
        threading.Thread(target=process_question, args=(current_id, question, ai_bot, selectedOptions)).start()

        return jsonify({"question_id": current_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ask_bp.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    """
    Endpoint for EventSource to fetch the response.
    """
    def generate_response():
        while True:
            response = pending_responses.get(question_id)
            
            if response == "Processing" or response is None:
                yield "data: Processing your question...\n\n"
            elif response:
                formatted_response = response.replace("\n", "<br>")
                yield f"data: {formatted_response}\n\n"
                break
            else:
                yield "data: Error: Invalid question ID\n\n"
                break
            time.sleep(1)  

    return Response(generate_response(), content_type="text/event-stream")