# app/routes/api.py
from flask import Blueprint, request, jsonify
import os, json, time, threading
from app.services.ollama_bot import ai_bot
from app.services.session_manager import save_chat_session
from app.services.processor import process_question  # you may move this to its own service file
from app import state  # import shared state

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        question = data.get("question", "").strip()
        selectedOptions = data.get("selectedOptions", "")
        incoming_session_id = data.get("session_id")

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Manage session ID and messages
        if incoming_session_id:
            if state.current_session_id != incoming_session_id:
                state.current_session_id = incoming_session_id
                session_file = os.path.join("Data/user_sessions/ChatSessions", f"{state.current_session_id}.json")

                if os.path.exists(session_file):
                    with open(session_file, "r", encoding="utf-8") as f:
                        state.current_session_messages = json.load(f)
                else:
                    state.current_session_messages = []
        elif state.current_session_id is None:
            state.current_session_id = f"chat_session_{time.strftime('%Y%m%d_%H%M%S')}"
            state.current_session_messages = []

        state.current_session_messages.append({"role": "user", "content": question})

        def process_question_wrapper(*args):
            start_time = time.time()
            process_question(*args)
            state.execution_time = time.time() - start_time

        with state.lock:
            current_id = str(state.question_id)
            state.question_id += 1

        state.pending_responses[current_id] = "Processing..."

        process_question_start_time = time.time()
        thread = threading.Thread(
            target=process_question_wrapper,
            args=(current_id, question, ai_bot, selectedOptions)
        )
        thread.start()
        thread.join()
        process_time = time.time() - process_question_start_time

        print(f"⏱️ Process time: {process_time:.4f} seconds.")

        return jsonify({
            "question_id": current_id,
            "session_id": state.current_session_id
        }), 200

    except Exception as e:
        print(f"❌ Error in /ask: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500