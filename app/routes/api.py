# app/routes/api.py
from flask import Blueprint, request, jsonify, Response
import os, json, time, threading
from app.services.ollama_bot import get_bot
from app.services.question_handler import process_question
from app import state  # import shared state

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    """
    SSE endpoint for EventSource to fetch the response.
    """
    def generate_response():
        while True:
            response = state.pending_responses.get(question_id)

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

@api_blueprint.route("/selection", methods=["GET"])
def update_model_name():
    model_name = request.args.get("model")  # Retrieve model name from URL parameters

    if not model_name:
        return jsonify({"error": "No model selected"}), 400

    # Update the global variable (if it's mutable; else use global keyword and redefine)
    selected_model_name = model_name
    print(f"Selected model name: \"{selected_model_name}\".")

    return jsonify({"message": f"Model updated to {model_name}"}), 200

@api_blueprint.route("/ask", methods=["POST"])
def ask():
    try:
        print("📥 Received a POST request to /ask")
        data = request.json
        if not data:
            print("⚠️ No JSON payload received")
            return jsonify({"error": "No JSON payload received"}), 400

        question = data.get("question", "").strip()
        selectedOptions = data.get("selectedOptions", "")
        incoming_session_id = data.get("session_id")

        if not question:
            print("⚠️ Question is empty")
            return jsonify({"error": "Question cannot be empty"}), 400

        print(f"💬 User question: {question}")
        print(f"🧾 Selected options: {selectedOptions}")
        print(f"🔑 Incoming session ID: {incoming_session_id}")

        # Manage session ID and messages
        if incoming_session_id:
            if state.current_session_id != incoming_session_id:
                print("🔄 Session ID changed, loading new session")
                state.current_session_id = incoming_session_id
                session_file = os.path.join("Data/user_sessions/ChatSessions", f"{state.current_session_id}.json")

                if os.path.exists(session_file):
                    print("📂 Loading existing session file")
                    with open(session_file, "r", encoding="utf-8") as f:
                        state.current_session_messages = json.load(f)
                else:
                    print("📁 No session file found, starting new session")
                    state.current_session_messages = []
        elif state.current_session_id is None:
            state.current_session_id = f"chat_session_{time.strftime('%Y%m%d_%H%M%S')}"
            print(f"🆕 Created new session ID: {state.current_session_id}")
            state.current_session_messages = []

        state.current_session_messages.append({"role": "user", "content": question})

        def process_question_wrapper(*args):
            try:
                start_time = time.time()
                print("🚀 Starting question processing thread")
                question_id, question, selectedOptions = args
                print("📦 Calling get_bot()...")
                bot = get_bot()
                print("✅ get_bot() returned.")
                process_question(question_id, question, bot, state.current_session_id, state.current_session_messages, state.stored_responses)
                state.execution_time = time.time() - start_time
                print(f"✅ Finished processing question in {state.execution_time:.4f} seconds")
            except Exception as e:
                print(f"❌ Error in thread: {str(e)}")

        with state.lock:
            current_id = str(state.question_id)
            state.question_id += 1

        state.pending_responses[current_id] = "Processing..."

        print(f"🆔 Assigned question ID: {current_id}")
        print("🧠 Spawning background thread for processing...")

        process_question_start_time = time.time()
        thread = threading.Thread(
            target=process_question_wrapper,
            args=(current_id, question, selectedOptions)
        )
        thread.start()
        process_time = time.time() - process_question_start_time

        print(f"⏱️ Returned response immediately after starting thread in {process_time:.4f} seconds")

        return jsonify({
            "question_id": current_id,
            "session_id": state.current_session_id
        }), 200

    except Exception as e:
        print(f"❌ Error in /ask: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500