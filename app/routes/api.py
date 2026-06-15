import json
import os
import re
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, request

from app import config, state
from app.services.ollama_bot import get_bot
from app.services.question_handler import process_question

api_blueprint = Blueprint("api", __name__)

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
RESPONSE_TIMEOUT_SECONDS = 300


def _is_valid_session_id(session_id):
    return bool(session_id and SESSION_ID_PATTERN.fullmatch(session_id))


def _new_session_id():
    return f"chat_session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _load_session_messages(session_id):
    if session_id in state.session_messages:
        return state.session_messages[session_id]

    session_file = os.path.join(config.CHAT_SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
    else:
        messages = []

    state.session_messages[session_id] = messages
    return messages


def _format_sse(data):
    lines = str(data).splitlines() or [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _set_selected_model(model_name):
    if not model_name:
        return None

    if model_name not in config.VALID_MODEL_NAMES:
        return jsonify({
            "error": f"Unsupported model '{model_name}'",
            "valid_models": sorted(config.VALID_MODEL_NAMES),
        }), 400

    if config.selected_model_name != model_name:
        config.selected_model_name = model_name
        get_bot(force_refresh=True)

    return None


def _cleanup_old_responses():
    cutoff = time.time() - RESPONSE_TIMEOUT_SECONDS
    expired_question_ids = [
        question_id
        for question_id, created_at in state.pending_response_created_at.items()
        if created_at < cutoff
    ]
    for question_id in expired_question_ids:
        state.clear_pending_response(question_id)



def format_sse_data(message):
    """Format multiline text safely for Server-Sent Events."""
    lines = str(message).splitlines() or [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


@api_blueprint.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    """SSE endpoint used by EventSource to fetch a completed answer."""

    def generate_response():
        start_time = time.monotonic()
        timed_out = False
        try:
            while True:
                with state.lock:
                    response = state.pending_responses.get(question_id)

                if response == state.PROCESSING_STATUS:
                    yield _format_sse("Processing your question...")
                elif response is None:
                    yield _format_sse("Error: Invalid question ID")
                    break
                else:
                    yield _format_sse(response)
                    break

                if time.monotonic() - start_time > RESPONSE_TIMEOUT_SECONDS:
                    timed_out = True
                    yield _format_sse("Error: Timed out waiting for a response")
                    break

                time.sleep(1)
        finally:
            with state.lock:
                response = state.pending_responses.get(question_id)
                if timed_out or response != state.PROCESSING_STATUS:
                    state.clear_pending_response(question_id)

    return Response(generate_response(), content_type="text/event-stream")


@api_blueprint.route("/selection", methods=["GET"])
def update_model_name():
    model_name = request.args.get("model")

    if not model_name:
        return jsonify({"error": "No model selected"}), 400

    validation_error = _set_selected_model(model_name)
    if validation_error:
        return validation_error

    return jsonify({"message": f"Model updated to {model_name}"}), 200


@api_blueprint.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        question = data.get("question", "").strip()
        selected_options = data.get("selectedOptions", "")
        incoming_session_id = data.get("session_id")
        model_name = data.get("model")

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        validation_error = _set_selected_model(model_name)
        if validation_error:
            return validation_error

        if incoming_session_id:
            if not _is_valid_session_id(incoming_session_id):
                return jsonify({"error": "Invalid session ID"}), 400
            session_id = incoming_session_id
        else:
            session_id = _new_session_id()

        with state.lock:
            _cleanup_old_responses()
            question_id = uuid.uuid4().hex
            session_messages = _load_session_messages(session_id)
            session_messages.append({"role": "user", "content": question})
            state.pending_responses[question_id] = state.PROCESSING_STATUS
            state.pending_response_created_at[question_id] = time.time()

        def process_question_wrapper():
            try:
                start_time = time.time()
                bot = get_bot()
                process_question(
                    question_id,
                    question,
                    bot,
                    session_id,
                    session_messages,
                    state.pending_responses,
                    response_lock=state.lock,
                )
                state.execution_time = time.time() - start_time
            except Exception as exc:
                with state.lock:
                    if question_id in state.pending_responses:
                        state.pending_responses[question_id] = f"Error: {exc}"

        thread = threading.Thread(target=process_question_wrapper, daemon=True)
        thread.start()

        return jsonify({
            "question_id": question_id,
            "session_id": session_id,
        }), 200

    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500
