import time

from app.services.session_manager import save_chat_session


def process_question(
    question_id,
    question,
    ai_bot,
    current_session_id,
    current_session_messages,
    response_store,
    response_lock=None,
):
    """
    Process a question with the AI bot, store the completed answer, and persist
    the session transcript.
    """
    start_time = time.time()

    try:
        response = ai_bot.query(question)
    except Exception:
        response = "An error occurred while querying the model."

    if response_lock:
        with response_lock:
            if question_id in response_store:
                response_store[question_id] = response
            current_session_messages.append({"role": "assistant", "content": response})
            messages_snapshot = list(current_session_messages)
    else:
        if question_id in response_store:
            response_store[question_id] = response
        current_session_messages.append({"role": "assistant", "content": response})
        messages_snapshot = list(current_session_messages)

    save_chat_session(current_session_id, messages_snapshot)

    return time.time() - start_time
