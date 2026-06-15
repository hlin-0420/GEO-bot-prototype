from threading import Lock

PROCESSING_STATUS = "Processing"

session_messages = {}
pending_responses = {}
pending_response_created_at = {}
execution_time = 0.0
lock = Lock()


def clear_pending_response(question_id):
    pending_responses.pop(question_id, None)
    pending_response_created_at.pop(question_id, None)
