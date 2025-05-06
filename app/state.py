from threading import Lock

current_session_id = None
current_session_messages = []
pending_responses = {}
stored_responses = {}
question_id = 0
lock = Lock()