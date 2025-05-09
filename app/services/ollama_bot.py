# app/services/ollama_bot.py
import threading

_ai_bot = None
_lock = threading.Lock()  # Prevent race conditions on initialization

def get_bot(force_refresh=False):
    global _ai_bot

    with _lock:
        if force_refresh or _ai_bot is None:
            from app.services.ollama_bot_core import OllamaBot  # Split core logic
            _ai_bot = OllamaBot()
    return _ai_bot
