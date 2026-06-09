import json
import os

from flask import Blueprint, jsonify, render_template, request

from app import config
from app.services.ollama_bot import get_bot
from app.services.session_manager import load_chat_history

ui_blueprint = Blueprint("ui", __name__)

LLM_PARAM_DEFAULTS = {
    "CHUNK_SIZE": 250,
    "CHUNK_OVERLAP": 120,
    "RETRIEVER_TOP_K": 1,
    "DEFAULT_TEMPERATURE": 0,
    "NUM_PREDICT": 150,
    "SIMILARITY_THRESHOLD": 0.92,
    "MAX_CLUSTER_COUNT": 6,
    "MAX_KEYWORDS": 3,
}


@ui_blueprint.route("/")
def index():
    return render_template("index.html")


@ui_blueprint.route("/feedback")
def feedback():
    return render_template("feedback.html")


@ui_blueprint.route("/chathistory")
def chathistory():
    return render_template(
        "chathistory.html",
        model_options=config.MODEL_OPTIONS,
        selected_model=config.selected_model_name,
    )


@ui_blueprint.route("/knowledge-tree")
def knowledge_tree():
    return render_template("knowledge_tree.html")


@ui_blueprint.route("/llm_settings")
def llm_settings():
    return render_template("llm_settings.html", **LLM_PARAM_DEFAULTS)


@ui_blueprint.route("/update_params", methods=["POST"])
def update_params():
    payload = request.get_json(silent=True) or {}
    for key in LLM_PARAM_DEFAULTS:
        if key in payload:
            LLM_PARAM_DEFAULTS[key] = payload[key]
    return jsonify({"message": "Parameters updated", "params": LLM_PARAM_DEFAULTS})


@ui_blueprint.route("/chat-history", methods=["GET"])
def get_chat_history():
    return jsonify(load_chat_history())


@ui_blueprint.route("/chat-history/<session_id>", methods=["GET"])
def get_single_chat_session(session_id):
    session_file = os.path.join(config.CHAT_SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_file):
        return jsonify({"error": "Session not found"}), 404

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return jsonify({"session_id": session_id, "messages": messages}), 200
    except Exception:
        return jsonify({"error": "Internal Server Error"}), 500


def _run_semantic_search(query):
    bot = get_bot()
    rag_application = getattr(bot, "rag_application", None)
    retriever = getattr(rag_application, "retriever", None)

    if retriever is None:
        return []

    results = []
    for document in retriever.invoke(query):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("file_path") or "GEO knowledge base"
        content = getattr(document, "page_content", "").strip()
        if content:
            results.append({
                "source": source,
                "content": content[:1500],
                "score": metadata.get("score"),
            })
    return results


@ui_blueprint.route("/semantic-search", methods=["GET", "POST"])
def semantic_search_page():
    results = []
    query = ""
    error = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                results = _run_semantic_search(query)
            except Exception:
                error = "Semantic search is unavailable while the knowledge base is loading."

    return render_template("semantic_search.html", results=results, query=query, error=error)
