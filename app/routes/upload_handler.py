import os
import uuid

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from app import config
from app.services.ollama_bot import get_bot
from app.services.ollama_bot_helpers import list_htm_files
from app.utils.file_helpers import append_to_excel, process_file

upload_routes = Blueprint("upload", __name__)


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_UPLOAD_EXTENSIONS


@upload_routes.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    if not filename or not _allowed_file(filename):
        allowed = ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed extensions: {allowed}"}), 400

    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(config.UPLOADS_DIR, unique_filename)
    file.save(file_path)

    result = process_file(file_path)
    return jsonify({"message": result, "filename": unique_filename})


@upload_routes.route("/view-file", methods=["GET"])
def view_file():
    filename = request.args.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    requested_filename = os.path.basename(filename)
    htm_filepaths = list_htm_files(config.DATA_DIR)
    file_path = next((path for path in htm_filepaths if os.path.basename(path) == requested_filename), None)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"content": content})
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 500


@upload_routes.route("/ask-file", methods=["POST"])
def ask_file():
    """Process a question from the uploaded file and store the answer."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    ai_bot = get_bot()
    response = ai_bot.query(question)

    append_to_excel(question, response)

    return jsonify({"message": response}), 200
