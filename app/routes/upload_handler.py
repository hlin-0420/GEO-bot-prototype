from flask import Blueprint, request, jsonify
from app.utils.file_helpers import process_file
from app.services.ollama_bot import get_bot
import os

upload_routes = Blueprint('upload', __name__)

@upload_routes.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./Data/{file.filename}"
    file.save(file_path)
    result = process_file(file_path)
    return jsonify({"message": result})

@upload_routes.route("/view-file", methods=["GET"])
def view_file():
    filename = request.args.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    ai_bot = get_bot()  # Lazy load AI bot

    htm_filepaths = ai_bot._list_htm_files()
    current_directory = os.getcwd()
    temp_directories = [os.path.join(current_directory, htm_filepath) for htm_filepath in htm_filepaths]
    
    file_path = next((path for path in temp_directories if path.endswith(filename)), None)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500
    
@upload_routes.route("/ask-file", methods=["POST"])
def ask_file():
    """Process a question from the uploaded file and store the answer."""
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    ai_bot = get_bot()
    response = ai_bot.query(question)

    append_to_excel(question, response)

    return jsonify({"message": response}), 200