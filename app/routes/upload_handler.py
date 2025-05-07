from flask import Blueprint, request, jsonify
from app.utils.file_helpers import process_file

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