from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import json
import logging
import os
from embedchain import App
from chromadb.utils import embedding_functions

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AIBot:
    def __init__(self, api_key, model_name, base_directory="Data"):
        """
        Initialize the AIBot with an embedding function and load content from the specified directory.
        
        Args:
            api_key (str): OpenAI API key.
            model_name (str): Model name for the embedding function.
            base_directory (str): Path to the base directory containing .htm files.
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=model_name
        )
        self.ai_bot = App()
        self.base_directory = base_directory
        self._load_content()

    def _list_htm_files(self):
        """
        Recursively finds all .htm files in the base directory and its subdirectories.
        
        Returns:
            list: A list of file paths relative to the base directory.
        """
        htm_files = []
        for root, _, files in os.walk(self.base_directory):
            for file in files:
                if file.endswith(".htm"):
                    relative_path = os.path.relpath(os.path.join(root, file), start=self.base_directory)
                    htm_files.append(self.base_directory + "/" + relative_path)
        return htm_files

    def _load_content(self):
        """
        Load and process all .htm files from the base directory.
        """
        file_paths = self._list_htm_files()
        logging.info(f"Found {len(file_paths)} .htm files.")

        for file_path in file_paths:
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    self._process_content(content)
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")
    
    def _process_content(self, content):
        """
        Process the content of an HTML file and add it to the AI bot.

        Args:
            content (str): The content of the HTML file.
        """
        soup = BeautifulSoup(content, "html.parser")

        # Extract table data if available
        limits_data = []
        if soup.find("table"):
            table = soup.find("table")
            rows = table.find_all("tr")
            current_section = None

            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)

                    if not value and key:
                        current_section = key
                    elif current_section and key:
                        limits_data.append({"Section": current_section, "Type": key, "Limit": value})
        
        if limits_data:
            limits_data_str = json.dumps(limits_data, indent=4)
            self.ai_bot.add(limits_data_str)

        # Add the full HTML content
        self.ai_bot.add(content)
        logging.info("Content added to AI bot.")

    def add(self, content):
        """
        Add new content to the AI bot.
        
        Args:
            content (str): Content to add.
        """
        self.ai_bot.add(content)
        logging.info("New content added to the AI bot.")

    def query(self, question):
        """
        Query the AI bot and get a response.

        Args:
            question (str): The user's question.

        Returns:
            str: The response from the AI bot.
        """
        logging.info(f"Processing question: {question}")
        return self.ai_bot.query(question)

api_key = "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"

ai_bot = AIBot(api_key, "gpt-4o")

# Process uploaded file
def process_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            soup = BeautifulSoup(content, "html.parser")

            limits_data = []

            if soup.find("table"):
                table = soup.find("table")
                rows = table.find_all("tr")
                current_section = None

                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) == 2:
                        key = cols[0].get_text(strip=True)
                        value = cols[1].get_text(strip=True)

                        # Check if it's a section header
                        if not value and key:
                            current_section = key
                        elif current_section and key:
                            # Add data to the limits list with section
                            limits_data.append({"Section": current_section, "Type": key, "Limit": value})

            limits_data_str = json.dumps(limits_data, indent=4)
            ai_bot.add(limits_data_str)

        ai_bot.add(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the file encoding.")
        return "Error: Invalid file encoding."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = f"./Data/{file.filename}"
        file.save(file_path)
        result = process_file(file_path)
        return jsonify({"message": result})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    response = ai_bot.query(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)