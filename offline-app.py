from flask import Flask, request, jsonify, render_template, Response
import logging
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import threading
import time
from threading import Lock
import re
from bs4 import BeautifulSoup
import pandas as pd

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_table_by_subheadings(df, column_name):
    sub_tables = {}
    current_subheading = None
    sub_table_data = []
    
    # skip the first row
    df = df[1:]
    
    # first row has the columns
    column_names = df.iloc[0].to_list()
    
    df = df[1:]

    for _, row in df.iterrows():
        
        if row['Limit'] == '':  # Identify subheadings based on NaN in the 'Limit' column
            if current_subheading and sub_table_data:
                
                sub_tables[current_subheading] = pd.DataFrame(sub_table_data, columns=column_names)
                print(f"Subtable: \n{sub_tables[current_subheading]}")
                sub_table_data = []

            current_subheading = row[column_name]
        else:
            row_list = row.tolist()
            print(f"Row data: {row_list}")
            sub_table_data.append(row_list)

    # Add the last collected sub-table
    if current_subheading and sub_table_data:
        sub_tables[current_subheading] = pd.DataFrame(sub_table_data)

    return sub_tables

class OllamaBot:
    def __init__(self, model_name):
        """
        Initialize the OllamaBot with the specified model.
        
        Args:
            model_name (str): Name of the Ollama model.
            base_directory (str): Path to the base directory containing .htm files.
        """
        self.model = OllamaLLM(model=model_name, host="https://geo-bot-prototype.vercel.app/")  # Instantiate the Ollama model
        self.base_directory = "Data"
        self.contents = []  # Store processed content
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
        htm_files = self._list_htm_files()
        logging.info(f"Found {len(htm_files)} .htm files.")

        for file_path in htm_files:
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    
                    # ignore the redundant header section from content
                    content = content[content.find("<body>")+6:content.find("</body>")]
                    
                    soup = BeautifulSoup(content, "html.parser")
                    
                    if soup.find("table"):
                        
                        table_data = []
                        
                        for table in soup.find_all('table'):
                            rows = table.find_all('tr')
                            
                            for row in rows:
                                cols = row.find_all("td")
                                cols = [col.get_text(strip=True) for col in cols]
                                if all(not col for col in cols):
                                    continue
                                # if the column contains "back" and "forward" - skip
                                if cols[0] == "Back" and cols[1] == "Forward":
                                    continue                            
                                table_data.append(cols)
                                
                        if len(table_data) > 1:
                            
                            table_headings = table_data[0]
                            table_data = table_data[1: ]
                                    
                            table_data_df = pd.DataFrame(table_data)
                            
                            table_data_df.columns = table_headings
                            
                            self.contents.append(table_data_df)
                    
                    if file_path.endswith("GEO_Limits.htm"):
                        print(f"Contents: \n {content}\n")
                    self.contents.append(content)
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")

    def add(self, content):
        """
        Add new content to the bot's memory.
        
        Args:
            content (str): Content to add.
        """
        self.contents.append(content)
        logging.info("New content added.")
        
    def train_model(self):
        """
        Train or fine-tune the Llama model using self.contents as training data.
        """
        logging.info("Training model with provided content data.")

        # Preprocess the contents for training
        training_data = "\n\n".join(self.contents)  # Join all contents into a single training text
        training_inputs = {"training_data": training_data}

        # Fine-tune or update the model
        try:
            self.model.train(training_inputs)  # Assuming the model has a `train` method
            logging.info("Model training completed successfully.")
        except AttributeError:
            logging.error("The current Llama model does not support training.")
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")

    def query(self, question):
        """
        Query the bot and get a response.

        Args:
            question (str): The user's question.

        Returns:
            str: The response generated by the Ollama model.
        """
        logging.info(f"Processing question: {question}")

        template = """
        As an experienced geologist specialised in the GEO application, a specialised help system \
        for guiding users working as a well site geologist, please provide an answer to the question:\
        \n {question} \n

        Given the list of topics as:
        {topics}

        Answer: 
        """

        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model = "llama3")

        chain = prompt | model # chain the operations together.

        topics = "Touch Screen Devices, GEO Navigation, File Processing, \
            Log structure and Presentation, Loading Curve Data, Displaying Curve Data, \
            Create Curve Data, Curve Shading, TVD, Interpreting Information, \
            Text and Annotations, Lines, Tables, Headers and Trailers, \
            Printing, Sidetrack, Sharing, Additional Applications, \
            Compute Curve Templates"

        response = chain.invoke({"question": question, "topics": topics})

        return response


# Initialize OllamaBot
model_name = "llama3"  
ai_bot = OllamaBot(model_name)
pending_responses = {}
stored_responses = {}
question_id = 0
lock = Lock()

# Process uploaded file
def process_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
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
        print("\nFile Saved.\n")
        result = process_file(file_path)
        return jsonify({"message": result})


def process_question(question_id, question):
    """
    Simulate long processing of the question and store the response.
    """
    time.sleep(2)  # Simulating "thinking time"
    try:
        print("Question: ", question)
        response = ai_bot.query(question)

        # check through the response string and add <br> to replace the new line character
        response = response.replace("\n", "<br>")

        # check if both end of a string has "**", the replace with bold font tags
        response = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', response)

        print("Response: ", response)

        stored_responses[question_id] = response
    except Exception as e:
        stored_responses[question_id] = "Still thinking about how to answer..."

@app.route("/ask", methods=["POST"])
def ask():
    global question_id
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400
        
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        with lock:
            current_id = str(question_id)
            question_id += 1

        pending_responses[current_id] = "Processing..."

        threading.Thread(target=process_question, args=(current_id, question)).start()

        return jsonify({"question_id": current_id}), 200
    
    except Exception as e:

        app.logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    """
    Endpoint for EventSource to fetch the response.
    """
    def generate_response():
        while True:
            response = stored_responses.get(question_id)
            
            if response == "Processing" or response is None:
                yield "data: Processing your question...\n\n"
            elif response:
                yield f"data: {response}\n\n"
                break
            else:
                yield "data: Error: Invalid question ID\n\n"
                break
            time.sleep(1)  # Polling interval

    return Response(generate_response(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
