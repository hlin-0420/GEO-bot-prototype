from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import logging
import os
import threading
import time
from threading import Lock
from bs4 import BeautifulSoup
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import json
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Initialize the selected bot. 
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
                sub_table_data = []

            current_subheading = row[column_name]
        else:
            row_list = row.tolist()
            sub_table_data.append(row_list)

    # Add the last collected sub-table
    if current_subheading and sub_table_data:
        sub_tables[current_subheading] = pd.DataFrame(sub_table_data)

    return sub_tables

class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

def load_feedback_dataset():
    with open("feedback_dataset.json", "r") as f:
        return json.load(f)
    
def calculate_reward(feedback_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    similarity_scores = []
    
    for entry in feedback_data:
        response = entry.get("response", "")
        feedback = entry.get("feedback", "")
        
        response_embedding = model.encode(response)
        feedback_embedding = model.encode(feedback)
        similarity_score = util.pytorch_cos_sim(response_embedding, feedback_embedding).item()

        similarity_scores.append(similarity_score)

    # Reward is proportional to the similarity score
    return similarity_scores

# initialise a default name for the models. 
selected_model_name = "llama3.2:latest"
# Declare global variable
rag_application = None 

def extract_table(soup):
    tables = soup.find_all("table")
    
    formatted_tables = []
                    
    # Process and format each table
    for i, table in enumerate(tables, start=1):
        rows = []
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            rows.append(cols)

        # Convert to DataFrame for better readability
        df = pd.DataFrame(rows)
                        
        formatted_table = tabulate(df, headers="firstrow", tablefmt="grid")
        
        formatted_tables.append(formatted_table)
        
    formatted_tables = "\n\n".join(formatted_tables)
    
    return formatted_tables

def extract_text(soup):
    # Extract only meaningful paragraph text
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]  # Exclude very short text
    clean_text = "\n\n".join(paragraphs)
    
    return clean_text

def extract_list(soup):
    # Extract lists properly
    lists = []
    for ul in soup.find_all("ul"):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        lists.append(items)
    return lists

class OllamaBot:
    def __init__(self):
        """
        Initialize the OllamaBot with the specified model.
        
        Args:
            model_name (str): Name of the Ollama model.
            base_directory (str): Path to the base directory containing .htm files.
        """
        self.base_directory = "Data"
        self.contents = []  # Store processed content
        self.web_documents = [] # stores the web documents
        self._load_content()
        self.llm_model = ChatOllama(
            model=selected_model_name,
            temperature=0,
        )
        # Initialize RAG application globally
        self._initialize_rag_application()
        
    def _initialize_rag_application(self):
        """
        Initializes the RAGApplication globally.
        """
        global rag_application
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        
        
        doc_splits = text_splitter.split_documents(self.web_documents)
        
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key="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"
            ),
        )
        
        retriever = vectorstore.as_retriever(k=4)
        
        prompt = PromptTemplate(
            template="""
            You are an assistant for helping the users becoming more familiar with using the GEO   \ 
            application. 
            
            GEO is an integrated a PC-based integrated well log authoring, analysis and reporting  \
            system which has been developed for petroleum geologists, geoscientists and engineers.
            
            Answer the user's questions accurately using retrieved information from the Documents  \
            section precisely. The Document section contains the help content written by software  \ 
            developers for the GEO application. 
            
            Ensure that the answer is concise and answers the question to the point without the    \
            inclusion of any irrelevant information. Only the answer to the question should be     \
            outputted as the generated response. 
            
            Use the information from the section under the title "---Feedback---" as feedback for  \
            making improvements to your answers. Use the feedback as guidelines to determine which \
            area you need to improve your answer after assessing their validity and feasibility. 
            
            Documents
            ----------------------------------------------------------------------------------------
            {documents}
            ----------------------------------------------------------------------------------------
                        
            Question: {question}
            Answer: """,
            input_variables=["question", "documents"],
        )

        # Save the prompt template to a file
        prompt_text = prompt.format(question="<QUESTION_PLACEHOLDER>", documents="<DOCUMENTS_PLACEHOLDER>", answer="<ANSWER_PLACEHOLDER>")
        with open("prompt_visualisation.txt", "w", encoding="utf-8") as file:
            file.write(prompt_text)

        rag_chain = prompt | self.llm_model | StrOutputParser()

        # Set the global variable
        rag_application = RAGApplication(retriever, rag_chain)
        
    def add_contents(self, details):
        
        page_data = {
            'text': details,
        }
                    
        document = Document(
            page_content=page_data['text'],                
        )
        
        self.web_documents.append(document)

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

    def _load_content(self, selectedOptions=None):
        """
        Load and process all .htm files from the base directory.
        """
        htm_files = self._list_htm_files()
        logging.info(f"Found {len(htm_files)} .htm files.")
        
        if selectedOptions is None:
            selectedOptions = ["text", "table", "list"]
        
        # initialise empty training web documents.
        self.web_documents = []
        
        page_texts = []

        for file_path in htm_files:
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    
                    # ignore the redundant header section from content
                    content = content[content.find("<body>")+6:content.find("</body>")]
                    
                    soup = BeautifulSoup(content, "html.parser")
                    
                    page_links = [a['href'] for a in soup.find_all('a', href=True)]
                                                
                    if "text" in selectedOptions:
                        clean_text = extract_text(soup)
                    else:
                        clean_text = "" # when the text, table, or list is empty. 
                    
                    if "table" in selectedOptions:
                        formatted_table = extract_table(soup)
                    else:
                        formatted_table = ""    
                    
                    if list in selectedOptions:    
                        lists = extract_list(soup)
                    else:
                        lists = ""
                        
                    page_text = f"""
                    
                    Tables: 
                    ---
                    {formatted_table}
                    ---
                    
                    Text:
                    ---
                    {clean_text}
                    ---
                    
                    List:
                    ---
                    {lists}
                    ---
                    """
                    
                    page_texts.append(page_text)
                    
                    page_data = {
                        'text': page_text,
                        'link': page_links
                    }
                    
                    document = Document(
                        page_content=page_data['text'],
                        metadata={
                            'links': page_data['link'],
                        }
                    )
                    
                    self.web_documents.append(document)
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")
                
        # Define the file path where the output will be saved
        output_file = "processed_content.txt"
        
        temp_page_texts = "\n\n".join(page_texts)
        
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(temp_page_texts)

        logging.info(f"Processed content saved to {output_file}")
        
        # updates feedback data into the file.
        feedback_data = load_feedback_dataset()
        
        for feedback_entry in feedback_data:
            self.update_training(json.dumps(feedback_entry, indent=4), False)

    def add(self, content):
        """
        Add new content to the bot's memory.
        
        Args:
            content (str): Content to add.
        """
        self.contents.append(content)
        logging.info("New content added.")
            
    def get_model_type(self, model):
        return model.model_name
    
    def update_training(self, data_string, reinitialise):
        
        feedback_heading = "---Feedback---"

        if self.web_documents:
            last_document = self.web_documents[-1]

            if last_document.page_content.startswith(feedback_heading):
                last_document.page_content += f"{data_string}\n"
            else:
                new_document = Document(page_content=f"{feedback_heading}\n{data_string}\n")
                self.web_documents.append(new_document)

        if reinitialise:
            # retrains the application whenever new training data is updated.
            self._initialize_rag_application()

    def query(self, question):
        """
        Query the bot and get a response.

        Args:
            question (str): The user's question.

        Returns:
            str: The response generated by the Ollama model.
        """
        global rag_application  # Access the global variable

        if rag_application is None:
            logging.error("RAG application is not initialized.")
            return "Error: RAG application is not initialized."

        logging.info(f"Processing question: {question}")
        print(f"The selected Model Name for this training is {selected_model_name}")

        response = rag_application.run(question)

        return response


ai_bot = OllamaBot()
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

def append_feedback(feedback_entry, filename="feedback_dataset.json"):
    try:
        # Check if the file exists and is non-empty
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Load existing data
                    if not isinstance(data, list):  # Ensure it's a list
                        data = []
                except json.JSONDecodeError:
                    data = []  # If there's a parsing error, start fresh
        else:
            data = []  # Initialize an empty list if file doesn't exist or is empty

        # Append new entry
        data.append(feedback_entry)

        # Write back as a proper JSON list
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)  # Pretty formatting for readability

        print("Feedback appended successfully!")

    except Exception as e:
        print(f"Error while appending feedback: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit-feedback", methods=["POST"])
def submitFeedback():
    try:    
        global selected_model_name
        data = request.json
        details = data.get("details")
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        if not details and not question:
            return jsonify({"error": "Both feedback details and question details are required"}), 400

        feedback_entry = {
            "model-name": selected_model_name,
            "question": question,
            "response": response,
            "feedback": details,
            "rating-score": rating
        }

        append_feedback(feedback_entry)
        
        feedback_data = load_feedback_dataset()
        
        for feedback_entry in feedback_data:
            ai_bot.update_training(json.dumps(feedback_entry, indent=4), True) # will retrain model after the feedback is updated to the training data. 
        
        return jsonify({"message": "Thank you for your detailed feedback!"}), 200
    except Exception as e:
        app.logger.error(f"Error in /submit-feedback endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

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


def check_selected_options(selectedOptions):
    expected_options = ["text", "table", "list"]
    return set(selectedOptions) == set(expected_options)

def process_question(question_id, question, ai_bot, selectedOptions):
    """
    Simulate long processing of the question and store the response.
    """
    time.sleep(2)  # Simulating "thinking time"
    # try:
    response = ai_bot.query(question)
    
    print(f"Check selected Options: {check_selected_options(selectedOptions)}")
    
    if check_selected_options(selectedOptions) == False:
        # if not all the options are selected, customise the training function. 
        ai_bot._load_content(selectedOptions) # resets the web documents information with the feedback.
    
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)

    stored_responses[question_id] = formatted_response

@app.route("/ask", methods=["POST"])
def ask():
    global question_id
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400
        
        question = data.get("question", "").strip()
        
        selectedOptions = data.get("selectedOptions", "")
        
        print(selectedOptions)
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        with lock:
            current_id = str(question_id)
            question_id += 1

        pending_responses[current_id] = "Processing..."

        threading.Thread(target=process_question, args=(current_id, question, ai_bot, selectedOptions)).start()

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
                formatted_response = response.replace("\n", "<br>")
                yield f"data: {formatted_response}\n\n"
                break
            else:
                yield "data: Error: Invalid question ID\n\n"
                break
            time.sleep(1)  # Polling interval

    return Response(generate_response(), content_type="text/event-stream")

@app.route("/selection", methods=["GET"])
def update_model_name():
    global selected_model_name
    
    model_name = request.args.get("model")  # Retrieve model name from URL parameters

    if not model_name:
        return jsonify({"error": "No model selected"}), 400
    
    selected_model_name = model_name

    return jsonify({"message": f"Model updated to {model_name}"}), 200

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/feedback_dataset.json')
def feedback_data():
    return send_from_directory('.', 'feedback_dataset.json')

if __name__ == "__main__":
    app.run(debug=True)
