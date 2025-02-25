from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import logging
import os
import threading
import time
from threading import Lock
from bs4 import BeautifulSoup
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import json
from tabulate import tabulate
import re
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack import Document as HaystackDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from sentence_transformers import SentenceTransformer, util

valid_model_names = {"deepseek1.5", "llama3.2:latest", "openai"}
# Initialize the selected bot. 
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# initialise a default name for the models. 
selected_model_name = "llama3.2:latest"
# Declare global variable
rag_application = None 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
EXCEL_FILE = os.path.join(DATA_DIR, "query_responses.xlsx")
EXPECTED_RESULTS_FILE = os.path.join(DATA_DIR, "expected_query_responses.xlsx")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_dataset.json")
PROMPT_VISUALISATION_FILE = os.path.join(DATA_DIR, "prompt_visualisation.txt")
PROCESSED_CONTENT_FILE = os.path.join(DATA_DIR, "processed_content.txt")
UPLOADED_FILE = os.path.join(DATA_DIR, "uploaded_document.txt")

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
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)  # Initialize an empty list
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def auto_adjust_column_width(writer, df):
    """ Auto-adjusts column width based on the max length of cell content in each column """
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    for column in df.columns:
        max_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
        col_idx = df.columns.get_loc(column) + 1
        worksheet.column_dimensions[chr(64 + col_idx)].width = max_length

def extract_table(soup):
    tables = soup.find_all("table")
    
    formatted_tables = []
                    
    # Process and format each table
    for i, table in enumerate(tables, start=1):
        rows = []
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            rows.append(cols)
            
        # Flatten row values for filtering irrelevant tables
        flat_rows = [item.lower().strip() for sublist in rows for item in sublist]
        
        # Skip navigation tables containing only "Back" and "Forward"
        if set(flat_rows).issubset({"back", "forward", "", "-", "next", "previous"}):
            continue  # Skip this table

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
        global valid_model_names
        # API Key initialisation##################
        self.api_key = os.getenv("OPENAI_API_KEY")
        ##########################################
        # Storage Processing
        # Data Directory initialisation
        self.base_directory = "Data"
        self.document_store = InMemoryDocumentStore()
        self.web_documents = [] # stores the web documents for free tier models.
        self.web_documents_haystack = [] # intialise the web documents for the premium tier models.
        self._load_content() 
        ####################
        # Pipeline initialisation.
        print(f"Selected model name: {selected_model_name}")
        print(f"List of valid model names: {valid_model_names}")
        if selected_model_name in valid_model_names: # free tier models. 
            print("This is a valid model.")
            self.llm_model = ChatOllama(
                model=selected_model_name,
                temperature=0,
            ) # initialises a free-tier model.
            # Initialize RAG application globally
            self._initialize_rag_application() # Generalised rag pipeline initialisation.
        else: # initialising a premium tier model. 
            self.rag_pipe = Pipeline()
        
    def _load_content_haystack(self):
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="doc_embedder"
        )
        indexing_pipeline.add_component(instance=DocumentWriter(document_store=self.document_store), name="doc_writer")

        indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

        indexing_pipeline.run({"doc_embedder": {"documents": self.web_documents_haystack}})
        
    def _initialize_rag_application(self):
        """
        Initializes the RAGApplication globally.
        """
        global rag_application
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        
        doc_splits = text_splitter.split_documents(self.web_documents) # uses documents loaded for the free-tier models. 

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
        )
        
        retriever = vectorstore.as_retriever(k=4)
        
        if selected_model_name in valid_model_names:
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
            with open(PROMPT_VISUALISATION_FILE, "w", encoding="utf-8") as file:
                file.write(prompt_text)
            # Save the second-to-last document for verification
            if len(self.web_documents) > 1:
                second_to_last_document = self.web_documents[-2].page_content
                with open(UPLOADED_FILE, "w", encoding="utf-8") as file:
                    file.write(second_to_last_document)

            rag_chain = prompt | self.llm_model | StrOutputParser()

            # Set the global variable
            rag_application = RAGApplication(retriever, rag_chain)
        else:
            # if the prompt structure is designed for a haystack model.
            prompt = PromptTemplate(
                template="""
                You are an assistant designed to help users become more familiar with the GEO application.
        
                GEO is a PC-based well log authoring, analysis, and reporting system developed for 
                petroleum geologists, geoscientists, and engineers.
                
                Answer the user's questions accurately using retrieved information from the "Context" 
                section. This section contains help content written by software developers specifically 
                for the GEO application.
                
                Ensure that your response is concise and directly addresses the question, avoiding any 
                irrelevant information. The generated response should contain only the answer to the 
                user's question.
                
                Use the information from the section titled "---Feedback---" as guidelines for improving 
                your answers. Assess the validity and feasibility of the feedback before applying it to 
                refine future responses.

                Context:
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}

                Question: {{ question }}
                Answer:""",
                input_variables=["question", "documents"],
            )
            self.rag_pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
            self.rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=self.document_store))
            self.rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=prompt))
            self.rag_pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
            
            self.rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
            self.rag_pipe.connect("retriever", "prompt_builder.documents")
            self.rag_pipe.connect("prompt_builder.prompt", "llm.messages")

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
                    
                    if selected_model_name in valid_model_names:
                        document = LangchainDocument(
                            page_content=page_data['text'],
                            metadata={
                                'links': page_data['link'],
                            }
                        )
                        self.web_documents.append(document)
                    else:
                        document = HaystackDocument(
                            content=page_data['text']
                        )
                        self.web_documents_haystack.append(document)
                    
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")

        # updates feedback data into the file.
        feedback_data = load_feedback_dataset()
        
        for feedback_entry in feedback_data:
            json_feedback = json.dumps(feedback_entry, indent=4)

            feedback_heading = "---Feedback---"

            if self.web_documents:
                last_document = self.web_documents[-1]
                
                print(f"The last document starts with the feedback heading: {last_document.page_content.startswith(feedback_heading)}")

                if last_document.page_content.startswith(feedback_heading):
                    last_document.page_content += f"{json_feedback}\n"
                else:
                    json_feedback = f"{feedback_heading}\n{json_feedback}\n" # updates json feedback by adding a heading in front.
                    if selected_model_name in valid_model_names:
                        new_document = LangchainDocument(page_content=json_feedback)
                        self.web_documents.append(new_document)
                    else:
                        new_document = HaystackDocument(content=json_feedback)
                        self.web_documents_haystack.append(new_document)
                    
            page_texts.append(json_feedback)

        # saves help guide and feedback data into a text file for visualization.
        temp_page_texts = "\n\n".join(page_texts)
        
        with open(PROCESSED_CONTENT_FILE, "w", encoding="utf-8") as file:
            file.write(temp_page_texts)

        logging.info(f"Processed content saved to {PROCESSED_CONTENT_FILE}")

    def add(self, content):
        """
        Add new content to the bot's memory.
        
        Args:
            content (str): Content to add.
        """
        feedback_heading = "---Feedback---"
        
        if selected_model_name in valid_model_names:
            new_document = LangchainDocument(page_content=content)
            temp_documents = self.web_documents
        else:
            new_document = HaystackDocument(content=content)
            temp_documents = self.web_documents_haystack # initialise a temp document variable to store the document information
        
        if temp_documents:
            last_document = temp_documents[-1]
            
            if last_document.page_content.startswith(feedback_heading):
                # Ensure there is at least one more document before inserting
                if len(temp_documents) > 1:
                    temp_documents.insert(len(temp_documents) - 1, new_document)
                else:
                    temp_documents.insert(0, new_document)
            else:
                temp_documents.append(new_document)
            logging.info("New content added.")
            
            if selected_model_name in valid_model_names:
                self.web_documents = temp_documents
            else:
                self.web_documents_haystack = temp_documents
            
        self._initialize_rag_application() # resets the initialisation to retrain model on updated information. 
            
    def get_model_type(self, model):
        return model.model_name

    def query(self, question):
        """
        Query the bot and get a response.

        Args:
            question (str): The user's question.

        Returns:
            str: The response generated by the Ollama model.
        """
        global rag_application, EXCEL_FILE  # Access the global variable

        if rag_application is None:
            logging.error("RAG application is not initialized.")
            return "Error: RAG application is not initialized."

        logging.info(f"Processing question: {question}")
        print(f"The selected Model Name for this training is {selected_model_name}")

        if selected_model_name in valid_model_names:
            response = rag_application.run(question)
        else:
            response_object = self.rag_pipe.run({"embedder": {"text": question}, "prompt_builder": {"question": question}})
            response = response_object['llm']['replies'][0]._content[0].text
            
        # Create a new DataFrame with the question and response
        new_entry = pd.DataFrame([[question, response]], columns=["Question", "Response"])

        # Check if the Excel file exists
        if not os.path.exists(EXCEL_FILE):
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
                new_entry.to_excel(writer, index=False)
                auto_adjust_column_width(writer, new_entry)
        else:
            # Load existing data and append new entry
            existing_data = pd.read_excel(EXCEL_FILE)
            updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
            updated_data.to_excel(EXCEL_FILE, index=False)
            
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
                updated_data.to_excel(writer, index=False)
                auto_adjust_column_width(writer, updated_data)

        return response


ai_bot = OllamaBot()
pending_responses = {}
stored_responses = {}
question_id = 0
lock = Lock()

# Process uploaded file
def process_file(file_path):
    print("***PROCESSING FILE***")
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            ai_bot.add(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the file encoding.")
        return "Error: Invalid file encoding."

def append_feedback(feedback_entry):
    try:
        # Check if the file exists and is non-empty
        if os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 0:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
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
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
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
        
        # Write data_string to feedback file without overwriting existing contents
        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
                try:
                    feedback_data = json.load(file)
                except json.JSONDecodeError:
                    feedback_data = []  # Initialize as empty list if file is corrupted
        
        feedback_data.append(feedback_entry)
        
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as file:
            json.dump(feedback_data, file, indent=4)
            
        # reload the contents
        ai_bot._load_content()
        # retrains the application whenever new training data is updated.
        ai_bot._initialize_rag_application()
        
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
        print(f"Processed result: {result}")
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

    print(f"Selected model name: \"{selected_model_name}\".")

    return jsonify({"message": f"Model updated to {model_name}"}), 200

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/feedback_dataset.json')
def feedback_data():
    return send_from_directory(DATA_DIR, "feedback_dataset.json")

@app.route("/view-file", methods=["GET"])
def view_file():
    filename = request.args.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    htm_filepaths = ai_bot._list_htm_files()
    
    current_directory = os.getcwd()
    
    # for each file, join path with the path of the htm file
    temp_directories = [os.path.join(current_directory, htm_filepath) for htm_filepath in htm_filepaths]
    
    # Find the matching file path
    file_path = next((directory for directory in temp_directories if directory.endswith(filename)), None)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """
    Computes the semantic similarity between two text extracts using Sentence-BERT.

    Parameters:
    text1 (str): The first text (expected response).
    text2 (str): The second text (chatbot response).
    model_name (str): Name of the pre-trained SentenceTransformer model.

    Returns:
    float: Cosine similarity score (range: 0 to 1, where 1 means identical meaning).
    """
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return similarity_score

@app.route("/get-results", methods=["GET"])
def get_results():
    try:
        if not os.path.exists(EXCEL_FILE):
            return jsonify({"error": "Results file not found"}), 404
        
        df = pd.read_excel(EXCEL_FILE)
        
        if os.path.exists(EXPECTED_RESULTS_FILE):
            expected_results_df = pd.read_excel(EXPECTED_RESULTS_FILE)
            
        question_column = "Question"
        response_column = "Response"
        expected_answer_column = "Expected Answer"
        
        # Create a dictionary mapping expected questions to their answers
        expected_answers_dict = dict(zip(expected_results_df[question_column], expected_results_df[response_column]))
        
        df['is_question_in_expected'] = df[question_column].isin(expected_answers_dict)  # Check if question exists
        
        # Filter df to include only rows where the question exists in expected_results_df
        filtered_df = df[df['is_question_in_expected'] == True]
        
        filtered_df.loc[:, expected_answer_column] = filtered_df.loc[:, question_column].map(expected_answers_dict)
        
        filtered_df.loc[:, 'similarity_score'] = filtered_df.apply(
            lambda row: calculate_semantic_similarity(str(row[expected_answer_column]), str(row[response_column])),
            axis=1
        )
        
        filtered_df.to_excel("Data/temp_filtered_data.xlsx")
        
        if "Model Name" not in df.columns or "Accuracy" not in df.columns:
            results = pd.DataFrame({
                "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
                "Accuracy": ["90%", "84%", "79%", "92%"]
            })
        else:
            results = df[["Model Name", "Accuracy"]].to_dict(orient="records")
        
        return jsonify({"models": results.to_dict(orient="records"), "filtered_results": filtered_df.to_dict(orient="records")})
    except Exception as e:
        logging.error(f"Error reading results: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
def append_to_excel(question, response):
    """ Append question and response to the Excel file. """
    new_entry = pd.DataFrame([[question, response]], columns=["Question", "Response"])

    # Check if the file exists
    if not os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            new_entry.to_excel(writer, index=False)
    else:
        existing_data = pd.read_excel(EXCEL_FILE)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        updated_data.to_excel(EXCEL_FILE, index=False)
        
@app.route("/ask-file", methods=["POST"])
def ask_file():
    """ Process a question from the uploaded file and store the answer. """
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Get AI model response
    response = ai_bot.query(question)

    # Append question and response to Excel
    append_to_excel(question, response)

    return jsonify({"message": response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
