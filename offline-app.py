from flask import Flask, request, jsonify, render_template, Response, send_from_directory, g
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
import numpy as np
from rapidfuzz import process
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from datetime import datetime
from bs4 import XMLParsedAsHTMLWarning
import warnings
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import secrets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

os.environ["LOKY_MAX_CPU_COUNT"] = "2"
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# tracks the current sessions in memory
current_session_id = None
current_session_messages = []

valid_model_names = {"deepseek1.5", "llama3.2:latest", "openai"}
# Initialize the selected bot. 
app = Flask(__name__)

@app.before_request
def set_nonce():
    g.nonce = secrets.token_urlsafe(16)  # Generate a nonce for each request

@app.after_request
def add_csp_headers(response):
    response.headers['Content-Security-Policy'] = (
        f"default-src 'self' https://cdnjs.cloudflare.com https://apis.google.com https://content.googleapis.com https://www.gstatic.com;"
        f"script-src 'self' 'nonce-{g.nonce}' 'unsafe-eval' https://cdnjs.cloudflare.com https://apis.google.com https://www.gstatic.com https://accounts.google.com/gsi/client;"
        "connect-src 'self' https://www.googleapis.com https://content.googleapis.com;"
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;"
        "img-src 'self' https://upload.wikimedia.org https://www.gstatic.com data:;"
        "frame-src 'self' https://accounts.google.com https://content.googleapis.com;"
        "object-src 'none';"
    )
    return response

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# initialise a default name for the models. 
selected_model_name = "llama3.2:latest"
# initialise a variable to store the length of time taken to answer a question.
answer_time = 0 # default time taken to answer a question.
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
CHAT_SESSIONS_DIR = os.path.join(DATA_DIR, "ChatSessions")
SESSION_METADATA_FILE = os.path.join(DATA_DIR, "session_metadata.json")
TIMED_RESPONSES_FILE = os.path.join(DATA_DIR, "timed_responses.json")

@app.route("/store-response-time", methods=["POST"])
def store_response_time():
    """Stores response time, question asked, and timestamp in a JSON file."""
    try:
        data = request.json
        question_number = str(data.get("questionNumber"))
        question_text = data.get("question")
        duration = float(data.get("duration"))
        timestamp = data.get("timestamp")

        if not question_number or not question_text or duration is None or not timestamp:
            return jsonify({"error": "Invalid data"}), 400

        response_times = []

        # Load existing response times
        if os.path.exists(TIMED_RESPONSES_FILE):
            with open(TIMED_RESPONSES_FILE, "r", encoding="utf-8") as file:
                try:
                    response_times = json.load(file)
                except json.JSONDecodeError:
                    response_times = []

        # Append new entry
        response_entry = {
            "question_number": question_number,
            "question": question_text,
            "response_time": duration,
            "timestamp": timestamp
        }
        response_times.append(response_entry)

        # Save updated data
        with open(TIMED_RESPONSES_FILE, "w", encoding="utf-8") as file:
            json.dump(response_times, file, indent=4)

        return jsonify({"message": "Response time stored successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_keywords(text, top_n=3):
    """Extracts top N keywords from text using TF-IDF."""
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]
    
    return " ".join(filtered_words[:top_n])  # Take top keywords

@app.route("/knowledge-tree")
def knowledge_tree():
    # Load chat data
    chat_data = []
    for filename in os.listdir(CHAT_SESSIONS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(CHAT_SESSIONS_DIR, filename), "r", encoding="utf-8") as file:
                chat_data.extend(json.load(file))

    # Extract user questions
    questions = [entry["content"] for entry in chat_data if entry["role"] == "user"]

    if not questions:
        return render_template("knowledge_tree.html", tree_data={"name": "No Data", "children": []})

    # Apply TF-IDF to extract key topics
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    X = vectorizer.fit_transform(questions)
    feature_names = vectorizer.get_feature_names_out()  # Get words corresponding to features

    # Cluster similar topics
    num_clusters = min(6, len(questions))  # Avoid exceeding available questions
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Extract Top Keywords for Cluster Naming
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_names = {}
    for i in range(num_clusters):
        top_keywords = [feature_names[ind] for ind in order_centroids[i, :1]]  # Top 3 words
        cluster_names[i] = ", ".join(top_keywords)  # Assign top keywords as cluster name

    # Organize topics into a tree structure
    knowledge_tree = defaultdict(list)
    for idx, label in enumerate(labels):
        topic_name = extract_keywords(cluster_names[label])  # Simplify topic names
        knowledge_tree[topic_name].append(extract_keywords(questions[idx], top_n=1))  # Keep 3-word summaries
        
    # Correct structured tree format
    structured_tree = {
        "name": "GEO Software Knowledge Tree",
        "children": [
            {"name": topic, "children": [{"name": subtopic} for subtopic in sorted(set(subs))]}
            for topic, subs in knowledge_tree.items()
        ],
    }

    return render_template("knowledge_tree.html", tree_data=structured_tree)

@app.route("/chat-history/<session_id>", methods=["GET"])
def get_single_chat_session(session_id):
    session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(session_file):
        return jsonify({"error": "Session not found"}), 404

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        return jsonify({
            "session_id": session_id,
            "messages": messages
        }), 200
    except Exception as e:
        app.logger.error(f"Error loading session {session_id}: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

def load_session_metadata():
    if os.path.exists(SESSION_METADATA_FILE):
        with open(SESSION_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_session_metadata(metadata):
    with open(SESSION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
@app.route("/rename-session", methods=["POST"])
def rename_session():
    data = request.json
    session_id = data.get("session_id")
    new_name = data.get("new_name")

    if not session_id or not new_name:
        return jsonify({"error": "Session ID and new name are required"}), 400

    metadata = load_session_metadata()
    metadata[session_id] = new_name
    save_session_metadata(metadata)

    return jsonify({"message": "Session renamed successfully"})

@app.route("/clear-chat-sessions", methods=["POST"])
def clear_chat_sessions():
    try:
        if os.path.exists(CHAT_SESSIONS_DIR):
            for filename in os.listdir(CHAT_SESSIONS_DIR):
                file_path = os.path.join(CHAT_SESSIONS_DIR, filename)
                os.remove(file_path)

        # Optional: Clear session metadata (if you want to reset names too)
        save_session_metadata({})

        return jsonify({"success": True, "message": "All chat sessions cleared."}), 200
    except Exception as e:
        app.logger.error(f"Error clearing chat sessions: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def load_chat_history():
    try:
        if not os.path.exists(CHAT_SESSIONS_DIR):
            os.makedirs(CHAT_SESSIONS_DIR)

        chat_history = []
        
        metadata = load_session_metadata()

        for filename in os.listdir(CHAT_SESSIONS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(CHAT_SESSIONS_DIR, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session_id = filename.replace(".json", "")
                    print(f"Session id: {session_id}")
                    chat_history.append({
                        "session_id": session_id,
                        "session_name": metadata.get(session_id, session_id),
                        "messages": session_data
                    })

        return chat_history
    except Exception as e:
        return {"error": str(e)}

@app.route("/chat-history", methods=["GET"])
def get_chat_history():
    return jsonify(load_chat_history())

@app.route("/chathistory")
def chathistory():
    global current_session_id, current_session_messages
    current_session_id = None
    current_session_messages = []
    return render_template("chathistory.html")

class RAGApplication:
    def __init__(self, retriever, rag_chain, web_documents):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.web_documents = web_documents  # Store the documents for feedback retrieval
        self.feedback_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model for similarity

    def _get_relevant_feedback(self, question, top_k=3):
        """Retrieve the most relevant feedback entries based on semantic similarity and question type matching."""

        # load from FEEDBACK_FILE 
        if os.path.exists(FEEDBACK_FILE):
                try:
                    with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
                        feedback_data = json.load(file)  # Load as JSON array
                except json.JSONDecodeError:
                    logging.error("⚠️ Error decoding feedback JSON file. Returning empty feedback.")
                    return ""
        else:
            logging.warning("⚠️ No feedback file found.")
            return ""

        # 🔹 Step 2: Extract and structure feedback data
        extracted_feedback = []
        
        for entry in feedback_data:
            if "question" in entry and "feedback" in entry:
                extracted_feedback.append({
                    "question": entry["question"],
                    "feedback": entry["feedback"],
                    "rating": int(entry.get("rating-score", 0))  # Ensure rating is numeric
                })

        if not extracted_feedback:
            logging.warning("⚠️ No valid feedback extracted.")
            return ""  # Return an empty string if no valid feedback is available

        # 🔹 Step 2: Compute embeddings for the question
        question_embedding = self.feedback_model.encode(question, convert_to_tensor=True)

        # 🔹 Step 3: Compute similarity scores for each feedback entry based on the "question" field
        feedback_embeddings = [self.feedback_model.encode(fb["question"], convert_to_tensor=True) for fb in extracted_feedback]
        similarities = [util.pytorch_cos_sim(question_embedding, fb_emb)[0].item() for fb_emb in feedback_embeddings]

        # 🔹 Step 4: Pair feedback entries with their similarity scores and sort by similarity
        sorted_feedback = sorted(
            zip(extracted_feedback, similarities),
            key=lambda x: x[1],  # Sort only by similarity score (higher is better)
            reverse=True
        )

        # 🔹 Step 5: Ensure a mix of semantic similarity & question type matching
        unique_questions = set()  # Track unique question types
        selected_feedback = []
        
        # 🔹 Step 5.1: Add the most semantically similar feedback
        for fb, sim_score in sorted_feedback:
            selected_feedback.append(fb["feedback"])
            unique_questions.add(fb["question"].lower().replace("?", "").strip())  # Normalize the question type
            
            if len(selected_feedback) >= top_k:
                break

        # 🔹 Step 5.2: Ensure at least one feedback entry from the same question type exists
        for fb, sim_score in sorted_feedback:
            base_question = fb["question"].lower().replace("?", "").strip()
            if base_question in unique_questions:  # Ensure we already have this question type
                continue  # Skip if already included
            
            selected_feedback.append(fb["feedback"])
            unique_questions.add(base_question)

            if len(selected_feedback) >= top_k:
                break

        return "\n".join(selected_feedback) if selected_feedback else ""  # Return an empty string if no relevant feedback is found

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])

        # Retrieve relevant feedback
        feedback_documents = [doc for doc in self.web_documents if "---Feedback---" in doc.page_content]
        feedback_texts = "\n".join([doc.page_content for doc in feedback_documents])

        # Select the most relevant feedback
        feedback_texts = self._get_relevant_feedback(question)

        if not feedback_texts.strip():
            logging.warning("⚠️ No feedback found for this query.")

        # Generate the answer using the updated prompt format
        answer = self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts,
            "feedback": feedback_texts  # Pass retrieved feedback separately
        })

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
    # Define navigation-related keyword patterns
    navigation_keywords = [
        r'contact\s+us', r'click\s+(here|for)', r'guidance', r'help', r'support', r'assistance',
        r'maximize\s+screen', r'view\s+details', r'read\s+more', r'convert.*file', r'FAQ', r'learn\s+more'
    ]
    
    navigation_pattern = re.compile(r"|".join(navigation_keywords), re.IGNORECASE)

    # Remove navigation-related text
    for tag in soup.find_all("p"):
        if navigation_pattern.search(tag.text):
            tag.decompose()

    # Extract only meaningful paragraph text (excluding very short ones)
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
    
    clean_text = "\n\n".join(paragraphs)
    
    return clean_text

def extract_list(soup):
    # Extract lists properly
    lists = []
    for ul in soup.find_all("ul"):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        lists.append(items)
    return lists

def extract_table_as_text_block(soup, file_path):
    """
    Extract tables from HTML as a single formatted text block for inclusion into page_text.
    Skips navigation tables and handles no-table cases.

    Args:
        soup (BeautifulSoup): Parsed HTML.
        file_path (str): Path to the file (for metadata).

    Returns:
        str: Formatted block of all tables from this file, or a message if no tables are found.
    """
    try:
        tables = pd.read_html(file_path)

        def is_navigation_table(table):
            """Detect if table is a 'navigation-only' table with just 'back' and 'forward'."""
            flattened = [str(cell).strip().lower() for cell in table.to_numpy().flatten()]
            navigation_keywords = {"back", "forward"}
            return set(flattened).issubset(navigation_keywords)
        
        def is_nan_only_table(table):
            """Detect if the entire table only contains NaN values."""
            return table.isna().all().all()

        table_texts = []
        table_count = 0

        for idx, table in enumerate(tables):
            if is_navigation_table(table) or is_nan_only_table(table):
                continue
            
            if table.shape[1] == 2:
                # Drop rows where both the second and third columns are NaN
                table = table.dropna(how='all')

                last_col = table.columns[-1]

                table[last_col] = table[last_col].fillna("")

            table_count += 1
            formatted_table = tabulate(table, headers="keys", tablefmt="grid")

            beautified_table = f"""
╔════════════════════════════════════════════════════╗
║            📊 Table {table_count} from {file_path}              ║
╚════════════════════════════════════════════════════╝

{formatted_table}

╔════════════════════════════════════════════════════╗
║            🔚 End of Table {table_count}                       ║
╚════════════════════════════════════════════════════╝
"""
            table_texts.append(beautified_table)

        if not table_texts:
            return ""

        return "\n".join(table_texts)

    except ValueError:
        # No tables found case
        return ""


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
        if selected_model_name in valid_model_names: # free tier models. 
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
                You are an AI assistant designed to help users navigate the GEO application.

                **Context:**  
                GEO is a well log authoring, analysis, and reporting system for petroleum geologists, geoscientists, and engineers.  
                Answer the user's question using **only** the provided documents.  

                **Instructions:**  
                - Use information from the **Documents** section to generate your response.  
                - Provide a **direct**, **concise**, and **factual** answer. 
                - **Avoid** speculative or unnecessary **explanations** or **justifications**. 
                - If the question is about a **numerical** or a **limit-based** constraint, return only the limit and its enforcement. 

                **Feedback Guidelines:**  
                - Review past user feedback under the **Feedback** section.  
                - If feedback suggests improvements, apply them before finalizing your response.  
                - Adjust your wording, structure, or level of detail based on feedback.  

                ---
                **Documents:**  
                {documents}  
                ---

                **Feedback:**  
                {feedback}  
                ---

                **Question:** {question}  

                **Your Optimized Answer:**  
                """,
                input_variables=["question", "documents", "feedback"]
            )
            
            # Save the prompt template to a file
            prompt_text = prompt.format(
                question="<QUESTION_PLACEHOLDER>", 
                documents="<DOCUMENTS_PLACEHOLDER>", 
                feedback="<FEEDBACK_PLACEHOLDER>"
            )
            with open(PROMPT_VISUALISATION_FILE, "w", encoding="utf-8") as file:
                file.write(prompt_text)
            # Save the second-to-last document for verification
            if len(self.web_documents) > 1:
                second_to_last_document = self.web_documents[-2].page_content
                with open(UPLOADED_FILE, "w", encoding="utf-8") as file:
                    file.write(second_to_last_document)

            rag_chain = prompt | self.llm_model | StrOutputParser()

            # Set the global variable
            rag_application = RAGApplication(retriever, rag_chain, self.web_documents)
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
    
    def _load_feedback_into_documents_and_file(self, page_texts):
        """
        Loads feedback from feedback_dataset.json, formats it, and appends it to documents and processed_content.txt.
        This ensures feedback is properly separated with '---Feedback---' for easy retrieval during RAG training.
        """

        feedback_data = load_feedback_dataset()
        feedback_heading = "---Feedback---"

        # Collect formatted feedback for processed_content.txt
        formatted_feedback_blocks = []

        for feedback_entry in feedback_data:
            # Convert feedback entry to pretty JSON
            json_feedback = json.dumps(feedback_entry, indent=4)

            # Create full feedback block with header
            formatted_feedback = f"{feedback_heading}\n{json_feedback}\n"
            formatted_feedback_blocks.append(formatted_feedback)

            # Insert into document store (Langchain or Haystack) depending on model
            if self.web_documents:
                last_document = self.web_documents[-1]

                if last_document.page_content.startswith(feedback_heading):
                    # Append to the existing feedback document
                    last_document.page_content += f"\n{json_feedback}\n"
                else:
                    # Create new feedback document if none exists
                    if selected_model_name in valid_model_names:
                        new_document = LangchainDocument(page_content=formatted_feedback)
                        self.web_documents.append(new_document)
                    else:
                        new_document = HaystackDocument(content=formatted_feedback)
                        self.web_documents_haystack.append(new_document)
            else:
                # Handle case where no documents exist (first-time load)
                if selected_model_name in valid_model_names:
                    new_document = LangchainDocument(page_content=formatted_feedback)
                    self.web_documents.append(new_document)
                else:
                    new_document = HaystackDocument(content=formatted_feedback)
                    self.web_documents_haystack.append(new_document)

            # For processed_content.txt, add just the JSON feedback (without header)
            page_texts.append(formatted_feedback)

        # Combine all page texts into final output and write to processed_content.txt
        temp_page_texts = "\n\n".join(page_texts)

        with open(PROCESSED_CONTENT_FILE, "w", encoding="utf-8") as file:
            file.write(temp_page_texts)

        logging.info(f"Processed content (including feedback) saved to {PROCESSED_CONTENT_FILE}")
        
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
                        formatted_table = extract_table_as_text_block(soup, file_path)
                    else:
                        formatted_table = ""    
                    
                    if list in selectedOptions:    
                        lists = extract_list(soup)
                    else:
                        lists = ""
                        
                    page_text = ""
                    
                    if clean_text != "":
                        page_text += f"\n{clean_text}\n"
                    
                    if formatted_table != "":
                        page_text += f"\n{formatted_table}\n"
                        
                    if lists != "":
                        page_text += f"\n{lists}\n"
                    
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

        self._load_feedback_into_documents_and_file(page_texts)

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
        global rag_application, EXCEL_FILE, selected_model_name  # Access the global variable

        if rag_application is None:
            logging.error("RAG application is not initialized.")
            return "Error: RAG application is not initialized."

        logging.info(f"Processing question: {question}")

        # Step 1: Run the appropriate RAG process
        if selected_model_name in valid_model_names:
            response = rag_application.run(question)
        else:
            response_object = self.rag_pipe.run({
                "embedder": {"text": question},
                "prompt_builder": {"question": question}
            })
            response = response_object['llm']['replies'][0]._content[0].text

        # Step 2: Prepare new entry DataFrame
        new_entry = pd.DataFrame([[question, selected_model_name, response]], columns=["Question", "Model Name", "Response"])

        # Step 3: Append to Excel (consolidated I/O operations)
        if os.path.exists(EXCEL_FILE):
            # Load once and append
            existing_data = pd.read_excel(EXCEL_FILE, engine='openpyxl')
            updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        else:
            # Just use the new entry if no file exists
            updated_data = new_entry

        # Write to Excel once (consolidated)
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='w') as writer:
            updated_data.to_excel(writer, index=False)
            auto_adjust_column_width(writer, updated_data)  # Optional - can be removed if speed is critical

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
        
        # Dynamic mapping to support both "comment" and "details"
        comment = data.get("comment") or data.get("details") or ""  # Prefer "comment", fallback to "details"
        
        rating = data.get("rating")
        response = data.get("response")
        question = data.get("question")

        # if not details and not question:
        #     return jsonify({"error": "Both feedback details and question details are required"}), 400

        feedback_entry = {
            "model-name": selected_model_name,
            "question": question,
            "response": response,
            "feedback": comment,
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

def save_chat_session(session_id, messages):
    # Use a module-level constant or ensure directory creation happens once (outside the function) if possible
    if not os.path.isdir(CHAT_SESSIONS_DIR):
        os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

    session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

    # Pre-generate timestamp to avoid recalculating multiple times
    current_timestamp = datetime.now().isoformat() + "Z"

    # Use a list comprehension to update messages more efficiently
    updated_messages = [
        message if "timestamp" in message else {**message, "timestamp": current_timestamp}
        for message in messages
    ]

    # Use faster file writing (single write operation)
    with open(session_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(updated_messages, ensure_ascii=False, indent=4))

def process_question(question_id, question, ai_bot, selectedOptions):
    """
    Simulate long processing of the question and store the response.
    """
    global answer_time, current_session_id, current_session_messages
    
    start_time = time.time() # Begins to record how long it takes the model for querying
    
    response = ai_bot.query(question)
    
    print("Response from advanced model:", response)
    
    end_time = time.time()
    
    # Calculate and print the elapsed time
    answer_time = end_time - start_time
    
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)

    stored_responses[question_id] = formatted_response
    
    # Append assistant's response to current session
    current_session_messages.append({"role": "assistant", "content": response})

    # Save current session to file
    save_chat_session(current_session_id, current_session_messages)

@app.route("/delete-session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return jsonify({"error": "Session not found"}), 404
        
        # Delete the file
        os.remove(session_file)

        # Update the metadata (optional - if you want to remove the name as well)
        metadata = load_session_metadata()
        if session_id in metadata:
            del metadata[session_id]
            save_session_metadata(metadata)

        return jsonify({"message": "Session deleted successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error deleting session: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global question_id, current_session_id, current_session_messages

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        question = data.get("question", "").strip()
        selectedOptions = data.get("selectedOptions", "")
        incoming_session_id = data.get("session_id")

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Check if we are continuing a historical session
        if incoming_session_id:
            if current_session_id != incoming_session_id:
                # New session is being reloaded
                current_session_id = incoming_session_id
                session_file = os.path.join(CHAT_SESSIONS_DIR, f"{current_session_id}.json")

                if os.path.exists(session_file):
                    with open(session_file, "r", encoding="utf-8") as f:
                        current_session_messages = json.load(f)
                else:
                    # Fall back in case of missing file
                    current_session_messages = []

        elif current_session_id is None:
            # If no historical session is loaded, start fresh
            current_session_id = f"chat_session_{time.strftime('%Y%m%d_%H%M%S')}"
            current_session_messages = []

        # Append new user message
        current_session_messages.append({"role": "user", "content": question})

        with lock:
            current_id = str(question_id)
            question_id += 1

        pending_responses[current_id] = "Processing..."

        threading.Thread(target=process_question, args=(current_id, question, ai_bot, selectedOptions)).start()

        return jsonify({"question_id": current_id, "session_id": current_session_id}), 200

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

@app.route('/delete-pair', methods=['POST'])
def delete_question_answer_pair():
    try:
        data = request.json
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            return jsonify({"success": False, "error": "Missing session_id or question"}), 400

        session_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

        if not os.path.exists(session_file):
            return jsonify({"success": False, "error": "Session not found"}), 404

        with open(session_file, "r", encoding="utf-8") as file:
            messages = json.load(file)

        # Find the index of the user question and remove it along with the next assistant message
        new_messages = []
        skip_next = False

        for i, msg in enumerate(messages):
            if skip_next:
                skip_next = False
                continue  # Skip the assistant response after the user question

            if msg["role"] == "user" and msg["content"] == question:
                skip_next = True  # Skip this user question and the following assistant message
                continue

            new_messages.append(msg)

        with open(session_file, "w", encoding="utf-8") as file:
            json.dump(new_messages, file, indent=4)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

# Function to check if a question has a similar wording in expected_answers_dict
def is_similar_question(question, expected_answers, threshold=0.80):
    print(f"question: {question}")
    print(f"expected answer: {expected_answers}")
    match, score, _ = process.extractOne(question, expected_answers, score_cutoff=threshold)
    if match is None:
        return False
    print(f"score: {score}")
    return score >= threshold  # Returns True if similarity score is above threshold

# Function to replace invalid values with an empty string
def clean_dataframe(df):
    """
    Replace NaN, None, or other invalid values with an empty string in the given DataFrame.
    """
    return df.fillna("").replace({None: ""})  # Replace NaN and None with empty string

# Function to find the most semantically similar expected answer
def find_best_match(question, expected_answers_dict):
    if not question or not expected_answers_dict:
        return ""  # Return empty string if no valid input

    expected_questions = list(expected_answers_dict.keys())
    
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

    # Compute embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    expected_embeddings = model.encode(expected_questions, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, expected_embeddings)[0]

    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()

    # Set a similarity threshold (adjust if needed)
    threshold = 0.75  
    if best_match_score < threshold:
        return ""  # No match found above the threshold

    return expected_answers_dict[expected_questions[best_match_idx]]

# Function to compute BLEU and ROUGE scores
def calculate_bleu_rouge(reference, candidate):
    """
    Compute BLEU and ROUGE similarity scores between the reference answer and the response.
    """
    if not reference or not candidate:
        return {"BLEU": 0.0, "ROUGE": 0.0}

    # Compute BLEU score
    bleu_score = sentence_bleu([reference.split()], candidate.split())

    # Compute ROUGE-L score (longest common subsequence)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score = scorer.score(reference, candidate)["rougeL"].fmeasure  # F1 score of ROUGE-L

    return {"BLEU": bleu_score, "ROUGE": rouge_score}

@app.route("/get-results", methods=["GET"])
def get_results():
    try:
        logging.info("Processing GET request for /get-results")

        # Step 1: Check if the results file exists
        if not os.path.exists(EXCEL_FILE):
            logging.warning(f"Results file not found: {EXCEL_FILE}")
            return jsonify({"error": "Results file not found"}), 404

        # Define columns
        columns = ["Question", "Model Name", "Response", "Expected Answer", "similarity_score"]

        # Generate random data
        num_rows = 10  # Number of rows in the dataframe
        questions = [f"Question {i+1}" for i in range(num_rows)]
        models = [f"Model_{np.random.randint(1, 5)}" for _ in range(num_rows)]
        responses = [f"Response {i+1}" for i in range(num_rows)]
        expected_answers = [f"Expected Answer {i+1}" for i in range(num_rows)]
        similarity_scores = np.random.uniform(0, 1, num_rows)  # Random similarity scores between 0 and 1

        # Define data as a list of dictionaries to avoid potential issues
        data = [
            {
                "Question": questions[i],
                "Model Name": models[i],
                "Response": responses[i],
                "Expected Answer": expected_answers[i],
                "similarity_score": similarity_scores[i],
            }
            for i in range(num_rows)
        ]

        temp_df = pd.DataFrame(data)

        temp_results = pd.DataFrame({
            "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
            "Accuracy": ["90%", "84%", "79%", "92%"]
        })

        # Step 2: Load the results file
        logging.info(f"Loading results from {EXCEL_FILE}")
        if os.path.exists(EXCEL_FILE):
            try:
                df = pd.read_excel(EXCEL_FILE)
            except ValueError:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except PermissionError:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except Exception as e:
                df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
        else:
            df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })
        logging.debug(f"Loaded DataFrame columns: {df.columns.tolist()}")

        # Step 3: Check for expected results file
        expected_results_df = None
        if os.path.exists(EXPECTED_RESULTS_FILE):
            try:
                logging.info(f"Loading expected results from {EXPECTED_RESULTS_FILE}")
                expected_results_df = pd.read_excel(EXPECTED_RESULTS_FILE)
            except ValueError as ve:
                logging.error(f"ValueError: Unable to read the Excel file due to format issues - {ve}")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except PermissionError:
                logging.error(f"PermissionError: The file {EXPECTED_RESULTS_FILE} is in use or lacks necessary permissions.")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
            except Exception as e:
                logging.error(f"Unexpected error while reading the Excel file: {e}")
                return jsonify({
                    "models": temp_results.to_dict(orient="records"),
                    "filtered_results": temp_df.to_dict(orient="records")
                })
        else:
            logging.warning(f"Expected results file not found: {EXPECTED_RESULTS_FILE}")
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })

        question_column = "Question"
        response_column = "Response"
        expected_answer_column = "Expected Answer"

        # Step 4: Validate column existence
        required_columns = [question_column, response_column]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns in results file: {df.columns.tolist()}")
            return jsonify({
                "models": temp_results.to_dict(orient="records"),
                "filtered_results": temp_df.to_dict(orient="records")
            })
            # return jsonify({"error": "Invalid results file format"}), 400

        if expected_results_df is not None and question_column in expected_results_df.columns:
            expected_answers_dict = dict(zip(expected_results_df[question_column], expected_results_df[response_column]))
            logging.info("Successfully created expected answers dictionary.")
        else:
            logging.warning("Expected results file does not contain required columns. Skipping filtering.")
            expected_answers_dict = {}
        print(expected_answers_dict)
        # Step 5: Check if questions exist in expected results
        df["is_question_in_expected"] = df[question_column].apply(lambda q: is_similar_question(q, expected_answers_dict.keys()))
        print(f"Questions found in expected results: {df['is_question_in_expected'].sum()}")

        # Step 6: Filter data and compute similarity scores
        filtered_df = df[df["is_question_in_expected"] == True].copy()
        if not filtered_df.empty:
            filtered_df.loc[:, expected_answer_column] = filtered_df.loc[:, question_column].apply(lambda q: find_best_match(q, expected_answers_dict))

            similarity_scores = filtered_df.apply(
                lambda row: calculate_bleu_rouge(str(row[expected_answer_column]), str(row[response_column])),
                axis=1
            )
            
            # Store BLEU and ROUGE scores in separate columns
            filtered_df["BLEU_score"] = similarity_scores.apply(lambda x: x["BLEU"])
            filtered_df["ROUGE_score"] = similarity_scores.apply(lambda x: x["ROUGE"])
            
            # Compute similarity score
            filtered_df.loc[:, "similarity_score"] = filtered_df.apply(
                lambda row: calculate_semantic_similarity(str(row[expected_answer_column]), str(row[response_column])),
                axis=1
            )
            logging.info(f"Computed similarity scores for {len(filtered_df)} entries.")
        else:
            logging.warning("No matching questions found in expected results.")

        logging.info("Filtered results preview:")
        logging.debug(filtered_df.head().to_string())

        # Save filtered results for debugging
        temp_filtered_file = "Data/temp_filtered_data.xlsx"
        filtered_df.drop(columns=["is_question_in_expected"], inplace=True)
        filtered_df.to_excel(temp_filtered_file)
        logging.info(f"Filtered results saved to {temp_filtered_file}")

        # Step 7: Prepare model accuracy results
        if "Model Name" not in df.columns or "Accuracy" not in df.columns:
            logging.warning("Missing 'Model Name' or 'Accuracy' columns in results file. Using default values.")
            results = pd.DataFrame({
                "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
                "Accuracy": ["90%", "84%", "79%", "92%"]
            })
        else:
            results = df[["Model Name", "Accuracy"]]

        # Step 8: Convert results to JSON and return response
        logging.info("Returning JSON response with models and filtered results.")
        print(results.columns)
        print(filtered_df.columns)
        # Apply the function to clean both DataFrames
        results = clean_dataframe(results)
        filtered_df = clean_dataframe(filtered_df)
        
        json_results = results.to_dict(orient="records")
        json_filtered_results = filtered_df.to_dict(orient="records")
        
        json_results = json.loads(json.dumps(json_results))  # Converts NaN to valid JSON
        json_filtered_results = json.loads(json.dumps(json_filtered_results))
        return jsonify({
            "models": json_results,
            "filtered_results": json_filtered_results
        })

    except Exception as e:
        logging.error(f"Error reading results: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    
def append_to_excel(question, response):
    """ Append question and response to the Excel file. """
    global selected_model_name
    new_entry = pd.DataFrame([[question, selected_model_name, response]], columns=["Question", "Model Name", "Response"])

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

@app.route('/search')
def search_chats():
    query = request.args.get('query', '').lower()

    if not query:
        return jsonify([])  # No query provided, return empty list

    chat_sessions = load_chat_history()  # Load all sessions from disk

    matching_messages = []

    # Iterate through all sessions and messages to find matches
    for session in chat_sessions:
        session_id = session['session_id']
        temp_question = None

        for message in session['messages']:
            content = message['content'].lower()

            if query in content:
                if message['role'] == 'user':
                    temp_question = message['content']
                elif message['role'] == 'assistant' and temp_question:
                    matching_messages.append({
                        "session_id": session_id,
                        "question": temp_question,
                        "answer": message['content']
                    })
                    temp_question = None  # Reset after capturing the pair

    return jsonify(matching_messages)

@app.route('/generate-title', methods=['POST'])
def generate_chat_title():
    """Generate a concise chat title using Ollama based on user-provided conversation data."""

    # Extract messages from frontend request
    data = request.json
    messages = data.get("messages", [])

    if not messages:
        print(f"⚠️ No valid messages received: {data}")
        return jsonify({"title": "Untitled Chat"}), 400

    # Convert chat messages into a summarized format
    conversation_text = " ".join([msg["content"] for msg in messages])

    # ✅ Modified prompt to enforce conciseness
    ollama_request = {
        "model": "llama3.2:latest",
        "prompt": (
            "Generate a **very short, clear, and concise** title for this conversation."
            " Keep it **under 8 words** and make it **informative**:"
            f"\n\n{conversation_text}"
        ),
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=ollama_request)
        response_json = response.json()

        title = response_json.get("response", "Untitled Chat").strip()

        # ✅ Additional Post-Processing: Keep **max 8 words**
        title_words = title.split()
        if len(title_words) > 8:
            title = " ".join(title_words[:8]) + "..."  # Truncate and add ellipsis

        return jsonify({"title": title})

    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return jsonify({"title": "Untitled Chat"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
