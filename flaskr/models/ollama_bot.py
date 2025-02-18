# Standard Libraries
import os
import logging
import json

# Third-Party Libraries
import pandas as pd
from bs4 import BeautifulSoup

# Machine Learning and NLP Libraries

# OpenAI and Language Model-Related Libraries
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# import RAGApplication from custom defined rag_model.py
from .rag_model import RAGApplication

# import data processing functions from data_processing.file_processing
from ..data_processing.file_processing import extract_text, extract_table, extract_list

# import the load_feedback_dataset function to load feedback from data_loading.load_feedback_dataset
from ..data_processing.data_loading import load_feedback_dataset

from ..data_processing.excel_formatting import auto_adjust_column_width
from flask import Flask # imports global variable "stored_responses" from __init__.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

selected_model_name = "llama3.2:latest"
EXCEL_FILE = "query_responses.xlsx"

class OllamaBot:
    def __init__(self):
        """
        Initialize the OllamaBot with the specified model.
        
        Args:
            model_name (str): Name of the Ollama model.
            base_directory (str): Path to the base directory containing .htm files.
        """
        global selected_model_name
        self.base_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
        self.contents = []  # Store processed content
        self.web_documents = [] # stores the web documents
        self._load_content()
        
        app = Flask(__name__)
        with app.app_context():
            self.llm_model = ChatOllama(
                model=selected_model_name,
                temperature=0,
            )
            self.selected_model_name = selected_model_name
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
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
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
        # Save the second-to-last document for verification
        if len(self.web_documents) > 1:
            second_to_last_document = self.web_documents[-2].page_content
            with open("uploaded_document.txt", "w", encoding="utf-8") as file:
                file.write(second_to_last_document)

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
                    
                    if "list" in selectedOptions:    
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
        feedback_heading = "---Feedback---"
        
        new_document = Document(page_content=content)
        
        if self.web_documents:
            last_document = self.web_documents[-1]
            
            if last_document.page_content.startswith(feedback_heading):
                print("Inserting new uploaded document.")
                print(f"Length of the web documents: {len(self.web_documents)}")
                # Ensure there is at least one more document before inserting
                if len(self.web_documents) > 1:
                    self.web_documents.insert(len(self.web_documents) - 1, new_document)
                else:
                    self.web_documents.insert(0, new_document)
            else:
                self.web_documents.append(new_document)
            logging.info("New content added.")
            
            # Confirm the position of the new document
            inserted_index = self.web_documents.index(new_document)
            print(f"New document inserted at position: {inserted_index}")
            
            # Check if the index is in the second to last position
            if inserted_index == len(self.web_documents) - 2:
                print("New document correctly inserted in the second to last position.")
            else:
                print("New document is NOT in the expected position.")
            
        self._initialize_rag_application() # resets the initialisation to retrain model on updated information. 
            
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
        global rag_application, EXCEL_FILE  # Access the global variable

        if rag_application is None:
            logging.error("RAG application is not initialized.")
            return "Error: RAG application is not initialized."

        logging.info(f"Processing question: {question}")
        print(f"The selected Model Name for this training is {self.selected_model_name}")

        response = rag_application.run(question)
        
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