import os
import json
import logging
import time
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
import pandas as pd

from app.services.rag_application import RAGApplication
from app.utils.file_helpers import auto_adjust_column_width
from app.services.document_processor import DocumentProcessor
# from app.services.document_processor.DocumentProcessor import extract_text, extract_list, extract_table_as_text_block
from app.utils.feedback import load_feedback_dataset
from app.config import (
    PROCESSED_CONTENT_FILE,
    FEEDBACK_FILE,
    EXCEL_FILE,
    selected_model_name,
    valid_model_names
)

class OllamaBot:
    def __init__(self):
        """
        Initialize the OllamaBot with the specified model.
        
        Args:
            model_name (str): Name of the Ollama model.
            base_directory (str): Path to the base directory containing .htm files.
        """
        global valid_model_names
        ##########################################
        # Storage Processing
        # Data Directory initialisation
        self.base_directory = "Data"
        self.web_documents = [] # stores the web documents for free tier models.
        self._load_content() 
        ####################
        # Pipeline initialisation.
        self.llm_model = ChatOllama(
            model=selected_model_name,
            temperature=0,
            num_predict=150
        ) # initialises a free-tier model.
        # Initialize RAG application globally
        self._initialize_rag_application() # Generalised rag pipeline initialisation.
        
    def _initialize_rag_application(self):
        """
        Initializes the RAGApplication globally using LangChain components only.
        """
        global rag_application

        # Step 1: Split web documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=20
        )
        doc_splits = text_splitter.split_documents(self.web_documents)

        # Step 2: Load the offline embedding model
        embedding_model = OllamaEmbeddings(model="llama3.2:latest")

        # Step 3: Create vector store and retriever
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
        )
        retriever = vectorstore.as_retriever(k=3)

        # Step 4: Define prompt template for supported models
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
                - If the past feedback **corrects** a numerical limit, interpret and apply the correct value.  

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

            # Create the RAG chain
            rag_chain = prompt | self.llm_model | StrOutputParser()

            # Set the global application object
            rag_application = RAGApplication(retriever, rag_chain, self.web_documents)

        else:
            raise ValueError(f"Model '{selected_model_name}' is not supported in this configuration.")

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
                    new_document = LangchainDocument(page_content=formatted_feedback)
                    self.web_documents.append(new_document)
            else:
                # Handle case where no documents exist (first-time load)
                new_document = LangchainDocument(page_content=formatted_feedback)
                self.web_documents.append(new_document)

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
                        clean_text = DocumentProcessor.extract_text(soup)
                    else:
                        clean_text = "" # when the text, table, or list is empty. 
                    
                    if "table" in selectedOptions:
                        formatted_table = DocumentProcessor.extract_table_as_text_block(soup, file_path)
                    else:
                        formatted_table = ""    
                    
                    if list in selectedOptions:    
                        lists = DocumentProcessor.extract_list(soup)
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
                    
                    page_name = os.path.basename(file_path)  # Extracts the file name, e.g., "example.htm"
                    
                    document = LangchainDocument(
                        page_content=page_data['text'],
                        metadata={
                            'links': page_data['link'],
                            'page_name': page_name
                        }
                    )
                    self.web_documents.append(document)
                    
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
        
        new_document = LangchainDocument(page_content=content)
        temp_documents = self.web_documents
        
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
            
            self.web_documents = temp_documents
            
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
        rag_start_time = time.time()  # Start timing RAG application run
        response = rag_application.run(question)  # Actual function execution
        rag_execution_time = time.time() - rag_start_time  # Measure RAG execution time
            
        print(f"⏱️ The time taken to implement the `run` function of the RAG application is {rag_execution_time:.4f} seconds.")

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