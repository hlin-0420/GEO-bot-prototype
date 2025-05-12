from app.services.rag_application import RAGApplication
from app.utils.file_helpers import auto_adjust_column_width
from app.utils.html_file_loader import process_single_file
# from app.services.document_processor.DocumentProcessor import extract_text, extract_list, extract_table_as_text_block
from app.config import (
    PROCESSED_CONTENT_FILE,
    EXCEL_FILE,
    selected_model_name,
    valid_model_names
)
from app.services.ollama_bot_helpers import (
    list_htm_files,
    get_prompt_template
)

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
import pandas as pd
import os
import logging
import time
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class OllamaBot:
    def __init__(self):
        """
        Initialize the OllamaBot with the specified model.
        """
        try:
            print("🚀 [OllamaBot] Starting initialization...")
            init_start = time.time()

            # Step 1: Resolve base directory
            self.base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
            print(f"📁 [OllamaBot] Base directory set to: {self.base_directory}")

            # Step 2: Create document cache
            self.web_documents = []
            print("📚 [OllamaBot] Initialized empty document cache.")

            # Step 3: Initialize LLM
            model_init_start = time.time()
            self.llm_model = ChatOllama(
                model=selected_model_name,
                temperature=0,
                num_predict=150
            )
            print(f"🤖 [OllamaBot] LLM model '{selected_model_name}' loaded in {time.time() - model_init_start:.2f} seconds.")

            # Step 4: Initialize RAG pipeline
            rag_init_start = time.time()
            self.refresh()
            print(f"🔄 [OllamaBot] RAG pipeline refreshed in {time.time() - rag_init_start:.2f} seconds.")

            total_time = time.time() - init_start
            print(f"✅ [OllamaBot] Initialization completed in {total_time:.2f} seconds.\n")

        except Exception as e:
            print(f"❌ [OllamaBot] Initialization failed: {str(e)}")
            raise
        
    def _initialize_rag_application(self):
        print("⚙️ [_initialize_rag_application] Starting RAG setup...")

        try:
            # Step 1: Split documents
            print("✂️ Splitting documents...")
            split_start = time.time()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=20
            )
            doc_splits = text_splitter.split_documents(self.web_documents)
            print(f"✅ Document split into {len(doc_splits)} chunks in {time.time() - split_start:.2f} seconds.")

            # Step 2: Load embedding model
            print("🧠 Initializing embedding model (OllamaEmbeddings)...")
            embed_start = time.time()
            print("🧠 Using all-MiniLM-L6-v2 from HuggingFace for embeddings")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print(f"✅ Embedding model initialized in {time.time() - embed_start:.2f} seconds.")

            # Step 3: Build vector store
            print("📦 Creating vector store from document chunks...")
            vector_start = time.time()
            vectorstore = SKLearnVectorStore.from_documents(
                documents=doc_splits,
                embedding=embedding_model,
            )
            print(f"✅ Vector store created in {time.time() - vector_start:.2f} seconds.")

            retriever = vectorstore.as_retriever(k=3)

            # Step 4: Build prompt
            if selected_model_name in valid_model_names:
                print("📜 Creating prompt template and RAG chain...")
                prompt = get_prompt_template()
                rag_chain = prompt | self.llm_model | StrOutputParser()
                self.rag_application = RAGApplication(retriever, rag_chain, self.web_documents)
                print("✅ RAG application initialized successfully.")
            else:
                raise ValueError(f"Model '{selected_model_name}' is not supported in this configuration.")
        except Exception as e:
            print(f"❌ [_initialize_rag_application] Failed: {str(e)}")
            raise
        
    def _load_content(self, selectedOptions=None):
        """
        Load and process all .htm files from the base directory.
        """
        htm_files = list_htm_files(self.base_directory)
        logging.info(f"Found {len(htm_files)} .htm files.")
        selectedOptions = selectedOptions or ["text", "table", "list"]
        # initialise empty training web documents.
        self.web_documents = []

        for file_path in htm_files:
            try:
                doc = process_single_file(file_path, selectedOptions)
                
                if doc:
                    self.web_documents.append(doc)
                    
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")

        logging.info(f"Processed content saved to {PROCESSED_CONTENT_FILE}")

    def add(self, content):
        """
        Add new content to the bot's memory.
        
        Args:
            content (str): Content to add.
        """
        feedback_heading = "---Feedback---"
        
        new_document = LangchainDocument(page_content=content)
                
        if self.web_documents:
            last_document = self.web_documents[-1]
            
            if last_document.page_content.startswith(feedback_heading):
                # Ensure there is at least one more document before inserting
                if len(self.web_documents) > 1:
                    self.web_documents.insert(len(self.web_documents) - 1, new_document)
                else:
                    self.web_documents.insert(0, new_document)
            else:
                self.web_documents.append(new_document)
            logging.info("New content added.")
            
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
        if self.rag_application is None:
            logging.error("RAG application is not initialized.")
            return "Error: RAG application is not initialized."

        logging.info(f"Processing question: {question}")

        # Step 1: Run the appropriate RAG process
        rag_start_time = time.time()  # Start timing RAG application run
        response = self.rag_application.run(question)  # Actual function execution
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
    
    def refresh(self):
        print("🔁 [refresh] Starting document loading and RAG setup...")
        self._load_content()

        if not self.web_documents:
            logging.warning("⚠️ No documents loaded, skipping RAG initialization.")
            return

        self._initialize_rag_application()
