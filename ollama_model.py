import openai
import os
import logging
import pandas as pd
from bs4 import BeautifulSoup
# from ChatOllama import ChatOllama  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_index.readers.file.flat import FlatReader
from llama_index.core.node_parser.relational import UnstructuredElementNodeParser 
import nest_asyncio
from pathlib import Path
import pickle
import uuid

nest_asyncio.apply()

reader = FlatReader()
node_parser = UnstructuredElementNodeParser()

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

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

    def _load_content(self):
        """
        Load and process all .htm files from the base directory.
        """
        htm_files = self._list_htm_files()
        print(f"Found {len(htm_files)} .htm files.")

        for file_path in htm_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                text = soup.get_text(separator="\n").strip()  # Extract text content from HTML

                if not text:
                    logging.warning(f"Skipping {file_path} as it contains no readable text.")
                    continue  # Skip empty files

                directory = os.path.dirname(file_path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                if file_path.endswith("GEO_Limits.htm"):
                    print(f"File name: \"{file_name}.\"")

                document = Document(page_content=text)  # Wrap extracted text in a Document
                
                if not os.path.exists(directory + "/" + file_name + ".pkl"):
                    raw_nodes = node_parser.get_nodes_from_documents([document])  # Ensure input is a list
                    pickle.dump(raw_nodes, open(directory + "/" + file_name + ".pkl", "wb"))
                else:
                    raw_nodes = pickle.load(open(directory + "/" + file_name + ".pkl", "rb"))
                
                base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(
                    raw_nodes
                )
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the file encoding.")
            except Exception as e:
                continue

        print("Node initialization is successful.")

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
    
    def update_training(self, data_string):
        
        print(f"Received Feedback: \n\"{data_string}\"\n")
        data_document = Document(
            page_content=data_string
        )
        
        self.web_documents.append(data_document)

ai_bot = OllamaBot()
ai_bot._load_content()