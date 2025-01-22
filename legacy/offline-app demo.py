from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

template = """
Answer questions for users who wanted to look for help from the GEO help Guide.

Question: {question}

As a GEO help guide, I can help you with the following topics:
{topics}

Answer: 
"""

model = OllamaLLM(model = "llama3")
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model # chain the operations together.

topics = "Curve Data, MWD, LWD"

def handle_conversation(topics):

    print("Welcome to the GEO Help Guide!")

    while True:
        user_input = input("Please enter your question or type 'exit' to quit: ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break

        result = chain.invoke({"question": user_input, "topics": topics})
        print("Bot: ")
        print(result)

def _load_content():
        """
        Load and process all .htm files from the base directory,
        including files in nested directories. Also, list all folders and files.
        """
        base_directory = os.getcwd()  # Ensure this is set to your root directory
        logging.info(f"Starting to load content from base directory: {base_directory}")
        
        all_folders = []
        all_files = []
        htm_files = []

        try:
            # Walk through the directory structure
            for root, dirs, files in os.walk(base_directory):
                all_folders.append(root)  # Collect all folders
                all_files.extend([os.path.join(root, file) for file in files if file.endswith(".htm")])  # Collect all files

                # Filter .htm files specifically
                htm_files.extend([os.path.join(root, file) for file in files if file.endswith(".htm")])

            logging.info(f"Total folders found: {len(all_folders)}")
            logging.info(f"Total files found: {len(all_files)}")
            logging.info(f"Total .htm files found: {len(htm_files)}")

        except Exception as e:
            logging.error(f"An error occurred while traversing the directory: {e}")

        relative_paths = []

        for file_path in all_files:
            # remove the first part of the path to get the relative path
            relative_path = file_path.replace(base_directory, "")

            # remove the first slash if it exists
            if relative_path.startswith(("/", "\\")):
                relative_path = relative_path[1:]

            print(f"Processing file: {relative_path}")

            relative_paths.append(relative_path)

        return relative_paths

relative_paths = _load_content()
print(relative_paths)
# handle_conversation(topics)