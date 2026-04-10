# app/services/ollama_bot_helpers.py

import os
import json
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LangchainDocument
from app.utils.feedback import load_feedback_dataset
from app.config import PROCESSED_CONTENT_FILE

def list_htm_files(base_directory):
    htm_files = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".htm"):
                relative_path = os.path.join(root, file)
                htm_files.append(relative_path)
    return htm_files

def get_prompt_template():
    return PromptTemplate(
        template="""
        You are an AI assistant designed to help users navigate the GEO application.

        **Context:**  
        GEO is a well log authoring, analysis, and reporting system for petroleum geologists, geoscientists, and engineers.  
        Answer the user's question using **only** the provided documents.  

        **Instructions:**  
        - Use information from the **Documents** section to generate your response.  
        - Generate **up to 3 alternative answer options** for the same question.
        - Provide a **direct**, **concise**, and **factual** answer for each option.
        - **Avoid** speculative or unnecessary **explanations** or **justifications**. 
        - If the question is about a **numerical** or a **limit-based** constraint, return only the limit and its enforcement. 
        - If the past feedback **corrects** a numerical limit, interpret and apply the correct value.
        - Return output in this exact structure:
          Option 1: <answer>
          Option 2: <answer>
          Option 3: <answer>
        - If you can only provide one strong answer, still return:
          Option 1: <answer>

        **Feedback Guidelines:**  
        - Review past user feedback under the **Feedback** section.  
        - If feedback suggests improvements, apply them before finalizing your response.  
        - Adjust your wording, structure, or level of detail based on feedback.  

        ---
        **Documents:**  
        {documents}  
        ---

        **Question:** {question}  

        **Your Optimized Answers:**  
        """,
        input_variables=["question", "documents"]
    )