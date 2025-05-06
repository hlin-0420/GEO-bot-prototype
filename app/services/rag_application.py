import os
import json
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback", "feedback_dataset.json")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "local_models", "offline_model")

class RAGApplication:
    def __init__(self, retriever, rag_chain, web_documents):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.web_documents = web_documents  # Store the documents for feedback retrieval
        self.feedback_model = SentenceTransformer(MODEL_PATH)  # Embedding model for similarity
        self.feedback_data, self.feedback_embeddings = self._load_feedback()

    def _load_feedback(self):
        """Loads feedback from file and precomputes embeddings to optimize retrieval."""
        if not os.path.exists(FEEDBACK_FILE):
            logging.warning("⚠️ No feedback file found.")
            return [], []

        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
                feedback_data = json.load(file)  # Load feedback JSON array
        except json.JSONDecodeError:
            logging.error("⚠️ Error decoding feedback JSON file. Returning empty feedback.")
            return [], []

        extracted_feedback = [
            {
                "question": entry["question"],
                "feedback": entry["feedback"],
                "rating": int(entry.get("rating-score", 0))
            }
            for entry in feedback_data if "question" in entry and "feedback" in entry
        ]

        if not extracted_feedback:
            logging.warning("⚠️ No valid feedback extracted.")
            return [], []

        # Compute embeddings in parallel
        with ThreadPoolExecutor() as executor:
            feedback_embeddings = list(executor.map(
                lambda fb: self.feedback_model.encode(fb["question"], convert_to_tensor=True),
                extracted_feedback
            ))

        return extracted_feedback, feedback_embeddings
        
    def _get_relevant_feedback(self, question, top_k=3):
        """Retrieve the most relevant feedback based on semantic similarity."""
        if not self.feedback_data:
            return ""

        # Compute embedding for the new question
        question_embedding = self.feedback_model.encode(question, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = np.array([
            util.pytorch_cos_sim(question_embedding, fb_emb)[0].item()
            for fb_emb in self.feedback_embeddings
        ])

        # Get indices of top-k similar feedback
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Extract unique questions while maintaining order
        selected_feedback = []
        unique_questions = set()

        for idx in top_indices:
            fb = self.feedback_data[idx]
            base_question = fb["question"].lower().strip("?")
            if base_question not in unique_questions:
                selected_feedback.append(fb["feedback"])
                unique_questions.add(base_question)
            if len(selected_feedback) >= top_k:
                break

        return "\n".join(selected_feedback) if selected_feedback else ""

    def run(self, question):
        """Runs the RAG retrieval and generates a response with detailed runtime analysis."""
        
        total_start_time = time.perf_counter()  # Start total execution timer
        
        # Step 1: Retrieve relevant documents
        retrieval_start_time = time.perf_counter()
        documents = self.retriever.invoke(question)
        retrieval_end_time = time.perf_counter()
        retrieval_time = retrieval_end_time - retrieval_start_time

        doc_texts = "\n".join(doc.page_content for doc in documents)

        # Step 2: Retrieve relevant feedback
        feedback_start_time = time.perf_counter()
        feedback_texts = self._get_relevant_feedback(question)
        feedback_end_time = time.perf_counter()
        feedback_time = feedback_end_time - feedback_start_time

        if not feedback_texts.strip():
            logging.warning("⚠️ No feedback found for this query.")

        # Step 3: Generate the answer using the updated prompt format
        response_start_time = time.perf_counter()
        response = self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts,
            "feedback": feedback_texts,
            "stream": True
        })
        response_end_time = time.perf_counter()
        response_time = response_end_time - response_start_time

        total_end_time = time.perf_counter()
        total_execution_time = total_end_time - total_start_time

        # Logging detailed runtime analysis
        logging.info(f"🕒 RAG Execution Time Breakdown:")
        logging.info(f"   - Document Retrieval Time: {retrieval_time:.4f} seconds")
        logging.info(f"   - Feedback Extraction Time: {feedback_time:.4f} seconds")
        logging.info(f"   - Response Generation Time: {response_time:.4f} seconds")
        logging.info(f"   - Total Execution Time: {total_execution_time:.4f} seconds")

        return response