import os
import json
import logging
from sentence_transformers import SentenceTransformer, util

FEEDBACK_FILE = "feedback.json"

class RAGApplication:
    """
    A Retrieval-Augmented Generation (RAG) model that enhances responses by combining 
    retrieved GEO help guide documents with generative AI capabilities and user feedback.

    Attributes:
        retriever (object): Retrieves the most relevant documents based on user queries, ensuring context-aware responses.
        rag_chain (object): A language model trained with structured prompts to generate well-formatted and accurate answers.
        web_documents (list): A repository of GEO help guide documents indexed for retrieval.
        feedback_model (SentenceTransformer): A sentence transformer that encodes and ranks user feedback for similarity-based selection.
    """
    
    def __init__(self, retriever, rag_chain, web_documents):
        """
        A Retrieval-Augmented Generation (RAG) model that enhances responses by combining 
        retrieved GEO help guide documents with generative AI capabilities and user feedback.

        Args:
            retriever (object): Retrieves the most relevant documents based on user queries, ensuring context-aware responses.
            rag_chain (object): A language model trained with structured prompts to generate well-formatted and accurate answers.
            web_documents (list): A repository of GEO help guide documents indexed for retrieval.
        """
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.web_documents = web_documents  # Store the documents for feedback retrieval
        self.feedback_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model for similarity

    def _get_relevant_feedback(self, question, top_k=3):
        """
        Retrieves the top-k most relevant feedback entries based on semantic similarity and question type matching.

        This function selects feedback related to the user's query and past responses to enhance future answer accuracy.
        
        This function:
        1. Loads user feedback stored in a JSON file.
        2. Computes semantic similarity between the query and past feedback questions.
        3. Sorts and returns the most relevant feedback.

        Args:
            question (str): The user's query.
            top_k (int, optional): The number of top relevant feedback responses to return. Defaults to 3.

        Returns:
            str: The concatenated relevant feedback responses.
        """


        logging.info(f"🔍 Received question: {question}")

        # 🔹 Step 1: Parse feedback documents correctly (handling multiple JSON objects)
        logging.info("🔍 Parsing feedback documents...")

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
        logging.info(f"🔍 Computing embedding for question: {question}")
        question_embedding = self.feedback_model.encode(question, convert_to_tensor=True)

        # 🔹 Step 3: Compute similarity scores for each feedback entry based on the "question" field
        feedback_embeddings = [self.feedback_model.encode(fb["question"], convert_to_tensor=True) for fb in extracted_feedback]
        similarities = [util.pytorch_cos_sim(question_embedding, fb_emb)[0].item() for fb_emb in feedback_embeddings]

        # Log similarity scores for debugging
        for i, (fb, sim_score) in enumerate(zip(extracted_feedback, similarities)):
            logging.info(f"🔹 Feedback [{i}] - Question: {fb['question']} | Similarity Score: {sim_score}")

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
        logging.info("🔹 Selecting semantically similar feedback...")
        for fb, sim_score in sorted_feedback:
            selected_feedback.append(fb["feedback"])
            unique_questions.add(fb["question"].lower().replace("?", "").strip())  # Normalize the question type
            logging.info(f"✅ Added feedback: {fb['feedback']} | Similarity Score: {sim_score}")
            
            if len(selected_feedback) >= top_k:
                break

        # 🔹 Step 5.2: Ensure at least one feedback entry from the same question type exists
        logging.info("🔹 Ensuring at least one feedback entry from the same question type...")
        for fb, sim_score in sorted_feedback:
            base_question = fb["question"].lower().replace("?", "").strip()
            if base_question in unique_questions:  # Ensure we already have this question type
                continue  # Skip if already included
            
            selected_feedback.append(fb["feedback"])
            unique_questions.add(base_question)
            logging.info(f"✅ Added additional question-type match feedback: {fb['feedback']}")

            if len(selected_feedback) >= top_k:
                break

        # 🔹 Final Debugging Log
        logging.info(f"✅ Final selected feedback:\n{selected_feedback}")

        return "\n".join(selected_feedback) if selected_feedback else ""  # Return an empty string if no relevant feedback is found

    def run(self, question):
        """
        Runs the RAG model pipeline for a given question.

        This method:
        1. Retrieves relevant documents using the retriever.
        2. Finds user feedback related to the query.
        3. Generates an answer using retrieved documents and feedback.

        Args:
            question (str): The user's query.

        Returns:
            str: The generated response from the RAG model.
        """
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])

        # Retrieve relevant feedback
        feedback_documents = [doc for doc in self.web_documents if "---Feedback---" in doc.page_content]
        feedback_texts = "\n".join([doc.page_content for doc in feedback_documents])

        # Select the most relevant feedback
        feedback_texts = self._get_relevant_feedback(question)

        # **🔍 Debugging: Log Retrieved Feedback**
        logging.info(f"🔎 Selected Feedback for Question:\n{feedback_texts}")

        if not feedback_texts.strip():
            logging.warning("⚠️ No feedback found for this query.")

        # Generate the answer using the updated prompt format
        answer = self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts,
            "feedback": feedback_texts  # Pass retrieved feedback separately
        })

        # **🔍 Debugging: Log Model Output**
        logging.info(f"Model Output:\n{answer}")

        return answer