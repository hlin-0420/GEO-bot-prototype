import os
import time
import logging

class RAGApplication:
    def __init__(self, retriever, rag_chain, web_documents):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.web_documents = web_documents  # For potential future extensions

    def run(self, question):
        """Runs the RAG retrieval and generates a response with detailed runtime analysis."""

        total_start_time = time.perf_counter()

        # Step 1: Retrieve relevant documents
        retrieval_start_time = time.perf_counter()
        documents = self.retriever.invoke(question)
        retrieval_time = time.perf_counter() - retrieval_start_time

        doc_texts = "\n".join(doc.page_content for doc in documents)

        # Step 2: Generate the answer using the retrieved documents
        response_start_time = time.perf_counter()
        response = self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts,
            "stream": True
        })
        response_time = time.perf_counter() - response_start_time

        total_execution_time = time.perf_counter() - total_start_time

        # Logging detailed runtime analysis
        print("🕒 RAG Execution Time Breakdown:")
        print(f"   - Document Retrieval Time: {retrieval_time:.4f} seconds")
        print(f"   - Response Generation Time: {response_time:.4f} seconds")
        print(f"   - Total Execution Time: {total_execution_time:.4f} seconds")

        return response