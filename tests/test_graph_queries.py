import sys
import os
import time
import re

# Add src/ to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from load_htm_file import load_all_htm_files_content
from rag_pipeline import build_offline_chatbot

# 🧠 Step 1: Warm up the model
def warm_up_llm(qa_chain):
    print("🔥 Warming up LLM model (first dummy call)...")
    _ = qa_chain.invoke({"input": "This is a warm-up question. You can ignore this."})
    time.sleep(2)
    print("✅ Warm-up complete.\n")

# 🗂 Step 2: Load and clean HTM content
def load_and_prepare_content():
    print("📄 Loading and processing HTM content...")
    content = load_all_htm_files_content()
    content = re.sub(r"\n\s*\n", "\n", content)
    return content

# ❓ Step 3: Run sample questions
def run_sample_questions(qa_chain):
    questions = [
        "What features does the Template Creation Wizard enable?",
        "What does the ODF Template File (ODT) contain?",
        "Which tool helps to create ODF template files?",
        "Which service customizes the ODT template?",
        "What warnings are detected in the generated template?",
        "What is a computed curve?",
        "Why can't I add 251 curve shades to my log?",
        'I want to use the name "Hydrocarbon bearing zone highlighted" as my curve shade name. Why is it not allowed?'
    ]

    print("❓ Running sample queries...\n")
    for q in questions:
        print(f"Question: {q}")
        start_time = time.time()
        response = qa_chain.invoke({"input": q})  # ✅ Use dynamic retrieval
        duration = time.time() - start_time
        print(f"Answer: {response['answer'].strip()}")
        print(f"⏱️ Time taken: {duration:.2f} seconds\n{'-'*60}")

# 🔁 Main Orchestrator
def main():
    content = load_and_prepare_content()
    qa_chain = build_offline_chatbot(content)
    # warm_up_llm(qa_chain)
    run_sample_questions(qa_chain)

if __name__ == "__main__":
    main()