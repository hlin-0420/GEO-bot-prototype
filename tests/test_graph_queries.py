import sys
import os
import time  # ⏱️ Import timing module

# Add src/ to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, CYPHER_FILE_PATH
from graph_utils import load_cypher_file_to_neo4j, clear_database
from rag_pipeline import build_neo4j_rag_pipeline, format_graph_info  # ✅ import helper
from langchain_neo4j import Neo4jGraph

def main():
    print("🧹 Clearing graph database...")
    clear_database(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    print("📥 Loading Cypher script into Neo4j...")
    load_cypher_file_to_neo4j(CYPHER_FILE_PATH, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    print("⚙️ Initialising Neo4j RAG pipeline...")
    
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    rag_chain = build_neo4j_rag_pipeline(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    # Fetch and format graph data as context
    records = graph.query("""
        MATCH (n)-[r]->(m)
        RETURN n.name AS from, type(r) AS relationship, m.name AS to
        LIMIT 15
    """)
    graph_context = format_graph_info(records)

    questions = [
        "What tool helps create ODF templates?",
        "What features does the Template Creation Wizard enable?",
        "What does the ODF Template File (ODT) contain?",
        "Which service customizes the ODT template?",
        "What warnings are detected in the generated template?"
    ]

    print("❓ Running sample queries...\n")
    for q in questions:
        print(f"Question: {q}")
        start_time = time.time()  # Start timing

        response = rag_chain.invoke({
            "question": q,
            "graph_context": graph_context
        })

        end_time = time.time()  # End timing
        duration = end_time - start_time

        print(f"Answer: {response.strip()}")
        print(f"⏱️ Time taken: {duration:.2f} seconds\n{'-'*60}")

if __name__ == "__main__":
    main()