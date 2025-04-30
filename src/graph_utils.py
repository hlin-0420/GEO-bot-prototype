# src/graph_utils.py
from neo4j import GraphDatabase

def clear_database(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

def load_cypher_file_to_neo4j(file_path, uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        with open(file_path, "r", encoding="utf-8") as f:
            cypher_script = f.read()

        statements = [stmt.strip() for stmt in cypher_script.split(";") if stmt.strip()]

        for i, stmt in enumerate(statements, 1):
            try:
                print(f"▶ Running statement {i}/{len(statements)}:")
                print(stmt)
                session.run(stmt)
            except Exception as e:
                print(f"❌ Failed on statement {i}: {e}")
                break

    driver.close()