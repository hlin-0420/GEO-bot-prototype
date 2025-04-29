from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os
from pathlib import Path
import webbrowser
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import AuthError
from dotenv import load_dotenv
from urllib.parse import quote_plus
import ollama

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

script_dir = Path(__file__).resolve().parent

# Path to the Data folder
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'Data'))

# Aggregate all .htm file contents recursively
all_texts = []
loaded_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".htm"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                all_texts.append(text)
                loaded_files.append(file_path)

# Debug: print all loaded file paths
print("Loaded the following .htm files:")
for f in loaded_files:
    print(f)

# Join all document texts into a single string
combined_text = " ".join(all_texts)

# Path to output visualisation
playground_output_lda_vis_path = os.path.normpath(os.path.join(script_dir, '..', 'playground', 'output', 'lda_visualisation.html'))

# Authenticate Neo4j connection
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ Neo4j authentication successful.")
except AuthError as e:
    print("❌ Neo4j authentication failed. Please check your credentials.")
    print(f"Details: {e}")
    exit(1)

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(doc):
    return [
        lemmatizer.lemmatize(token)
        for token in simple_preprocess(doc, deacc=True)
        if token not in stop_words
    ]

# Preprocess documents
processed_docs = [preprocess(combined_text)]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=10)

# Display topics
topics = lda_model.print_topics()
for topic in topics:
    print(topic)

# Extract topic keywords
topic_keywords = {}
for topic_id, topic_terms in lda_model.show_topics(formatted=False):
    keywords = [word for word, prob in topic_terms]
    topic_keywords[f"Topic_{topic_id}"] = keywords

# Save LDA visualisation
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, playground_output_lda_vis_path)
webbrowser.open(playground_output_lda_vis_path)

def create_ontology(tx, topic, keyword):
    tx.run("""
        MERGE (t:Topic {name: $topic})
        MERGE (k:Keyword {name: $keyword})
        MERGE (k)-[:RELATED_TO]->(t)
    """, topic=topic, keyword=keyword)

# Insert ontology into Neo4j
def insert_ontology_to_neo4j():
    with driver.session() as session:
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                session.execute_write(create_ontology, topic, keyword)

insert_ontology_to_neo4j()

print("Ontology successfully created in Neo4j.")

# Generate Cypher visualisation link
cypher_query = """
MATCH (k:Keyword)-[:RELATED_TO]->(t:Topic)
RETURN k, t
"""
encoded_query = quote_plus(cypher_query)

if "localhost" in NEO4J_URI:
    neo4j_browser_url = f"http://localhost:7474/browser/?cmd=play&arg={encoded_query}"
else:
    neo4j_browser_url = f"https://db168837adc35d7d42f6c550e803f71a.neo4jsandbox.com/browser/?cmd=play&arg={encoded_query}"

print(f"Opening Neo4j Browser with visualisation: {neo4j_browser_url}")
webbrowser.open(neo4j_browser_url)

def query_neo4j(user_query):
    with driver.session() as session:
        cypher_query = """
        MATCH (k:Keyword)-[:RELATED_TO]->(t:Topic)
        RETURN k.name AS keyword, t.name AS topic
        """
        result = session.run(cypher_query)
        records = [f"Keyword: {record['keyword']} -> Topic: {record['topic']}" for record in result]
    return "\n".join(records)

def query_ollama(prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

def answer_user_query(user_question):
    context = query_neo4j(user_question)
    full_prompt = f"""
You are an assistant who answers questions based on the following ontology information:

{context}

Now answer the user's question: {user_question}
"""
    response = query_ollama(full_prompt)
    print("🔵 Answer:\n", response)

# Example Usage
user_question = input("Ask your question: ")
answer_user_query(user_question)

driver.close()