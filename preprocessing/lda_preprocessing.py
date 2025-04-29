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
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

script_dir = Path(__file__).resolve().parent

# Construct the full path to GEO_Limits.htm
geo_limits_path = os.path.join(script_dir, '..', 'Data', 'Introduction', 'GEO_Limits.htm')

# construct the full path to the output folder from "playground"
playground_output_lda_vis_path = os.path.join(script_dir, '..', 'playground', 'output', 'lda_visualisation.html')

# Normalize the path to eliminate any redundant separators or up-level references
geo_limits_path = os.path.normpath(geo_limits_path)

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    # Try opening a session to force auth check
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ Neo4j authentication successful.")
except AuthError as e:
    print("❌ Neo4j authentication failed. Please check your credentials.")
    print(f"Details: {e}")
    exit(1)

with open(geo_limits_path, encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Extract visible text
text = soup.get_text(separator=" ", strip=True)

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

processed_docs = [preprocess(text)]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=10)

# Display the topics
topics = lda_model.print_topics()
for topic in topics:
    print(topic)
    
topic_keywords = {}
for topic_id, topic_terms in lda_model.show_topics(formatted=False):
    keywords = [word for word, prob in topic_terms]
    topic_keywords[f"Topic_{topic_id}"] = keywords
    
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, playground_output_lda_vis_path)

webbrowser.open(playground_output_lda_vis_path)

def create_ontology(tx, topic, keyword):
    tx.run("""
        MERGE (t:Topic {name: $topic})
        MERGE (k:Keyword {name: $keyword})
        MERGE (k)-[:RELATED_TO]->(t)
    """, topic=topic, keyword=keyword)
    
with driver.session() as session:
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            session.execute_write(create_ontology, topic, keyword)
            
print("Ontology successfully created in Neo4j.")

# --- Generate Cypher visualisation link ---
# This will open the Neo4j Browser with a pre-filled MATCH query
from urllib.parse import quote_plus

cypher_query = """
MATCH (k:Keyword)-[:RELATED_TO]->(t:Topic)
RETURN k, t
"""
encoded_query = quote_plus(cypher_query)

# Update this URL if you are using a remote instance
if "localhost" in NEO4J_URI:
    neo4j_browser_url = f"http://localhost:7474/browser/?cmd=play&arg={encoded_query}"
else:
    # NOTE: Replace with your actual sandbox URL
    neo4j_browser_url = f"https://db168837adc35d7d42f6c550e803f71a.neo4jsandbox.com/browser/?cmd=play&arg={encoded_query}"

print(f"Opening Neo4j Browser with visualisation: {neo4j_browser_url}")
webbrowser.open(neo4j_browser_url)

driver.close()