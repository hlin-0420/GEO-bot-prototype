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

script_dir = Path(__file__).resolve().parent

# Construct the full path to GEO_Limits.htm
geo_limits_path = os.path.join(script_dir, '..', 'Data', 'Introduction', 'GEO_Limits.htm')

# construct the full path to the output folder from "playground"
playground_output_lda_vis_path = os.path.join(script_dir, '..', 'playground', 'output', 'lda_visualisation.html')

# Normalize the path to eliminate any redundant separators or up-level references
geo_limits_path = os.path.normpath(geo_limits_path)

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
    
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, playground_output_lda_vis_path)

webbrowser.open(playground_output_lda_vis_path)