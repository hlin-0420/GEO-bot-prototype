import cohere
import pickle
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('../../'))))

from playground.config.config import COHERE_API_KEY

co = cohere.Client(COHERE_API_KEY)

def compute_embeddings(texts, input_type="search_document"):
    response = co.embed(
        texts=texts,
        input_type=input_type
    )
    return response.embeddings

def save_embeddings(embeddings, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
