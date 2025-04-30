import cohere
import numpy as np
import pandas as pd
import faiss

import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('../../'))))

from playground.config.config import COHERE_API_KEY

co = cohere.Client(COHERE_API_KEY)

# Function to create and populate FAISS index
def create_faiss_index(embeds):
    embeds_np = np.array(embeds).astype('float32')
    dim = embeds_np.shape[1]
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeds_np)
    
    return index


def search(query, texts, index, num_of_results=3):
    # Get query embedding from Cohere
    query_embedding = co.embed(texts=[query], input_type="search_query").embeddings[0]
    
    # Perform FAISS search
    distances, similar_item_ids = index.search(np.float32([query_embedding]), num_of_results)
    
    texts_np = np.array(texts)
    
    results = pd.DataFrame(data={
        'texts': texts_np[similar_item_ids[0]],
        'distance': distances[0]
    })
    
    return results