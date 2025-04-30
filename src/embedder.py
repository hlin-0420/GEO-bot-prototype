# src/embedding.py
from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(model_path="./local_models/offline_model"):
    return HuggingFaceEmbeddings(model_name=model_path)