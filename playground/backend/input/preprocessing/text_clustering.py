import os
from bs4 import BeautifulSoup
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

print(f"Dataset types: {type(dataset)}.")

script_dir = Path(__file__).resolve().parent

# Path to the Data folder
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'Data'))

data = []

# Aggregate all .htm file contents recursively
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".htm") or file.endswith(".html"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                data.append({
                    "filename": os.path.relpath(file_path, data_dir),  # optional: relative path
                    "content": text
                })
                
# Convert to a Huggingface Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

print(f"Dataset types: {type(dataset)}.")
print(dataset)

embedding_model = SentenceTransformer("thenlper/gte-small")

# Instead of passing 'dataset', pass the list of 'content' values
texts = dataset["content"]  # Extract the 'content' field as a list
embeddings = embedding_model.encode(texts, show_progress_bar=True)

print(embeddings.shape)

hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom"
).fit(embeddings)

clusters = hdbscan_model.labels_

print(f"Cluster size: ({len(set(clusters))}).")

reduced_embeddings = UMAP(n_components = 2, min_dist = 0.0, metric = "cosine", random_state=42).fit_transform(embeddings)
# Plot
plt.figure(figsize=(10, 8))
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    c=clusters,
    cmap='Spectral',
    s=10
)
plt.colorbar(label='Cluster')
plt.title('HDBSCAN Clusters visualized with UMAP', fontsize=14)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()