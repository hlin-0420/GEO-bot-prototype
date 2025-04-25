<h1>🧪 Semantic Search Playground</h1>

This playground enables experimentations on semantic search using Cohere's embedding models and FAISS for efficient similary search on query items. 

<h2>🎯Objectives </h2>
1. Implement semantic search functionalities using Cohere's embedding API.
2. Apply FAISS for rapid nearest-neighbor searches over embedded textual data.
3. Experiment with various datasets to evaluate semantic search performance.

<h2> 🗂️ Directory Structure </h2>

```
playground/
├── config/                 # Configuration files (e.g., API keys)
│   └── config.py
├── data/                   # Sample datasets for experimentation
│   └── interstellar.txt
├── embeddings/             # Stored embeddings for reuse
├── notebooks/              # Jupyter notebooks demonstrating usage
│   └── semantic_search_demo.ipynb
├── utils/                  # Utility modules for embedding and search operations
│   ├── embedding_utils.py
│   └── search_utils.py
└── .env                    # Environment variables (e.g., COHERE_API_KEY)
```

<h2>⚙️ Setup Instructions</h2>

1. Clone the Repository:

```
git clone https://github.com/hlin-0420/GEO-bot-prototype.git
cd GEO-bot-prototype/playground
```

2. Create and Activate a Virtual Environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies:

```
pip install -r requirements.txt
```

4. Configure Environment Variables:

+ Create a .env file in the playground directory. 

+ Add your Cohere API key: `COHERE_API_KEY=your_cohere_api_key_here`

<h2> 🧠 Methodology </h2>
<h3> Data Preparation </h3>

Load textual data from `data/interstellar.txt`

<h3> Embedding Generation </h3>

Use Cohere's embedding model to convert textual data into high-dimensional vector embeddings. Embeddings are generated with appropriate input_type parameters:
+ search_document for corpus data.
+ search_query for user queries.

<h3> Indexing with FAISS </h3>

Create a FAISS index (IndexFlatL2) using the generated document embeddings for efficient similarity search.​

<h3> Semantic Search </h3>

+ Embed user queries using the same Cohere model.
+ Perform a nearest-neighbor search in the FAISS index to retrieve top-k similar documents.
+ Display results with similarity scores and rankings.​

<h2> 📓 Usage Example </h2>
Refer to the Jupyter notebook `notebooks/semantic_search_demo.ipynb` for a step-by-step demonstration of the semantic search process, including <b>data loading</b>, <b>embedding generation</b>, <b>indexing</b>, and <b>querying</b>.​

<h2>🔒 Security Note</h2>
Ensure that your .env file is <b>not</b> committed to version control. The <i>.gitignore</i> file should already exclude it.

<b>Never</b> expose your Cohere API key in <b>public</b> repositories.​

<h2>📬 Feedback and Contributions </h2>
Feel free to open issues or submit pull requests for improvements, bug fixes, or new features. 

Your contributions are welcome!