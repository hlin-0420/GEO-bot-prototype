# GEO Bot Prototype

A Flask-based prototype chatbot for answering questions over geological and technical document collections using a retrieval-augmented generation (RAG) workflow. The application combines local document ingestion, semantic retrieval, and a local language model via Ollama to provide grounded responses with an interactive web UI.

## Current progress

The project has evolved from a simple concept into a working prototype with the following components already in place:

- Flask web app with a browser-based interface
- Document upload and processing pipeline
- HTML/text extraction utilities for source material
- RAG setup using LangChain + FAISS
- Local LLM integration via Ollama
- Session-based chat handling and in-memory state management
- Feedback capture flow and evaluation logging
- Model selection support for several local model options
- Separate route modules for UI, API, session management, uploads, and feedback

This means the repository is no longer just a scaffold: it contains a complete prototype loop for ingestion, retrieval, answer generation, and user interaction.

## What the app does

The core workflow is:

1. Load source documents from the project data folder.
2. Extract content from supported file types such as HTML/text-like documents.
3. Split documents into chunks.
4. Embed chunks with Hugging Face embeddings.
5. Store them in a FAISS vector index.
6. Retrieve relevant passages for each question.
7. Pass the retrieved context and question into an Ollama LLM.
8. Return the response through the web interface or API.

This is a standard RAG architecture, adapted for a GEO-domain prototype where knowledge is grounded in uploaded or preprocessed local content.

## Project structure

```text
GEO-bot-prototype/
├── app/
│   ├── config.py             # project paths, models, upload limits
│   ├── main.py               # Flask app factory
│   ├── state.py              # in-memory request/session state
│   ├── routes/
│   │   ├── api.py            # main question/response API
│   │   ├── ui.py             # browser UI routes
│   │   ├── upload_handler.py # upload logic
│   │   ├── feedback_handler.py
│   │   ├── session_manager.py
│   │   ├── session_manager_extended.py
│   │   ├── timing_logger.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── ollama_bot_core.py     # main bot + RAG orchestration
│   │   ├── ollama_bot_helpers.py  # prompt/template support
│   │   ├── ollama_bot.py          # bot factory gateway
│   │   ├── rag_application.py     # retrieval + generation loop
│   │   ├── question_handler.py    # question processing helper
│   │   ├── session_manager.py
│   │   ├── document_processor.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── file_helpers.py
│   │   ├── html_file_loader.py
│   │   ├── analysis_helpers.py
│   │   ├── feedback.py
│   │   └── __init__.py
│   └── __init__.py
├── data/
│   ├── user_sessions/        # saved chat sessions and metadata
│   ├── feedback/             # feedback dataset files
│   ├── evaluation/           # query/answer evaluation records
│   ├── model_files/          # processed content and FAISS index artifacts
│   └── uploads/
├── docs/
│   ├── README.md
│   └── repository_structure.md
├── static/                   # CSS/JS assets
├── templates/                # HTML templates
├── requirements.txt
├── run.py                    # app entry point
├── LICENSE
├── .gitignore
└── .gitattributes
```

## Technology stack

- Python
- Flask
- LangChain
- LangChain Ollama
- LangChain Hugging Face embeddings
- FAISS
- pandas
- BeautifulSoup / bs4
- scikit-learn
- sentence-transformers
- Ollama-hosted local models

## Supported usage flow

The project is intended for local experimentation and prototype use. In practice the application can:

- accept uploaded documents,
- process them into searchable text,
- maintain per-session chat history,
- answer questions grounded in retrieved document chunks,
- log answers and feedback for evaluation,
- switch among configured model names such as Llama 3.2, DeepSeek, TinyLlama, Gemma, or OpenAI-compatible setups.

## Setup and run

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and required models

Install Ollama from the official site and pull the models used by the app:

```bash
ollama pull llama3.2:latest
ollama pull deepseek1.5
```

The project also references other models such as `tinyllama:latest` and `gemma3:1b` in configuration, though the default path is centered on the Llama 3.2 setup.

### 4. Start the application

```bash
python run.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Key strengths of the current prototype

- End-to-end RAG workflow is implemented, not just mocked.
- The app includes a workable UI and an API layer.
- It supports session tracking and persistent local logs.
- It is structured modularly with separate concerns for routes, services, and utilities.
- The project demonstrates a realistic pattern for local knowledge-base Q&A in a GEO context.

## Current limitations and known gaps

While the prototype is functional, there are still several limitations that should be addressed before treating it as production-ready:

- The app is optimized for local, demo-style usage rather than deployment in a production environment.
- Documentation and code references are slightly inconsistent; older instructions mention `offline-app.py`, while the actual entry point is `run.py`.
- The project is tightly coupled to local file paths and a specific directory structure.
- Model and data configuration are manually managed, which can make environment setup brittle.
- Retrieval and generation quality still rely strongly on the quality of the underlying document corpus and prompt design.
- The project does not yet appear to include a formal automated testing suite or CI validation pipeline.

## Notes for future work

Potential next improvements include:

- formal evaluation metrics for retrieval quality and answer correctness,
- better user experience around document ingestion and model selection,
- stronger configuration management for environment variables and secrets,
- improved deployment packaging for local or cloud hosting,
- stronger safeguards for hallucination control and source grounding,
- expanded support for multiple document types and enterprise datasets.

## Summary

The GEO Bot Prototype is a functioning local RAG application for answering questions from document collections in a GEO-oriented domain. It already demonstrates the central architecture of a practical AI assistant: data ingestion, retrieval, grounding, generation, and user interaction. The current codebase is a strong prototype foundation, with the main remaining work focused on robustness, validation, and operational maturity rather than core concept design.

This README reflects the current state of the codebase and should be treated as a progress-oriented overview of the repository as it exists today.
