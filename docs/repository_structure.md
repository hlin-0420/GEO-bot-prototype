# Repository Structure and Run Surfaces

## Top-level layout

- `app/`: Flask application package.
  - `main.py`: Flask app factory (`create_app`) and bootstrap logic.
  - `routes/`: Blueprint registration and HTTP route handlers.
  - `utils/`: Utility modules for file handling, feedback, and analysis helpers.
  - `services/`: Service-layer modules used by routes/app orchestration.
- `src/`: NLP/RAG and graph helper modules used by the application pipeline.
- `templates/`: Jinja templates for web pages.
- `static/`: Front-end assets (CSS, images, icons, and static data).
- `Data/`: Runtime and domain data (HTML corpus, model artifacts, logs, sessions, and evaluation files).
- `tests/`: Test suite.
- `docs/`: User/developer docs.
- `playground/`: Experimental semantic-search playground and backend/config artifacts.

## Primary application run surfaces

### 1) Flask app via `run.py` (recommended in current code)
- Entry script imports `create_app()` from `app.main`, builds the Flask app, and runs it with `debug=True`.
- Command:

```bash
python run.py
```

### 2) Flask app via module file directly
- `app/main.py` also supports direct execution (`if __name__ == "__main__": ...`).
- Command:

```bash
python app/main.py
```

### 3) Legacy/readme-noted launcher
- `docs/README.md` references `python offline-app.py`, but this file is not present in the current repository snapshot.
- Treat this as historical/outdated documentation unless the file is restored.

## Supporting/secondary runnable areas

### 4) Playground area (`playground/`)
- Intended for semantic-search experimentation (Cohere + FAISS), documented in `playground/README.md`.
- Typical run surfaces are notebook-driven (`notebooks/semantic_search_demo.ipynb`) and backend helpers under `playground/backend`.

### 5) Tests
- The repository includes a `tests/` directory and can be executed as a standard Python test surface (for example with `pytest`, if installed in environment).

## Runtime dependencies and external services

- Python dependencies are listed in `requirements.txt`.
- Documentation indicates optional/required external model backends:
  - Ollama runtime with local models (e.g., llama, deepseek variants).
  - OpenAI API key for OpenAI-backed flows.

## Notes for maintainers

- The app appears to be organized around a Flask route/UI layer (`app/`) and a retrieval/processing layer (`src/`).
- Large static/domain data under `Data/` is likely required for full local behavior (RAG context, sessions, feedback, and evaluation).
- Consider updating `docs/README.md` to align launcher instructions with `run.py` / `app/main.py`.
