## Installation Guide

### System Requirements
- Operating Systems: Windows 10/11
- Python version: 3.11
- Recommended Environment: Virtual Environment (venv)

### Data Folder Structure
The application uses a `Data` folder to store:
- Uploaded files (`uploaded_document.txt`)
- Processed content (`processed_content.txt`)
- Chat history (`ChatSessions/`)
- Feedback data (`feedback_dataset.json`)
- Expected query responses (`expected_query_responses.xlsx`)
- Query logs (`query_responses.xlsx`)

### Instructions
Follow these steps to setup the application:
1. Clone this repository. 
```
git clone https://github.com/hlin-0420/GEO-bot-prototype.git
```
2. Create and activate a virtual environment to prepare for installing relevant dependencies. 
```
python -m venv venv
venv/Scripts/activate
```
3. Download all relevant Python Libraries from requirements.txt
```
cd GEO-bot-prototype
pip install -r requirements.txt
```
4. Install Ollama
Install Ollama setup from https://ollama.com/download/windows


5. Download relevant models for Ollama 

This project uses Ollama model variants: Llama 3.2, Deepseek 1.5, and OpenAI, the following installation instructions will be used:

```
ollama pull llama3:2:latest
ollama pull deepseek-r1:1.5b
ollama pull openai
```

6. After setting up the models, run the chatbot via the following instruction:
```
python offline-app.py
```

After the instruction is run, wait until the local host link shows up on the command prompt -> open it for navigating to the Flask app homepage. 

Link: `http://127.0.0.1:5000/`