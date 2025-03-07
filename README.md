## Installation Guide

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