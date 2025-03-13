# GEO Bot Prototype

A chatbot application for querying geological data using LLMs (Llama 3.2, Deepseek 1.5, OpenAI) via Ollama.

## 🚀 Installation Guide

### ✅ System Requirements
- **Operating System:** Windows 10/11
- **Python Version:** 3.11+
- **Recommended Environment:** Virtual Environment (`venv`)

---

### 📂 Data Folder Structure
The application uses a `Data` folder to store:
- 📝 Uploaded files → `uploaded_document.txt`
- 📝 Processed content → `processed_content.txt`
- 💬 Chat history → `ChatSessions/`
- 💑 Feedback data → `feedback_dataset.json`
- 📊 Expected query responses → `expected_query_responses.xlsx`
- 🔍 Query logs → `query_responses.xlsx`

---

### 🛠️ Installation Steps

#### **1⃣ Clone the Repository**
```sh
git clone https://github.com/hlin-0420/GEO-bot-prototype.git
cd GEO-bot-prototype
```

#### **2⃣ Set Up a Virtual Environment**
```sh
python -m venv venv
# Windows
venv\Scripts\activate  
# macOS/Linux
source venv/bin/activate
```

#### **3⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

#### **4⃣ Install Ollama**
- Download and install **Ollama** from [Ollama Official Site](https://ollama.com/download).

#### **5⃣ Download Required Ollama Models**
This project uses **Llama 3.2, Deepseek 1.5, and OpenAI models**. Install them using:
```sh
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b
```
> ⚠️ Note: OpenAI models are not available through `ollama pull`. Instead, configure OpenAI API if needed.

#### **6⃣ Start the Chatbot**
Run the application:
```sh
python offline-app.py
```
- Wait until the **local host link** appears.
- Open the link in your browser:
  ```
  http://127.0.0.1:5000/
  ```

> ✅ **Before running**, make sure **Ollama is open** to enable the chatbot's connection with the model.

---

### 💡 **Troubleshooting**
#### ❌ **Ollama Models Not Found?**
- Ensure Ollama is running:
  ```sh
  ollama list
  ```
  If no models appear, re-run:
  ```sh
  ollama pull llama3:latest
  ollama pull deepseek:1.5
  ```

#### ❌ **Flask App Not Starting?**
- Run the command:
  ```sh
  python offline-app.py
  ```
- If you see `ModuleNotFoundError`, try:
  ```sh
  pip install -r requirements.txt
  ```

#### ❌ **404 Page Not Found?**
- Ensure Flask is running and open:
  ```
  http://127.0.0.1:5000/
  ```

---

### ✅ **Expected Output**
Upon successful execution, you should see:
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Copy and paste this link into your browser.

---

### 🎯 **Next Steps**
- 📚 Read the **[Documentation]()**.
- 🛠️ Customize models in **configurations**.
- 🚀 Extend functionality with additional APIs.

---