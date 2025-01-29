from flask import Flask, request, jsonify, render_template, Response
import logging
import os
import threading
import time
from threading import Lock
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LlamaBot:
    def __init__(self, model_name="/home/hlin656/.llama/checkpoints/Llama3.2-1B"):
        """
        Initialize the LlamaBot with the specified model.
        """
        self.device = "cpu"  # Force model to use CPU
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(self.device)
        self.base_directory = "Data"
        self.contents = []  # Store processed content
        self._load_content()

    def _list_htm_files(self):
        """Recursively finds all .htm files in the base directory."""
        htm_files = []
        for root, _, files in os.walk(self.base_directory):
            for file in files:
                if file.endswith(".htm"):
                    relative_path = os.path.relpath(os.path.join(root, file), start=self.base_directory)
                    htm_files.append(self.base_directory + "/" + relative_path)
        return htm_files

    def _load_content(self):
        """Load and process all .htm files from the base directory."""
        htm_files = self._list_htm_files()
        logging.info(f"Found {len(htm_files)} .htm files.")

        for file_path in htm_files:
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                    self.contents.append(content)
            except UnicodeDecodeError:
                logging.error(f"Could not read the file {file_path}. Check the encoding.")

    def fine_tune(self):
        """Fine-tune the Llama model using LoRA."""
        logging.info("Starting fine-tuning process...")
        dataset = [{"instruction": content, "response": content} for content in self.contents]
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config).to(self.device)
        
        training_args = TrainingArguments(
            output_dir="./fine_tuned_llama",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=1e-4,
            optim="adamw_torch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()
        self.model.save_pretrained("fine_tuned_llama")
        logging.info("Fine-tuning complete. Model saved.")

    def query(self, question):
        """Query the fine-tuned model."""
        logging.info(f"Processing question: {question}")
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Initialize LlamaBot
llama_bot = LlamaBot()
pending_responses = {}
stored_responses = {}
question_id = 0
lock = Lock()

def process_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            llama_bot.contents.append(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the encoding.")
        return "Error: Invalid file encoding."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = f"./Data/{file.filename}"
        file.save(file_path)
        result = process_file(file_path)
        return jsonify({"message": result})

def process_question(question_id, question):
    time.sleep(2)
    try:
        response = llama_bot.query(question)
        stored_responses[question_id] = response.replace("\n", "<br>")
    except Exception as e:
        print(f"The exception is {e}")
        stored_responses[question_id] = "Still thinking about how to answer..."

@app.route("/ask", methods=["POST"])
def ask():
    global question_id
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400
        
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        with lock:
            current_id = str(question_id)
            question_id += 1
        
        pending_responses[current_id] = "Processing..."
        threading.Thread(target=process_question, args=(current_id, question)).start()
        
        return jsonify({"question_id": current_id}), 200
    
    except Exception as e:
        app.logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/response/<question_id>", methods=["GET"])
def get_response(question_id):
    def generate_response():
        while True:
            response = stored_responses.get(question_id)
            if response:
                yield f"data: {response}\n\n"
                break
            else:
                yield "data: Processing your question...\n\n"
                time.sleep(1)
    return Response(generate_response(), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)