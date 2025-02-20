from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
import ace_tools_open as tools

app = Flask(__name__)

# Load dataset
filename = "Data.xlsx"
xls_data = pd.ExcelFile(filename)
train_data_df = pd.read_excel(xls_data, sheet_name="Sheet1")
train_data_df.columns = ["Questions", "Answers"]

# Load pre-trained QA model
model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template("index.html")  # HTML page for user interaction

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("question", "")
    
    if not user_input:
        return jsonify({"response": "Please ask a question!"})
    
    print(f"Asked Question: \"{user_input}\"")

    tools.display_dataframe_to_user(name="Training DataFrame", dataframe=train_data_df)

    # Search for a matching question
    match = train_data_df[train_data_df["Questions"].str.contains(user_input, case=False, na=False)]
    
    print(f"There is a match in questions or not: {match}")

    if not match.empty:
        response = match["Answers"].values[0]
    else:
        # Use the transformer model for QA
        context = " ".join(train_data_df["Questions"].tolist())  # Combine all questions
        model_response = qa_pipeline({"question": user_input, "context": context})
        response = model_response["answer"]
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
