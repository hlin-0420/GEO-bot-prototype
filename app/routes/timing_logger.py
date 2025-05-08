from flask import Blueprint, request, jsonify
import os, logging, json
import pandas as pd
import numpy as np
from app.config import EXCEL_FILE, EXPECTED_RESULTS_FILE, TIMED_RESPONSES_FILE
from app.utils.analysis_helpers import is_similar_question, find_best_match, calculate_semantic_similarity, clean_dataframe  # Assuming these exist

timing_routes = Blueprint('timing', __name__)

@timing_routes.route("/store-response-time", methods=["POST"])
def store_response_time():
    try:
        data = request.json
        question_number = str(data.get("questionNumber"))
        question_text = data.get("question")
        duration = float(data.get("duration"))
        timestamp = data.get("timestamp")

        if not question_number or not question_text or duration is None or not timestamp:
            return jsonify({"error": "Invalid data"}), 400

        response_times = []
        if os.path.exists(TIMED_RESPONSES_FILE):
            with open(TIMED_RESPONSES_FILE, "r", encoding="utf-8") as file:
                try:
                    response_times = json.load(file)
                except json.JSONDecodeError:
                    response_times = []

        response_times.append({
            "question_number": question_number,
            "question": question_text,
            "response_time": duration,
            "timestamp": timestamp
        })

        with open(TIMED_RESPONSES_FILE, "w", encoding="utf-8") as file:
            json.dump(response_times, file, indent=4)

        return jsonify({"message": "Response time stored successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@timing_routes.route("/get-results", methods=["GET"])
def get_results():
    try:
        logging.info("Processing GET request for /get-results")

        if not os.path.exists(EXCEL_FILE):
            logging.warning(f"Results file not found: {EXCEL_FILE}")
            return jsonify({"error": "Results file not found"}), 404

        columns = ["Question", "Model Name", "Response", "Expected Answer", "similarity_score"]
        num_rows = 10
        questions = [f"Question {i+1}" for i in range(num_rows)]
        models = [f"Model_{np.random.randint(1, 5)}" for _ in range(num_rows)]
        responses = [f"Response {i+1}" for i in range(num_rows)]
        expected_answers = [f"Expected Answer {i+1}" for i in range(num_rows)]
        similarity_scores = np.random.uniform(0, 1, num_rows)

        temp_df = pd.DataFrame([
            {
                "Question": questions[i],
                "Model Name": models[i],
                "Response": responses[i],
                "Expected Answer": expected_answers[i],
                "similarity_score": similarity_scores[i],
            }
            for i in range(num_rows)
        ])

        temp_results = pd.DataFrame({
            "Model Name": ["Llama 3.2", "Deepseek 1.5", "Haystack", "OpenAI"],
            "Accuracy": ["90%", "84%", "79%", "92%"]
        })

        try:
            df = pd.read_excel(EXCEL_FILE)
        except Exception:
            df = pd.DataFrame(columns=["Question", "Model Name", "Response"])
            return jsonify({"models": temp_results.to_dict(orient="records"), "filtered_results": temp_df.to_dict(orient="records")})

        if os.path.exists(EXPECTED_RESULTS_FILE):
            try:
                expected_results_df = pd.read_excel(EXPECTED_RESULTS_FILE)
                expected_answers_dict = dict(zip(expected_results_df["Question"], expected_results_df["Response"]))
            except Exception:
                expected_answers_dict = {}
        else:
            expected_answers_dict = {}

        df["is_question_in_expected"] = df["Question"].apply(lambda q: is_similar_question(q, expected_answers_dict.keys()))
        filtered_df = df[df["is_question_in_expected"]].copy()

        if not filtered_df.empty:
            filtered_df["Expected Answer"] = filtered_df["Question"].apply(lambda q: find_best_match(q, expected_answers_dict))
            filtered_df["similarity_score"] = filtered_df.apply(
                lambda row: calculate_semantic_similarity(str(row["Expected Answer"]), str(row["Response"])), axis=1
            )

        temp_filtered_file = "Data/temp_filtered_data.xlsx"
        filtered_df.drop(columns=["is_question_in_expected"], inplace=True)
        filtered_df.to_excel(temp_filtered_file)

        if "Model Name" not in df.columns or "Accuracy" not in df.columns:
            results = temp_results
        else:
            results = df[["Model Name", "Accuracy"]]

        results = clean_dataframe(results)
        filtered_df = clean_dataframe(filtered_df)

        return jsonify({
            "models": json.loads(json.dumps(results.to_dict(orient="records"))),
            "filtered_results": json.loads(json.dumps(filtered_df.to_dict(orient="records")))
        })

    except Exception as e:
        logging.error(f"Error reading results: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
