import json

def load_feedback_dataset():
    with open("feedback_dataset.json", "r") as f:
        return json.load(f)