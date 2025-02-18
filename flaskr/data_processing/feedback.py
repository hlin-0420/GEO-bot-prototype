import os
import json

def append_feedback(feedback_entry, filename="feedback_dataset.json"):
    try:
        # Check if the file exists and is non-empty
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Load existing data
                    if not isinstance(data, list):  # Ensure it's a list
                        data = []
                except json.JSONDecodeError:
                    data = []  # If there's a parsing error, start fresh
        else:
            data = []  # Initialize an empty list if file doesn't exist or is empty

        # Append new entry
        data.append(feedback_entry)

        # Write back as a proper JSON list
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)  # Pretty formatting for readability

        print("Feedback appended successfully!")

    except Exception as e:
        print(f"Error while appending feedback: {e}")