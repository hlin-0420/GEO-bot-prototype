import os
import json
import pandas as pd
import logging

# Process uploaded file
def process_file(file_path):
    from app.services.ollama_bot import get_bot

    print("***PROCESSING FILE***")
    ai_bot = get_bot()
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            ai_bot.add(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the file encoding.")
        return "Error: Invalid file encoding."

def clean_dataframe(df):
    """
    Replace NaN, None, or other invalid values with an empty string in the given DataFrame.
    """
    return df.fillna("").replace({None: ""})

def append_to_excel(question, response, excel_file, model_name):
    """
    Append a new question and response row to the Excel file.
    """
    new_entry = pd.DataFrame([[question, model_name, response]], columns=["Question", "Model Name", "Response"])

    if not os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            new_entry.to_excel(writer, index=False)
    else:
        existing_data = pd.read_excel(excel_file)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode='w') as writer:
            updated_data.to_excel(writer, index=False)
            auto_adjust_column_width(writer, updated_data)

def auto_adjust_column_width(writer, df):
    """
    Auto-adjust column width based on the max length of content in each column.
    """
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for column in df.columns:
        max_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
        col_idx = df.columns.get_loc(column) + 1
        col_letter = chr(64 + col_idx) if col_idx <= 26 else f"A{chr(64 + col_idx - 26)}"
        worksheet.column_dimensions[col_letter].width = max_length

def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []