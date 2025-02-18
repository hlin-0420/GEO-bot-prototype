import pandas as pd  # For handling tables as DataFrames
from tabulate import tabulate  # For formatting tables as readable text
import logging
import time
import re

from flask import current_app # imports global variable "stored_responses" from __init__.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_text(soup):
    # Extract only meaningful paragraph text
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]  # Exclude very short text
    clean_text = "\n\n".join(paragraphs)
    
    return clean_text

def extract_table(soup):
    tables = soup.find_all("table")
    
    formatted_tables = []
                    
    # Process and format each table
    for i, table in enumerate(tables, start=1):
        rows = []
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            rows.append(cols)

        # Convert to DataFrame for better readability
        df = pd.DataFrame(rows)
                        
        formatted_table = tabulate(df, headers="firstrow", tablefmt="grid")
        
        formatted_tables.append(formatted_table)
        
    formatted_tables = "\n\n".join(formatted_tables)
    
    return formatted_tables

def extract_list(soup):
    # Extract lists properly
    lists = []
    for ul in soup.find_all("ul"):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        lists.append(items)
    return lists

def process_file(file_path, ai_bot):
    print("***PROCESSING FILE***")
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            ai_bot.add(content)
        return "File processed successfully."
    except UnicodeDecodeError:
        logging.error(f"Error: Could not read the file {file_path}. Please check the file encoding.")
        return "Error: Invalid file encoding."

def check_selected_options(selectedOptions):
    expected_options = ["text", "table", "list"]
    return set(selectedOptions) == set(expected_options)

def process_question(question_id, question, ai_bot, selectedOptions):
    """
    Simulate long processing of the question and store the response.
    """
    time.sleep(2)  # Simulating "thinking time"
    # try:
    response = ai_bot.query(question)
    
    print(f"Check selected Options: {check_selected_options(selectedOptions)}")
    
    if check_selected_options(selectedOptions) == False:
        # if not all the options are selected, customise the training function. 
        ai_bot._load_content(selectedOptions) # resets the web documents information with the feedback.
    
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)
    
    print(f"The formatted response is \"{formatted_response}\"")

    with current_app.app_context():
        current_app.stored_responses[question_id] = formatted_response