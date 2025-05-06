import time
import re
from app.services.session_manager import save_chat_session

def process_question(question_id, question, ai_bot, current_session_id, current_session_messages, stored_responses):
    """
    Simulate long processing of the question and store the response.
    """
    start_time = time.time()
    
    response = ai_bot.query(question)
    
    print("Response from advanced model:", response)
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ Time taken to run `query`: {elapsed_time:.4f} seconds.")
    
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)
    
    stored_responses[question_id] = formatted_response
    current_session_messages.append({"role": "assistant", "content": response})
    
    save_chat_session(current_session_id, current_session_messages)
    return elapsed_time