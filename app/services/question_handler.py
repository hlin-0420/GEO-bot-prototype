import time
import re
from app.services.session_manager import save_chat_session

def process_question(question_id, question, ai_bot, current_session_id, current_session_messages, stored_responses):
    """
    Process the question using the AI bot, format the response, and store it.
    Includes timing logs to diagnose delays.
    """
    print(f"[🔍 process_question] Start processing question ID: {question_id}")
    print(f"[🧠 Question] {question}")
    
    start_time = time.time()
    
    # Step 1: Query the bot
    print("[⏳] Calling ai_bot.query()...")
    try:
        query_start = time.time()
        response = ai_bot.query(question)
        query_time = time.time() - query_start
        print(f"[✅] Response received from AI model in {query_time:.4f} seconds.")
    except Exception as e:
        print(f"[❌ ERROR] ai_bot.query failed: {str(e)}")
        response = "An error occurred while querying the model."
    
    print("[📨] Raw response:", response)
    
    # Step 2: Format the response
    format_start = time.time()
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)
    format_time = time.time() - format_start
    print(f"[🎨] Response formatting took {format_time:.4f} seconds.")

    # Step 3: Update shared state
    stored_responses[question_id] = formatted_response
    current_session_messages.append({"role": "assistant", "content": response})
    print("[🗂️] Stored response in state and updated message history.")

    # Step 4: Save session
    save_start = time.time()
    save_chat_session(current_session_id, current_session_messages)
    save_time = time.time() - save_start
    print(f"[💾] Session saved in {save_time:.4f} seconds.")
    
    total_time = time.time() - start_time
    print(f"[🏁 process_question] Total processing time: {total_time:.4f} seconds.\n")

    return total_time
