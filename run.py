import os
import platform

print("[DEBUG] run.py started")
print(f"[DEBUG] Python version: {platform.python_version()} ({platform.python_implementation()})")
print(f"[DEBUG] Working directory: {os.getcwd()}")

from app.main import create_app
print("[DEBUG] main.py loaded")

app = create_app()
print("[DEBUG] Flask app created")

if __name__ == "__main__":
    print("[DEBUG] Starting Flask app...")
    app.run(debug=True)