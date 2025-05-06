print("[DEBUG] run.py started")

import sys
import os
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] Working directory: {os.getcwd()}")

from app.main import app
print("[DEBUG] Imported app from main.py")

if __name__ == "__main__":
    print("[DEBUG] Launching Flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)