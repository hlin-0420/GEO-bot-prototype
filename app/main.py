import os
import time
from flask import Flask

print("[DEBUG] main.py loaded")

try:
    print("[DEBUG] Importing register_blueprints...")
    from app.routes import register_blueprints
    print("[DEBUG] register_blueprints imported")
except Exception as e:
    print(f"[ERROR] Failed to import register_blueprints: {e}")
    raise

def create_app():
    print("[DEBUG] Creating Flask app...")
    
    # Define paths
    TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
    STATIC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

    app = Flask(__name__, template_folder=TEMPLATE_PATH, static_folder=STATIC_PATH)

    # Register blueprints
    register_blueprints(app)

    # Simulated component loading block (e.g. models, DBs)
    print("[DEBUG] Starting heavy component loading")
    start = time.time()
    # Load models, DB, etc. here
    print(f"[DEBUG] Component load complete in {time.time() - start:.2f} seconds")

    return app


if __name__ == "__main__":
    app = create_app()
    print("[DEBUG] Starting Flask app via main.py")
    app.run(debug=True)