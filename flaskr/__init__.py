import os
from flask import Flask
from threading import Lock

from .models.ollama_bot import OllamaBot

from .routes.main_routes import main_bp
from .routes.feedback_routes import feedback_bp
from .routes.upload_routes import upload_bp
from .routes.ask_routes import ask_bp

ai_bot = OllamaBot()
lock = Lock()
pending_responses = {}
question_id = 0

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Register Blueprints
    app.register_blueprint(ask_bp, url_prefix="/")  # ✅ Ensure /ask is registered
    app.register_blueprint(upload_bp, url_prefix="/")
    app.register_blueprint(feedback_bp, url_prefix="/")
    app.register_blueprint(main_bp, url_prefix="/")
    
    return app