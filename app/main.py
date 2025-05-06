from flask import Flask
from app.routes.api import api_blueprint
from app.routes.ui import ui_blueprint
from app import state
from threading import Lock

state.lock = Lock()

app = Flask(__name__)

# Register blueprints
app.register_blueprint(api_blueprint)
app.register_blueprint(ui_blueprint)