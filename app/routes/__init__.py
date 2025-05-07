from .feedback_handler import feedback_routes
from .session_manager_extended import session_routes
from .timing_logger import timing_routes
from .api import api_blueprint
from .upload_handler import upload_routes

def register_blueprints(app):
    app.register_blueprint(feedback_routes)
    app.register_blueprint(session_routes)
    app.register_blueprint(timing_routes)
    app.register_blueprint(upload_routes)
    app.register_blueprint(api_blueprint)