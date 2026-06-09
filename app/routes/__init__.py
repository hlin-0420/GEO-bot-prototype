import logging

from .api import api_blueprint
from .feedback_handler import feedback_routes
from .session_manager_extended import session_routes
from .timing_logger import timing_routes
from .ui import ui_blueprint
from .upload_handler import upload_routes

logger = logging.getLogger(__name__)


def register_blueprints(app):
    app.register_blueprint(api_blueprint)
    app.register_blueprint(ui_blueprint)
    app.register_blueprint(feedback_routes)
    app.register_blueprint(session_routes)
    app.register_blueprint(timing_routes)
    app.register_blueprint(upload_routes)
    logger.debug("Registered Flask blueprints")
