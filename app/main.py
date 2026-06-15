import logging
import os

from flask import Flask

from app import config
from app.routes import register_blueprints

logger = logging.getLogger(__name__)


def create_app():
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
    static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

    app = Flask(__name__, template_folder=template_path, static_folder=static_path)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE

    register_blueprints(app)
    logger.debug("Flask app created")

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if config.DEBUG_MODE else logging.INFO)
    create_app().run(debug=config.DEBUG_MODE, port=config.PORT)
