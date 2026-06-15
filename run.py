import logging

from app import config
from app.main import create_app

logging.basicConfig(level=logging.DEBUG if config.DEBUG_MODE else logging.INFO)

app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG_MODE, port=config.PORT)
