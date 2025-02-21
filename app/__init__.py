from flask import Flask
from app.routes import bp
import os
from .logger_config import logger


def create_app():
    
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))
    logger.info("Створено Flask додаток")
    app.register_blueprint(bp)
    
    return app