from flask import Flask
import os
from .logger_config import logger
from .routes import bp

def create_app():
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))
    app.config['PROPAGATE_EXCEPTIONS'] = True  # Ensure async exceptions are properly handled
    app.config['SECRET_KEY'] = os.urandom(24)  # Required for session handling
    
    # Enable CORS to allow async requests
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    logger.info("Created Flask application with async support")
    app.register_blueprint(bp)
    return app