from flask import Flask
from app.routes import bp
import os

def create_app():
    
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))
    
    app.register_blueprint(bp)
    
    return app