from flask import Flask

def create_app():
    app = Flask(__name__)
    # app.config['MAX_CONTENT_LENGTH'] = 1500 * 1024 * 1024  # 1.5GB
    # app.config['UPLOAD_FOLDER'] = 'uploads'
    
    return app