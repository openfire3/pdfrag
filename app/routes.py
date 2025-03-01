from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import asyncio
from functools import wraps

from .config import Config
from .logger_config import logger
from .services.pdf_handler_service import PDFHandler
from .services.reply_service import ReplyService
from .services.vector_db_service import QuadrantService
from .services.image_service import ImageService

bp = Blueprint('main', __name__)

pdf_processor = PDFHandler()
reply_service = ReplyService()
vector_db_service = QuadrantService()
image_service = ImageService()

async def run_async(coro):
    try:
        return await coro
    except Exception as e:
        logger.error(f"Async error: {str(e)}")
        raise

def async_route(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(run_async(f(*args, **kwargs)))
    return wrapper

@bp.route('/')
def index():
    collections = vector_db_service.get_collections()
    return render_template('index.html', collections=collections)

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'File not found'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file chosen'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF supported'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"File {file.filename} is ready for processing")
        
        collection_name = pdf_processor.process_pdf(filepath)
        
        return jsonify({
            'success': True,
            'collection_name': collection_name,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/query', methods=['POST'])
@async_route
async def query():
    data = request.get_json()
    logger.info(f"Received query: {data.get('query')} for collection: {data.get('collection')}")
    
    if not data or 'query' not in data or 'collection' not in data:
        return jsonify({'error': 'Bad request'}), 400
    
    try:
        # Properly await the async response
        response = await reply_service.answer(
            data['query'], 
            data['collection']
        )
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in query route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/get_prompt', methods=['GET'])
def get_prompt():
    """Get current system prompt"""
    return jsonify({'prompt': Config.SYSTEM_PROMPT})

@bp.route('/update_prompt', methods=['POST'])
def update_prompt():
    """Update system prompt"""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        reply_service.update_system_prompt(data['prompt'])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500