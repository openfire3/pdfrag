from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from .config import Config
from .logger_config import logger
from .services.pdf_handler_service import PDFHandler
from .services.reply_service import ReplyService
from .services.vector_db_service import QuadrantService

bp = Blueprint('main', __name__)

pdf_processor = PDFHandler()
reply_service = ReplyService()
vector_db_service = QuadrantService()

@bp.route('/')
def index():
    collections = vector_db_service.get_collections()
    return render_template('index.html', collections=collections)

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'File not found'}), 400
        
    file = request.files['file']
    collection_name = request.form.get('collection_name')
    if file.filename == '':
        return jsonify({'error': 'No file chosen'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF supported'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"File {file.filename} is ready to processing")
        # Обробка PDF
        collection_name = pdf_processor.process_pdf(filepath, collection_name)
        
        return jsonify({
            'success': True,
            'collection_name': collection_name,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    logger.info(f"Recieved query: {data.get('query')} by file: {data.get('collection')}")
    if not data or 'query' not in data or 'collection' not in data:
        return jsonify({'error': 'Bad request'}), 400
    
    try:
        range_start = data.get('pageRangeStart') or None
        range_end = data.get('pageRangeEnd') or None
        searchWord = data.get('searchedElement') or None
        query = data.get('query') or ""
        response = reply_service.answer(query, data['collection'],range_start, range_end, searchWord)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500