from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import asyncio
from functools import wraps
import uuid

from .config import Config
from .logger_config import logger
from .services.pdf_handler_service import PDFHandler
from .services.reply_service import ReplyService
from .services.vector_db_service import QuadrantService
from .services.image_service import ImageService
from .services.conversation_service import ConversationService, Message

bp = Blueprint('main', __name__)

pdf_processor = PDFHandler()
reply_service = ReplyService()
vector_db_service = QuadrantService()
image_service = ImageService()
conversation_service = ConversationService()

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

def get_or_create_session_id():
    """Get existing session ID or create new one"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@bp.route('/')
def index():
    collections = vector_db_service.get_collections()
    session_id = get_or_create_session_id()
    history = conversation_service.get_conversation_history(session_id)
    return render_template('index.html', collections=collections, history=history)

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
        session_id = get_or_create_session_id()
        
        # Get relevant pages before processing query
        relevant_pages = reply_service.extract_page_numbers(data['query'])
        
        # Process query with session context
        response = await reply_service.answer(
            query=data['query'],
            collection_name=data['collection'],
            session_id=session_id
        )
        
        # Store in conversation history
        conversation_service.add_message(
            session_id=session_id,
            query=data['query'],
            response=response,
            pages_used=relevant_pages,
            collection_name=data['collection']
        )
        
        return jsonify({
            'response': response,
            'conversation_id': session_id
        })
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

@bp.route('/history', methods=['GET'])
def get_history():
    """Get conversation history for current session"""
    session_id = get_or_create_session_id()
    history = conversation_service.get_conversation_history(session_id)
    return jsonify({
        'history': [
            {
                'timestamp': msg.timestamp.isoformat(),
                'query': msg.query,
                'response': msg.response,
                'pages_used': msg.pages_used,
                'collection_name': msg.collection_name
            }
            for msg in history
        ]
    })