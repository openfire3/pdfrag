from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from .pdf_processor import PDFProcessor
from .models import ProcessedPDF
from .config import Config
from datetime import datetime

bp = Blueprint('main', __name__)
processor = PDFProcessor()

@bp.route('/')
def index():
    """Головна сторінка"""
    # Отримуємо список оброблених PDF
    collections = processor.get_collections()
    return render_template('index.html', collections=collections)

@bp.route('/upload', methods=['POST'])
def upload_file():
    """Завантаження та обробка PDF"""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не знайдено'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не вибрано'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Підтримуються тільки PDF файли'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Обробка PDF
        collection_name = processor.process_pdf(filepath)
        
        # Зберігаємо інформацію про оброблений PDF
        pdf_info = ProcessedPDF(
            filename=filename,
            collection_name=collection_name,
            created_at=datetime.now(),
            pages_count=len(PyPDF2.PdfReader(filepath).pages),
            size_bytes=os.path.getsize(filepath)
        )
        
        return jsonify({
            'success': True,
            'collection_name': collection_name,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/query', methods=['POST'])
def query():
    """Обробка запиту до PDF"""
    data = request.get_json()
    if not data or 'query' not in data or 'collection' not in data:
        return jsonify({'error': 'Невірний запит'}), 400
    
    try:
        response = processor.answer_query(data['query'], data['collection'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500