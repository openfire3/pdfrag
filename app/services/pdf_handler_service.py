from typing import List, Tuple
import uuid
from pathlib import Path
from datetime import datetime
import PyPDF2
import tiktoken
from nltk.tokenize import sent_tokenize
import nltk
from pdf2image import convert_from_path
from PIL import Image
import base64
import io

from app.config import Config
from app.logger_config import logger
from .vector_db_service import QuadrantService
from .sql_db_service import DatabaseService
from .embbedding_service import EmbeddingService

Image.MAX_IMAGE_PIXELS = None

nltk.download('punkt')

vector_db_service = QuadrantService()
sql_db_service = DatabaseService()
emmedding_service = EmbeddingService()

class PDFHandler():
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model(Config.EMBEDDING_MODEL)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def split_text(self, text: str, max_tokens: int = 8192) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self.count_tokens(para)
            
            if para_tokens > max_tokens:
                sentences = sent_tokenize(para)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    if current_length + sentence_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                        if sentence_tokens > max_tokens:
                            truncated = self.truncate_text(sentence, max_tokens)
                            chunks.append(truncated)
                        else:
                            current_chunk.append(sentence)
                            current_length += sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_tokens
            else:
                if current_length + para_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                current_chunk.append(para)
                current_length += para_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(text)[:max_tokens]
        return self.encoder.decode(tokens)
        
    def split_pdf(self, pdf_path: str) -> List[Tuple[str, int, int]]:
        chunks_dir = Path(Config.CHUNKS_FOLDER) / Path(pdf_path).stem
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for start in range(0, total_pages, Config.CHUNK_SIZE):
                    writer = PyPDF2.PdfWriter()
                    end = min(start + Config.CHUNK_SIZE, total_pages)
                    
                    for i in range(start, end):
                        writer.add_page(reader.pages[i])
                        
                    chunk_path = chunks_dir / f'chunk_{start+1}_{end}_{uuid.uuid4().hex[:8]}.pdf'
                    with open(chunk_path, 'wb') as chunk_file:
                        writer.write(chunk_file)
                    chunks.append((str(chunk_path), start+1, end))
                    logger.info(f"Created chunk {chunk_path} (page {start+1}-{end})")
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting PDF: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str, collection_name: str) -> str:
        pdf_path = Path(pdf_path)
        file_size = pdf_path.stat().st_size
        logger.info(f"Starting process {pdf_path.name} (size: {file_size/1024/1024:.2f} MB)")
        
        vector_db_service.create_collection(collection_name)
        sql_db_service.create_table(collection_name)

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
        logger.info(f"Extracting images...")
        images = convert_from_path(
            pdf_path
        )
        logger.info(f"{len(images)} images succesfully extracted")

        metadata = {
            'filename': collection_name,
            'created_at': datetime.now().isoformat(),
            'pages_count': total_pages,
            'size_bytes': file_size
        }

        chunks = self.split_pdf(str(pdf_path))
        for chunk_path, start_page, end_page in chunks:
            with open(chunk_path, 'rb') as chunk_file:
                reader = PyPDF2.PdfReader(chunk_file)
                total_pages = len(reader.pages)
                logger.info(f"Creating embedding for chunk {chunk_path}")   
                
                for page_num, page in enumerate(reader.pages, start=start_page):
                    text = page.extract_text()
                    
                    if not text.strip():
                        continue

                    token_count = self.count_tokens(text)
                    if token_count > 8192:
                        sub_chunks = self.split_text(text)
                    else:
                        sub_chunks = [text]
                    
                    for chunk_part, chunk_text in enumerate(sub_chunks, 1):
                        chunk_text = chunk_text.strip()
                        if not chunk_text:
                            continue

                        chunk_token_count = self.count_tokens(chunk_text)
                        if chunk_token_count > 8192:
                            chunk_text = self.truncate_text(chunk_text, 8192)
                        
                        embedding = emmedding_service.get_embedding(chunk_text)
                        
                        vector_db_service.save_point(collection_name, embedding, page_num, chunk_text, chunk_part, len(sub_chunks), metadata)
                        
                        # Convert image to desired format and quality
                        img = images[page_num - 1]
                        image_format="JPEG"
                        quality=85
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format=image_format, quality=quality)
                        img_binary = img_buffer.getvalue()
                        
                        # Create base64 encoding for the analyze_image function
                        img_base64 = base64.b64encode(img_binary).decode('utf-8')
                        # image_bytes = images[page_num - 1].tobytes()
                        sql_db_service.save_record(collection_name, page_num, text, img_base64)
                        
                        logger.info(f"Processed part {chunk_part} of page {page_num}: {page_num - start_page + 1} of {total_pages}")
                        
        logger.info(f"File processed: {pdf_path.name}")
        return collection_name
