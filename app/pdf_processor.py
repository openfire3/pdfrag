import os
from typing import List, Tuple, Dict, Any, Optional
import uuid
from pathlib import Path
from datetime import datetime
import PyPDF2
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import Config
from .logger_config import logger
import tiktoken
from nltk.tokenize import sent_tokenize
import nltk
import re

nltk.download('punkt')


class PDFProcessor:
    """Клас для обробки PDF файлів та семантичного пошуку"""
    
    def __init__(self):
        """Ініціалізація процесора"""
        self.encoder = tiktoken.encoding_for_model(Config.EMBEDDING_MODEL)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            # host=Config.QDRANT_HOST,
            # port=Config.QDRANT_PORT
            url=Config.QDRANT_PATH,
            api_key=Config.QDRANT_API_KEY
        )
    
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
                logger.info(f"Параграф має більше токенів, ніж ліміт")
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
        
    def get_collections(self) -> List[Dict[str, Any]]:
        """Отримання списку всіх колекцій з метаданими"""
        try:
            collections = self.qdrant_client.get_collections()
            result = []
            for collection in collections.collections:
                metadata = self._get_collection_metadata(collection.name)
                if metadata:
                    result.append({
                        'collection_name': collection.name,
                        'filename': metadata.get('filename'),
                        'created_at': metadata.get('created_at'),
                        'pages_count': metadata.get('pages_count'),
                        'size_bytes': metadata.get('size_bytes')
                    })
            return result
        except Exception as e:
            logger.error(f"Помилка при отриманні колекцій: {str(e)}")
            raise

    def _get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Отримання метаданих колекції"""
        try:
            points = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1
            )[0]
            if points:
                return points[0].payload.get('metadata', {})
            return None
        except Exception:
            return None

    # def delete_collection(self, collection_name: str) -> bool:
        """Видалення колекції"""
        try:
            self.qdrant_client.delete_collection(collection_name)
            logger.info(f"Колекцію {collection_name} видалено")
            return True
        except Exception as e:
            logger.error(f"Помилка при видаленні колекції {collection_name}: {str(e)}")
            return False

    def split_pdf(self, pdf_path: str) -> List[Tuple[str, int, int]]:
        """Розбиття PDF на частини"""
        chunks_dir = Path(Config.CHUNKS_FOLDER) / Path(pdf_path).stem
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Розбиваємо на частини по CHUNK_SIZE сторінок
                for start in range(0, total_pages, Config.CHUNK_SIZE):
                    writer = PyPDF2.PdfWriter()
                    end = min(start + Config.CHUNK_SIZE, total_pages)
                    
                    for i in range(start, end):
                        writer.add_page(reader.pages[i])
                        
                    chunk_path = chunks_dir / f'chunk_{start+1}_{end}_{uuid.uuid4().hex[:8]}.pdf'
                    with open(chunk_path, 'wb') as chunk_file:
                        writer.write(chunk_file)
                    chunks.append((str(chunk_path), start+1, end))
                    logger.info(f"Створено чанк {chunk_path} (сторінки {start+1}-{end})")
                
            return chunks
            
        except Exception as e:
            logger.error(f"Помилка при розбитті PDF: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embedding(self, text: str) -> List[float]:
        
        """Отримання ембедінгу тексту"""
        try:
            response = self.client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Помилка при отриманні ембедінгу: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> str:
        """Обробка PDF файлу"""
        pdf_path = Path(pdf_path)
        file_size = pdf_path.stat().st_size
        logger.info(f"Початок обробки файлу {pdf_path.name} (розмір: {file_size/1024/1024:.2f} MB)")
        
        # Створення унікальної назви колекції
        collection_name = f"pdf123_{pdf_path.stem}_{uuid.uuid4().hex[:8]}"
        
        # Створення колекції в Qdrant
        # if not self.qdrant_client.collection_exists(collection_name):
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        )
        
        # Отримання інформації про PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
        
        # Збереження метаданих
        metadata = {
            'filename': pdf_path.name,
            'created_at': datetime.now().isoformat(),
            'pages_count': total_pages,
            'size_bytes': file_size
        }
        
        # Обробка чанків
        chunks = self.split_pdf(str(pdf_path))
        for chunk_path, start_page, end_page in chunks:
            with open(chunk_path, 'rb') as chunk_file:
                reader = PyPDF2.PdfReader(chunk_file)
                total_pages = len(reader.pages)
                logger.info(f"Створюємо ембедінг для чанку {chunk_path}")   
                
                for page_num, page in enumerate(reader.pages, start=start_page):
                    text = page.extract_text()
                    
                    if not text.strip():
                        continue
                    
                    # Разбиваем текст на подчанки при необходимости
                    token_count = self.count_tokens(text)
                    if token_count > 8192:
                        sub_chunks = self.split_text(text)
                    else:
                        sub_chunks = [text]
                    
                    for chunk_part, chunk_text in enumerate(sub_chunks, 1):
                        chunk_text = chunk_text.strip()
                        if not chunk_text:
                            continue
                            
                        # Проверяем размер подчанка
                        chunk_token_count = self.count_tokens(chunk_text)
                        if chunk_token_count > 8192:
                            chunk_text = self.truncate_text(chunk_text, 8192)
                        
                        # Создаем эмбеддинг
                        embedding = self.get_embedding(chunk_text)
                        
                        # Сохраняем в Qdrant
                        point_id = uuid.uuid4().int & ((1 << 64) - 1)
                        self.qdrant_client.upsert(
                            collection_name=collection_name,
                            points=[models.PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    'page_num': page_num,
                                    'text': chunk_text,
                                    'chunk_part': chunk_part,
                                    'total_chunks': len(sub_chunks),
                                    'metadata': metadata
                                }
                            )]
                        )
                        logger.info(f"Оброблено частину {chunk_part} сторінки {page_num}: {page_num - start_page + 1} з {total_pages}")
        logger.info(f"Завершено обробку файлу {pdf_path.name}")
        return collection_name


    def answer(self, query: str, collection_name: str, page_range_start, page_range_end) -> str:
        top_k = Config.TOP_K
        """Відповідь на запит"""
        # Отримання ембедінгу запиту
        query_embedding = self.get_embedding(query)
        
        query_filter = None
        range_params = {}
        if page_range_start is not None:
            range_params["gte"] = page_range_start
        if page_range_end is not None:
            range_params["lte"] = page_range_end

        # Добавляем фильтр только если задан хотя бы один параметр
        if range_params:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_num",
                        range=models.Range(**range_params)
                    )
                ]
            )
        # Пошук релевантних фрагментів
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,  
            limit=top_k
        )
        logger.info(f"Отримано {len(search_results)} результатів пошуку")
        
        # Формування контексту
        context = "\n\n".join([
            # f"Сторінка {point.payload['page_num']}:\n{point.payload['text']}"
            f"Page number {point.payload['page_num']}:\n{point.payload['text']}"
            for point in search_results
        ])
        logger.info(f"Контекст: {context}")
        # Генерація відповіді
        try:
            logger.info(f"Відправляємо запит на генерацію відповіді")
            response = self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        # "content": "Ти є експертом з аналізу технічної документації та креслень. Надавай точні та конкретні відповіді на основі наданого контексту."
                        "content": "You are an expert in analyzing technical documentation and drawings. Provide accurate and specific answers based on the text you can read on pages."
                    },
                    {
                        "role": "user",
                        # "content": f"Контекст:\n{context}\n\nЗапит: {query}\n\n"
                        #          f"Надай детальну відповідь використовуючи тільки інформацію з контексту."
                        "content": f"Pages:\n{context}\n\nQuery: {query}\n\n"
                                  f"Provide a detailed answer using only the information from the context. List pages that were analyzed and don't forget to tell on which pages you found relevant info.. Reply with structured HTML, start only from opening container <div> from the very beginning ending with closing </div> tag."
                                  f"""Example:
                                  <div>
                                  <p>I have recieved and analyzed these pages: 2, 12, 33, ...(list all pages in context)</p>
                                  <p>On these pages I found relevant info: </p>
                                  <h3>Page 33</h3>
                                  <p> There are some ...</p>
                                  <h3>Page 45</h3>
                                  <p> Here I found...</p>
                                  ...
                                  <hr/>
                                  <h2>Conclusion</h2>
                                  <p>There re some...</p>"""
                    }
                ]
            )
            response_html = response.choices[0].message.content[7:-3]
            logger.info(f"Отримано відповідь: {response_html}")
            return response_html
            
        except Exception as e:
            logger.error(f"Помилка при генерації відповіді: {str(e)}")
            raise