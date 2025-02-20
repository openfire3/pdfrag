import os
from typing import List, Tuple, Dict, Any, Optional
import uuid
import logging
from pathlib import Path
from datetime import datetime
import PyPDF2
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Клас для обробки PDF файлів та семантичного пошуку"""
    
    def __init__(self):
        """Ініціалізація процесора"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        
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

    def delete_collection(self, collection_name: str) -> bool:
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
        chunks_dir = Path(Config.CHUNKS_FOLDER)
        chunks_dir.mkdir(exist_ok=True)
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
        collection_name = f"pdf_{pdf_path.stem}_{uuid.uuid4().hex[:8]}"
        
        # Створення колекції в Qdrant
        if not self.qdrant_client.collection_exists(collection_name):
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
                for page_num, page in enumerate(reader.pages, start=start_page):
                    text = page.extract_text()
                    if not text.strip():
                        continue
                        
                    # Обробка тексту та створення ембедінгу
                    embedding = self.get_embedding(text)
                    
                    # Збереження в Qdrant
                    point_id = uuid.uuid4().int & ((1 << 64) - 1)
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                'page_num': page_num,
                                'text': text,
                                'metadata': metadata
                            }
                        )]
                    )
                    logger.info(f"Оброблено сторінку {page_num} з {total_pages}")
            
            # Видалення тимчасового чанка
            Path(chunk_path).unlink()
        
        logger.info(f"Завершено обробку файлу {pdf_path.name}")
        return collection_name

    def answer_query(self, query: str, collection_name: str, top_k: int = 5) -> str:
        """Відповідь на запит"""
        # Отримання ембедінгу запиту
        query_embedding = self.get_embedding(query)
        
        # Пошук релевантних фрагментів
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Формування контексту
        context = "\n\n".join([
            f"Сторінка {point.payload['page_num']}:\n{point.payload['text']}"
            for point in search_results
        ])
        
        # Генерація відповіді
        try:
            response = self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Ти є експертом з аналізу технічної документації та креслень. "
                                 "Надавай точні та конкретні відповіді на основі наданого контексту."
                    },
                    {
                        "role": "user",
                        "content": f"Контекст:\n{context}\n\nЗапит: {query}\n\n"
                                 f"Надай детальну відповідь використовуючи тільки інформацію з контексту."
                    }
                ]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Помилка при генерації відповіді: {str(e)}")
            raise