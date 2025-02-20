import os
from typing import List, Tuple, Dict, Any
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, stop_after_attempt, wait_exponential

# Завантаження змінних середовища
load_dotenv()

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "200"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o")
        
    def split_pdf(self, pdf_path: str, output_folder: str = 'chunks') -> List[Tuple[str, int, int]]:
        """Розбиває PDF на частини"""
        Path(output_folder).mkdir(exist_ok=True)
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for start in range(0, total_pages, self.chunk_size):
                    writer = PyPDF2.PdfWriter()
                    end = min(start + self.chunk_size, total_pages)
                    
                    for page_num in range(start, end):
                        writer.add_page(reader.pages[page_num])
                        
                    chunk_filename = Path(output_folder) / f'chunk_{start+1}_{end}.pdf'
                    with open(chunk_filename, 'wb') as f_out:
                        writer.write(f_out)
                    chunks.append((str(chunk_filename), start+1, end))
                    
                logger.info(f"PDF розбито на {len(chunks)} частин")
                return chunks
                
        except Exception as e:
            logger.error(f"Помилка при розбитті PDF: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embedding(self, text: str) -> List[float]:
        """Отримує ембедінг тексту з OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Помилка при отриманні ембедінгу: {str(e)}")
            raise

    def save_to_qdrant(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Зберігає ембедінг в Qdrant"""
        point_id = uuid.uuid4().int & ((1 << 64) - 1)
        
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )]
            )
            logger.info(f"Точку {point_id} додано до колекції {collection_name}")
        except Exception as e:
            logger.error(f"Помилка при збереженні в Qdrant: {str(e)}")
            raise

    def process_text_chunk(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Розбиває текст на частини, що не перевищують ліміт токенів"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word.encode('utf-8'))
            if current_length + word_length > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_pdf(self, pdf_path: str) -> str:
        """Обробляє PDF: розбиває на частини, створює ембедінги, зберігає в Qdrant"""
        base_name = Path(pdf_path).stem
        collection_name = f"{base_name}_{uuid.uuid4().hex}"
        
        # Створення колекції
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Створено колекцію: {collection_name}")
        
        chunks = self.split_pdf(pdf_path)
        
        for chunk_file, start_page, end_page in chunks:
            with open(chunk_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=start_page):
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        logger.warning(f"Сторінка {page_num}: немає тексту")
                        continue
                        
                    # Розбиваємо текст на частини, якщо він завеликий
                    text_chunks = self.process_text_chunk(page_text)
                    
                    for idx, text_chunk in enumerate(text_chunks):
                        embedding = self.get_embedding(text_chunk)
                        metadata = {
                            "source_pdf": Path(pdf_path).name,
                            "chunk_file": Path(chunk_file).name,
                            "page": page_num,
                            "chunk_index": idx,
                            "page_text": text_chunk
                        }
                        self.save_to_qdrant(collection_name, embedding, metadata)
                        
        return collection_name

    def query_pdf(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Шукає відповідні документи в Qdrant"""
        query_embedding = self.get_embedding(query)
        
        try:
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            return search_result
        except Exception as e:
            logger.error(f"Помилка при пошуку: {str(e)}")
            raise

    def generate_answer(self, context: str, query: str) -> str:
        """Генерує відповідь використовуючи GPT-4o"""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "Ти є експертом у аналізі креслень та технічної документації."},
                    {"role": "user", "content": f"Використовуючи наданий контекст:\n{context}\n\nЗапит: {query}\n\nДай детальну відповідь:"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Помилка при генерації відповіді: {str(e)}")
            raise

    def answer_query(self, query: str, collection_name: str) -> str:
        """Отримує відповідь на запит"""
        results = self.query_pdf(query, collection_name)
        context = "\n".join([
            f"Файл: {r.payload['chunk_file']}, "
            f"Сторінка: {r.payload['page']}\n"
            f"Текст: {r.payload['page_text']}"
            for r in results
        ])
        return self.generate_answer(context, query)

def main():
    processor = PDFProcessor()
    pdf_path = "fs_small.pdf"
    
    try:
        collection_name = processor.process_pdf(pdf_path)
        logger.info(f"Створена колекція: {collection_name}")
        
        user_query = "find cam"
        response = processor.answer_query(user_query, collection_name)
        logger.info(f"Відповідь на запит: {response}")
        
    except Exception as e:
        logger.error(f"Помилка в основному процесі: {str(e)}")

if __name__ == "__main__":
    main()