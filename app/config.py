import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Конфігурація додатку"""
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    CHUNKS_FOLDER = BASE_DIR / "chunks"
    LOGS_FOLDER = BASE_DIR / "logs"
    
    # Створюємо необхідні директорії
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    CHUNKS_FOLDER.mkdir(exist_ok=True)
    LOGS_FOLDER.mkdir(exist_ok=True)
    
    # API ключі та налаштування бази
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_PATH = os.getenv("QDRANT_PATH", "localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Налаштування моделей
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o"  # опенаі модель
    
    # Налаштування обробки PDF
    CHUNK_SIZE = 50
    MAX_TOKENS = 8000
    
    # Налаштування Flask
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24))
    MAX_CONTENT_LENGTH = 1500 * 1024 * 1024  # 1.5 GB ліміт на файли
    UPLOAD_FOLDER = "uploads"
    
    TOP_K = 20