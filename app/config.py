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
    IMAGES_FOLDER = BASE_DIR / "images"
    
    
    # Створюємо необхідні директорії
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    CHUNKS_FOLDER.mkdir(exist_ok=True)
    LOGS_FOLDER.mkdir(exist_ok=True)
    IMAGES_FOLDER.mkdir(exist_ok=True)
    
    # API ключі та налаштування бази
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_PATH = os.getenv("QDRANT_PATH", "localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Налаштування моделей
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o"  # опенаі модель
    
    SYSTEM_PROMPT = "You are an expert in analyzing technical documentation and drawings. You recieve a list of pages with descriptions from document. Provide accurate and specific answers based on image analyzis of this page."
    SYSTEM_PROMPT_FOR_IMAGE_ANALYSIS = """You are a retrieval augmented generation agent specialized in analyzing technical drawings and diagrams of electrical infrastructure systems. When analyzing drawings:
    1. Be thorough and methodical - scan the entire drawing systematically
    2. Pay special attention to:
       - All device symbols and their labels (CAM, FPD, decoders, etc.)
       - Room numbers and names
       - Device locations and their spatial relationships
       - Connections and wiring between devices
       - Notes, legends, and annotations
    
    Base your responses solely on the visible content in the drawings - do not make assumptions or add external information. Always provide specific page references and be explicit about uncertainty if something is unclear."""
    
    # Налаштування обробки PDF
    CHUNK_SIZE = 25
    MAX_TOKENS = 8000
    
    # Налаштування Flask
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24))
    MAX_CONTENT_LENGTH = 1500 * 1024 * 1024  # 1.5 GB ліміт на файли
    UPLOAD_FOLDER = "uploads"
    
    TOP_K = 5