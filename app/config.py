import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings"""
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    CHUNKS_FOLDER = BASE_DIR / "chunks"
    LOGS_FOLDER = BASE_DIR / "logs"
    IMAGES_FOLDER = BASE_DIR / "images"
    
    # Create required directories
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    CHUNKS_FOLDER.mkdir(exist_ok=True)
    LOGS_FOLDER.mkdir(exist_ok=True)
    IMAGES_FOLDER.mkdir(exist_ok=True)
    
    # API keys and database settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_PATH = os.getenv("QDRANT_PATH", "localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Model settings
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CHAT_MODEL = "gpt-4o"
    VISION_MODEL = "gemini-2.0-flash"  # Keep Gemini model
    
    # Rate limiting settings
    MAX_TOKENS_PER_REQUEST = 800  # Set to 800 tokens
    MAX_CONCURRENT_REQUESTS = 3
    
    # PDF processing settings
    CHUNK_SIZE = 50
    MAX_TOKENS = 800  # Also update this to match
    
    # System prompt template
    SYSTEM_PROMPT = """You are an expert assistant analyzing technical documentation and drawings.
    Analyze the provided content carefully and provide detailed, accurate responses.
    If you see technical drawings, focus on identifying and explaining key components and their relationships.
    Base your responses only on the information available in the provided content."""
    
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24))
    MAX_CONTENT_LENGTH = 1500 * 1024 * 1024  # 1.5 GB file limit
    UPLOAD_FOLDER = "uploads"
    
    TOP_K = 20