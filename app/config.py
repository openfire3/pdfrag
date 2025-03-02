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
    CHAT_MODEL = "gpt-4o" # Keep gpt-4o model
    VISION_MODEL = "gemini-2.0-flash"  # Keep Gemini model
    
    # Rate limiting settings
    MAX_TOKENS_PER_REQUEST = 1600  # Increased from 800
    MAX_CONCURRENT_REQUESTS = 3
    
    # PDF processing settings
    CHUNK_SIZE = 50
    MAX_TOKENS = 1600  # Also increased to match
    TOP_K = 15  # Show up to 15 relevant pages
    
    # System prompt template
    SYSTEM_PROMPT = """You are a retrieval augmented generation agent specialized in analyzing technical drawings and diagrams of electrical infrastructure systems. When analyzing drawings:
    1. Be thorough and methodical - scan the entire drawing systematically
    2. Pay special attention to:
       - All device symbols and their labels (CAM, FPD, decoders, etc.)
       - Room numbers and names
       - Device locations and their spatial relationships
       - Connections and wiring between devices
       - Notes, legends, and annotations
    3. When counting elements:
       - Count ALL instances, not just the obvious ones
       - Verify the count multiple times
       - Specify exact locations of each element
    4. When describing locations:
       - Give precise room numbers/names
       - Describe relative positions (e.g., "near the north wall", "adjacent to")
       - Reference nearby landmarks or other devices
    
    Base your responses solely on the visible content in the drawings - do not make assumptions or add external information. Always provide specific page references and be explicit about uncertainty if something is unclear."""
    
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24))
    MAX_CONTENT_LENGTH = 1500 * 1024 * 1024  # 1.5 GB file limit
    UPLOAD_FOLDER = "uploads"