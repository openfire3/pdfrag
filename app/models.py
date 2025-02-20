from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessedPDF:
    """Модель для зберігання інформації про оброблений PDF"""
    filename: str
    collection_name: str
    created_at: datetime
    pages_count: int
    size_bytes: int