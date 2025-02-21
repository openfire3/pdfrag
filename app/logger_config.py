import logging
import os
from datetime import datetime

# Создаём папку для логов, если её нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Имя файла логов с датой и временем
log_filename = datetime.now().strftime("server_%Y-%m-%d_%H-%M-%S.log")
log_file_path = os.path.join(log_dir, log_filename)

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("server_logger")
logger.info("Логгер настроен!")  # Тестовый лог

# Экспортируем логгер для использования в других модулях
__all__ = ["logger"]