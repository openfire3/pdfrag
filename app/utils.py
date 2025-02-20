import logging
from functools import wraps
from time import time
from typing import Callable, Any

def setup_logger(name: str) -> logging.Logger:
    """Налаштування логера"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Файловий хендлер
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольний хендлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def timing_decorator(f: Callable) -> Callable:
    """Декоратор для вимірювання часу виконання функцій"""
    @wraps(f)
    def wrap(*args, **kwargs) -> Any:
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        logger.info(f'Функція: {f.__name__}, час: {te-ts:.2f} сек')
        return result
    return wrap

logger = setup_logger(__name__)