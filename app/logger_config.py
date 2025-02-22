import logging
import os
from datetime import datetime
import sys

# Важно: Singleton-подход через модуль
class _LoggerSingleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        # Настройка пути и имени файла
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        self.log_path = os.path.join(log_dir, log_filename)

        # Создаем логгер
        self.logger = logging.getLogger("project_main_logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Отключаем передачу корневому логгеру

        # Проверяем, нет ли уже обработчиков
        if not self.logger.handlers:
            # Форматирование
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Обработчик для файла (UTF-8)
            file_handler = logging.FileHandler(
                self.log_path, 
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Опционально: вывод в консоль
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

# Экспортируем объект логгера
logger = _LoggerSingleton().logger