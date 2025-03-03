# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Указываем порт, который будет использовать приложение
EXPOSE 5000

# Команда для запуска приложения
CMD ["flask", "run", "--host=0.0.0.0"]