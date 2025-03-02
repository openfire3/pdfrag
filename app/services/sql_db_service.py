import os
from app.config import Config
from app.logger_config import logger
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

class DatabaseService:
    def __init__(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
        
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            self.cursor = conn.cursor()
        except Exception as e:
            logger.error(f"Error connecting to SQL database: {str(e)}")
    
    def create_table(self, table_name):
        try:
            self.cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        page_number INTEGER NOT NULL,
                        content TEXT,
                        image BYTEA,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logger.info(f"Table {table_name} has been created")    
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")   
    
    def save_record(self, table_name, page_num, text, image):
        try:
            logger.info(f"Adding to PostgreSQL - Table: {table_name}, Page: {page_num}")
            self.cursor.execute(
                f"INSERT INTO {table_name} (page_number, content, image) VALUES (%s, %s, %s)",
                (page_num, text, image)
            )
            logger.info(f"Successfully added to PostgreSQL - Table: {table_name}, Page: {page_num}")
        except Exception as e:
            logger.error(f"Error saving to PostgreSQL - Table: {table_name}, Page: {page_num}: {str(e)}")
            raise
        
    def text_search(self, db_name, search_word):
        """Simple exact text matching search, used only for direct matching"""
        try:
            # Extract main search terms if they exist (e.g., "CAM", "FPD")
            search_term = search_word.lower()
            
            query = f"SELECT * FROM {db_name} WHERE LOWER(content) LIKE %s"
            params = [f"%{search_term}%"]
            
            logger.info(f"Executing SQL text search with term: {search_term}")
            self.cursor.execute(query, params)
            
            rows = self.cursor.fetchall()
            results = []
            
            if rows:
                for row in rows:
                    id, page_number, content, image, ts = row
                    results.append({
                        "page_number": page_number, 
                        "text": content
                    })
                logger.info(f"Text search found {len(results)} results on pages: {[r['page_number'] for r in results]}")
            else:
                logger.info("Text search found no results")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []
