import os
from openai import OpenAI

from app.config import Config
from app.logger_config import logger
from .sql_db_service import DatabaseService
from .embbedding_service import EmbeddingService
from .vector_db_service import QuadrantService

sql_db_service = DatabaseService()
vector_db_service = QuadrantService()
embedding_service = EmbeddingService()

class ReplyService():
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def answer(self, query: str, collection_name: str, page_range_start, page_range_end, search_word) -> str:
        
        top_k = Config.TOP_K
        prompt_start = ""
        if search_word:
            logger.info(f"Text search in {collection_name}")
            text_results = sql_db_service.text_search(collection_name, search_word)
            text_search = "\n\n".join([
            f"Page number {row['page_number']}:\n{row['text']}"
            for row in text_results[:top_k]
            ])
            prompt_start = f"""Pages: {text_search}.
            Search for element {search_word}. Most likely you'll find it on each provided page"""
        else:
            logger.info(f"Semantic search in {collection_name}")

            query_embedding = embedding_service.get_embedding(query)
            
            search_results = vector_db_service.search(collection_name, query_embedding, page_range_start, page_range_end, top_k)
            
            logger.info(f"Got {len(search_results)} results")

            semantic_search = "\n\n".join([
                f"Page number {point.payload['page_num']}:\n{point.payload['text']}"
                for point in search_results
            ])
            
            prompt_start = f"""Pages: {semantic_search}"""

        try:
            logger.info(f"Requesting model {Config.CHAT_MODEL}")
            response = self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        # "content": "Ти є експертом з аналізу технічної документації та креслень. Надавай точні та конкретні відповіді на основі наданого контексту."
                        "content": "You are an expert in analyzing technical documentation and drawings. You recieve a list of pages from document. Provide accurate and specific answers based on the text you can read on pages."
                    },
                    {
                        "role": "user",
                        # "content": f"Контекст:\n{context}\n\nЗапит: {query}\n\n"
                        #          f"Надай детальну відповідь використовуючи тільки інформацію з контексту."
                        "content": f"{prompt_start}\n\nQuery: {query}\n\n"
                                  f"Provide a detailed answer using only the information from the context. List pages that were analyzed and don't forget to tell on which pages you found relevant info.. Reply with structured HTML, start only from opening container <div> from the very beginning ending with closing </div> tag."
                                  f"""Example:
                                  <div>
                                  <p>I have recieved and analyzed these pages: <b>2</b>, <b>12</b>, <b>33</b>, ...(list all pages in context)</p>
                                  <p>On these pages I found relevant info: </p>
                                  <h3><b>Page 33</b></h3>
                                  <p> There are some ...</p>
                                  <h3><b>Page 45</b></h3>
                                  <p> Here I found...</p>
                                  ...
                                  <hr/>
                                  <h2><b>Conclusion</b></h2>
                                  <p>There re some...</p>"""
                    }
                ]
            )
            response_html = response.choices[0].message.content[7:-3]
            logger.info(f"Got response: {response_html}")
            return response_html
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise