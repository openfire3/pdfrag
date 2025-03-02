import os
from openai import AsyncOpenAI
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
import asyncio
import re
import tiktoken

from app.config import Config
from app.logger_config import logger
from .sql_db_service import DatabaseService
from .embbedding_service import EmbeddingService
from .vector_db_service import QuadrantService
from .image_service import ImageService
from .vision_service import VisionService
from .conversation_service import ConversationService

sql_db_service = DatabaseService()
vector_db_service = QuadrantService()
embedding_service = EmbeddingService()
image_service = ImageService()
vision_service = VisionService()
conversation_service = ConversationService()

class ReplyService():
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = Config.SYSTEM_PROMPT
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.max_tokens = Config.MAX_TOKENS  # Using config value
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to stay within token limit"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])

    def format_content(self, results: List[Dict[str, Any]], max_tokens: int) -> str:
        """Format and truncate content to stay within token limits"""
        formatted_texts = []
        current_tokens = 0
        
        for result in results:
            text = result.payload['text']
            page_num = result.payload['page_num']
            
            # Format this piece of content
            formatted = f"Page {page_num}:\n{text}\n\n"
            tokens_needed = self.count_tokens(formatted)
            
            # Check if adding this would exceed our limit
            if current_tokens + tokens_needed > max_tokens:
                # If we're over limit, stop adding content
                break
            
            formatted_texts.append(formatted)
            current_tokens += tokens_needed
        
        return "".join(formatted_texts)

    def extract_page_numbers(self, query: str) -> List[int]:
        """Extract page numbers mentioned in the query"""
        patterns = [
            r'page\s*[:]?\s*(\d+)',  # page 10, page: 10
            r'p\.?\s*(\d+)',         # p.10, p 10
            r'pages?\s+(\d+)\s*(?:and|&|,)?\s*(\d+)?',  # page 9 and 10, pages 9,10
            r'\bpage\s+(\d+)\b'      # specifically match "page X"
        ]
        
        pages = set()
        for pattern in patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                # Add all capturing groups (they might contain different numbers)
                pages.update(int(num) for num in match.groups() if num)
        
        # Filter out invalid page numbers
        valid_pages = [p for p in sorted(pages) if p > 0]
        logger.info(f"Extracted page numbers: {valid_pages}")
        return valid_pages

    def rerank_results(self, query: str, results: List[dict], top_k: int = 5) -> List[dict]:
        """Re-rank search results using BM25"""
        try:
            # Prepare documents for BM25
            documents = [result.payload['text'] for result in results]
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = np.array(bm25.get_scores(tokenized_query))
            
            # Combine with semantic search scores
            semantic_scores = np.array([result.score for result in results])
            
            # Safely normalize scores to avoid division by zero
            if len(bm25_scores) > 0 and np.max(bm25_scores) != np.min(bm25_scores):
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
            else:
                bm25_scores = np.zeros_like(bm25_scores)
                
            if len(semantic_scores) > 0 and np.max(semantic_scores) != np.min(semantic_scores):
                semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores))
            else:
                semantic_scores = np.zeros_like(semantic_scores)
            
            # Combine scores (0.5 weight to semantic, 0.5 to BM25)
            combined_scores = 0.5 * semantic_scores + 0.5 * bm25_scores
            
            # Sort results by combined score
            ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
            reranked_results = [results[i] for i in ranked_indices]
            
            return reranked_results
        except Exception as e:
            logger.error(f"Error in re-ranking: {str(e)}")
            return results[:top_k]  # Fallback to original ranking
    
    def update_system_prompt(self, new_prompt):
        """Update system prompt template"""
        self.system_prompt = new_prompt

    def is_page_location_query(self, query: str) -> bool:
        """Check if the query is asking about page locations"""
        patterns = [
            r'which page(s)?\s+(?:contain|has|have|show)',
            r'where.*(?:find|located?|shown?)',
            r'on what page(s)?',
            r'in which page(s)?'
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    def is_vision_required(self, query: str) -> bool:
        """Check if the query requires visual analysis"""
        # Always use vision for queries about elements in drawings
        if any(term in query.lower() for term in ['how many', 'how much', 'count', 'where', 'location', 'find', 'show']):
            return True
        if self.extract_page_numbers(query):  # If specific pages are mentioned
            return True
        return False

    def format_page_list(self, pages: List[int]) -> str:
        """Format a list of pages into a readable string"""
        if not pages:
            return "No pages found"
        pages = sorted(set(pages))
        return f"Pages: {', '.join(map(str, pages))}"

    async def analyze_pages_for_content(self, query: str, collection_name: str, pages: List[int]) -> List[int]:
        """Analyze specific pages for content"""
        found_pages = []
        search_term = re.sub(r'which pages? contains? |where is |find |show me |on what pages? |in which pages? ', '', query.lower())
        
        for page in pages:
            image_path = os.path.join(Config.IMAGES_FOLDER, f'page_{page}.jpg')
            if os.path.exists(image_path):
                logger.info(f"Starting vision analysis for page {page}")
                # Змінюємо формат запиту для кращого розпізнавання
                vision_query = f"Analyze this technical drawing. {query}"
                analysis = await vision_service.analyze_image(image_path, vision_query)
                if analysis:
                    found_pages.append(page)
                    logger.info(f"Vision analysis for page {page}: {analysis}")
                    
        return found_pages

    async def answer(self, query: str, collection_name: str, session_id: str = None, page_range_start=None, page_range_end=None) -> str:
        try:
            # Get conversation context
            conversation_context = ""
            if (session_id):
                recent_messages = conversation_service.get_recent_context(session_id)
                if recent_messages:
                    conversation_context = "Recent conversation context:\n" + "\n".join([
                        f"User: {msg.query}\nAssistant: {msg.response}\n"
                        for msg in recent_messages
                    ])
            
            # Get explicitly mentioned pages
            mentioned_pages = self.extract_page_numbers(query)
            logger.info(f"Pages mentioned in query: {mentioned_pages}")

            # Get semantic search results first
            query_embedding = embedding_service.get_embedding(query)
            semantic_results = vector_db_service.search(
                collection_name, 
                query_embedding, 
                page_range_start, 
                page_range_end, 
                Config.TOP_K
            )
            
            # Re-rank semantic results
            reranked_results = self.rerank_results(query, semantic_results)
            semantic_pages = [result.payload['page_num'] for result in reranked_results]
            logger.info(f"Semantic search found pages: {semantic_pages}")
            
            # Get explicitly mentioned pages
            mentioned_pages = self.extract_page_numbers(query)
            logger.info(f"Pages mentioned in query: {mentioned_pages}")
            
            # Determine pages to analyze
            pages_to_analyze = mentioned_pages if mentioned_pages else semantic_pages[:Config.TOP_K]
            logger.info(f"Pages to analyze: {pages_to_analyze}")
            
            # Format content from semantic search
            all_text = self.format_content(reranked_results, self.max_tokens // 2)
            
            # Always perform vision analysis for relevant pages when dealing with technical elements
            visual_context = ""
            needs_vision = self.is_vision_required(query)
            logger.info(f"Vision analysis needed: {needs_vision}, Query: {query}")
            
            if needs_vision and pages_to_analyze:
                visual_analyses = []
                for page in pages_to_analyze:
                    image_path = os.path.join(Config.IMAGES_FOLDER, f'page_{page}.jpg')
                    if os.path.exists(image_path):
                        logger.info(f"Analyzing technical drawing for page {page}")
                        analysis = await vision_service.analyze_image(
                            image_path,
                            f"Analyze this technical drawing focusing on: {query}"
                        )
                        if analysis:
                            visual_analyses.append(f"Page {page} analysis:\n{analysis}")
                            logger.info(f"Vision analysis for page {page}: {analysis}")
                    else:
                        logger.warning(f"Image not found for page {page}")
                
                visual_context = "\n\n".join(visual_analyses) if visual_analyses else ""
                if visual_context:
                    visual_context = self.truncate_to_token_limit(visual_context, self.max_tokens // 3)

            # Calculate remaining tokens with safety buffer
            used_tokens = (
                self.count_tokens(all_text) +
                self.count_tokens(visual_context) +
                self.count_tokens(conversation_context) +
                self.count_tokens(self.system_prompt) +
                self.count_tokens(query) +
                100  # Safety buffer
            )
            
            remaining_tokens = max(100, Config.MAX_TOKENS - used_tokens)
            
            # Generate response with guaranteed positive token limit
            prompt_content = ""
            if conversation_context:
                prompt_content += f"{conversation_context}\n\n"
            prompt_content += f"Context from document:\n{all_text}\n\n"
            if visual_context:
                prompt_content += f"Visual analysis of technical drawings:\n{visual_context}\n\n"
            prompt_content += f"User query: {query}\n\n"

            completion = await self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt_content + "Provide a comprehensive response focusing on the specific information requested. Include all relevant locations and details found in the analysis."
                    }
                ],
                max_tokens=remaining_tokens
            )
            
            response_text = completion.choices[0].message.content
            logger.info(f"PDFRAG Final (GPT4-o) response:\n{response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise