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

sql_db_service = DatabaseService()
vector_db_service = QuadrantService()
embedding_service = EmbeddingService()
image_service = ImageService()
vision_service = VisionService()

class ReplyService():
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = Config.SYSTEM_PROMPT
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.max_tokens = 800  # Updated to match config
    
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
        # Look for patterns like "page 10" or "page: 10" or "p.10" or "p 10"
        patterns = [
            r'page\s*[:]?\s*(\d+)',
            r'p\.?\s*(\d+)'
        ]
        
        pages = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            pages.extend([int(p) for p in matches])
        
        return pages if pages else []

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
    
    async def answer(self, query: str, collection_name: str, page_range_start=None, page_range_end=None) -> str:
        """Generate answer based on document content and vision analysis"""
        try:
            # Extract specifically mentioned pages from query
            mentioned_pages = self.extract_page_numbers(query)
            logger.info(f"Pages mentioned in query: {mentioned_pages}")

            # Get query embedding and search vector DB
            query_embedding = embedding_service.get_embedding(query)
            semantic_results = vector_db_service.search(
                collection_name, 
                query_embedding, 
                page_range_start, 
                page_range_end, 
                Config.TOP_K
            )
            
            # Re-rank results
            reranked_results = self.rerank_results(query, semantic_results)
            
            # Prioritize specifically mentioned pages
            relevant_pages = []
            if mentioned_pages:
                relevant_pages.extend(mentioned_pages)
            
            # Add semantically relevant pages if we don't have enough
            semantic_pages = [result.payload['page_num'] for result in reranked_results]
            for page in semantic_pages:
                if page not in relevant_pages:
                    relevant_pages.append(page)
            
            # Ensure we don't process too many pages
            relevant_pages = relevant_pages[:3]
            logger.info(f"Processing pages: {relevant_pages}")
            
            # Format and limit text content
            all_text = self.format_content(semantic_results, self.max_tokens // 2)  # Use half tokens for text
            
            # Process images for relevant pages
            analysis_tasks = []
            for page in relevant_pages:
                image_path = os.path.join(Config.IMAGES_FOLDER, f'page_{page}.jpg')
                if os.path.exists(image_path):
                    logger.info(f"Analyzing image for page {page}")
                    analysis_tasks.append(vision_service.analyze_image(image_path, query))
                else:
                    logger.warning(f"Image not found for page {page}")
            
            # Wait for all image analyses to complete
            visual_analyses = []
            if analysis_tasks:
                analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                # Filter out exceptions and None results
                visual_analyses = [a for a in analyses if a is not None and not isinstance(a, Exception)]
            
            # Combine and limit visual analysis results
            visual_context = "\n\n".join(visual_analyses) if visual_analyses else ""
            if visual_context:
                visual_context = self.truncate_to_token_limit(visual_context, self.max_tokens // 4)  # Use quarter tokens for visual
            
            # Calculate remaining tokens for the prompt
            used_tokens = (
                self.count_tokens(all_text) +
                self.count_tokens(visual_context) +
                self.count_tokens(self.system_prompt) +
                self.count_tokens(query)
            )
            
            # Generate final response using async client
            completion = await self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Context from document:\n{all_text}\n\n"
                                 f"Visual analysis of relevant drawings:\n{visual_context}\n\n"
                                 f"User query: {query}\n\n"
                                 f"Provide a short and concise response incorporating both textual and visual information."
                    }
                ],
                max_tokens=800 - used_tokens  # Ensure we don't exceed total limit
            )
            
            # Extract and return the response text
            response_text = completion.choices[0].message.content
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise