import base64
from openai import AsyncOpenAI
import aiohttp
import asyncio
from app.config import Config
from app.logger_config import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback
import json

class VisionService:
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("Gemini API key not configured. Vision analysis will be disabled.")
            return
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=30.0  # Simple float timeout in seconds
        )
        self._cache = {}  # Simple cache for vision results
    
    def encode_image(self, image_path):
        """Convert image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    def _cache_key(self, image_path: str, query_type: str) -> str:
        """Generate cache key for vision results"""
        return f"{image_path}:{query_type}"
        
    def _get_query_type(self, question: str) -> str:
        """Determine the type of vision query for consistent analysis"""
        question = question.lower()
        if 'how many' in question or 'how much' in question or 'count' in question:
            return 'count_elements'
        if 'where' in question or 'location' in question:
            return 'locate_elements'
        return 'analyze_elements'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def analyze_image(self, image_path: str, question: str) -> str:
        """Analyze image using Gemini Vision API with caching and standardized queries"""
        if not self.api_key:
            return "Vision analysis is disabled due to missing Gemini API key."
            
        try:
            # Determine query type for consistent analysis
            query_type = self._get_query_type(question)
            cache_key = self._cache_key(image_path, query_type)
            
            # Check cache first
            if cache_key in self._cache:
                logger.info(f"Using cached vision analysis for {image_path}")
                return self._cache[cache_key]
            
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return None
                
            try:
                # Standardize the question based on query type
                standardized_question = f"""Analyze this technical drawing systematically, focusing on equipment and devices:

1. Task: {question}

2. Required steps:
   - Examine the entire drawing methodically, section by section
   - Focus on device symbols (CAM, FPD, etc.) and their labels
   - Note room numbers and names for precise locations
   - Look for any connecting elements or relationships
   
3. For each device found:
   - List its exact location (room number/name and position)
   - Note any nearby reference points
   - Describe its orientation or direction if relevant

4. Additional requirements:
   - Double-check all findings
   - Report uncertainty if any areas are unclear
   - Be specific about room numbers and names
   - Count elements multiple times to ensure accuracy
   - Report exact positions using available landmarks

Remember to be thorough and systematic in the analysis."""

                logger.info(f"Starting Gemini analysis for image: {image_path}")
                response = await self.client.chat.completions.create(
                    model=Config.VISION_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": Config.SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": standardized_question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                if not response or not response.choices:
                    logger.error("Empty response from Gemini API")
                    return None
                    
                response_content = response.choices[0].message.content
                logger.info(f"Gemini Vision response for {image_path}:\n{response_content}")
                
                # Cache the result
                self._cache[cache_key] = response_content
                return response_content
                    
            except aiohttp.ClientError as e:
                error_info = {
                    'type': type(e).__name__,
                    'status': getattr(e, 'status', None),
                    'message': str(e),
                    'details': getattr(e, 'message', None),
                }
                logger.error(f"Gemini API connection error: {json.dumps(error_info, indent=2)}")
                raise
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout while calling Gemini API (30s limit exceeded)")
                raise
                
            except Exception as e:
                error_info = {
                    'type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'message': str(e)
                }
                logger.error(f"Unexpected error during Gemini API call: {json.dumps(error_info, indent=2)}")
                raise
                
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"Vision service error: {json.dumps(error_info, indent=2)}")
            return None