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
            
        # Using simple float timeout instead of ClientTimeout object
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=30.0  # Simple float timeout in seconds
        )
    
    def encode_image(self, image_path):
        """Convert image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def analyze_image(self, image_path: str, question: str) -> str:
        """Analyze image using Gemini Vision API"""
        if not self.api_key:
            return "Vision analysis is disabled due to missing Gemini API key."
            
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return None
                
            try:
                logger.info(f"Starting Gemini analysis for image: {image_path}")
                response = await self.client.chat.completions.create(
                    model=Config.VISION_MODEL,  # gemini-2.0-flash
                    messages=[
                        {
                            "role": "system",
                            "content": Config.SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
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