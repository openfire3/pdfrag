import base64
from openai import AsyncOpenAI
import aiohttp
import asyncio
from app.config import Config
from app.logger_config import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class VisionService:
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("Gemini API key not configured. Vision analysis will be disabled.")
            return
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=aiohttp.ClientTimeout(total=30)
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
    async def analyze_single_image(self, image_path: str, question: str) -> str:
        """Analyze a single image"""
        if not self.api_key:
            return "Vision analysis is disabled due to missing Gemini API key."
            
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return None
                
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=Config.VISION_MODEL,
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
                ),
                timeout=20
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return None

    async def analyze_image(self, image_path: str, question: str) -> str:
        """Main entry point for image analysis"""
        if not self.api_key:
            return "Vision analysis is disabled due to missing Gemini API key."
            
        return await self.analyze_single_image(image_path, question)