# import google.generativeai as genai
import base64
from openai import OpenAI
import os

from app.logger_config import logger
from app.config import Config

class ImageAnalyzer:
    def __init__(self):
        # Configure Google Generative AI
        # genai.configure(api_key="AIzaSyA6VAI5qzNfsMnCr3c5X4z5X7LkQNji6-I")
        # self.client = genai.GenerativeModel('gemini-2.0-flash')  # Use the updated model
        self.client = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def analyze_image(self, image_data, prompt="What is on this page?"):
        try:
            # Convert memoryview to bytes if necessary
            if isinstance(image_data, memoryview):
                image_data = image_data.tobytes()  # Convert memoryview to bytes
            elif isinstance(image_data, bytes):
                pass  # Already in bytes format
            else:
                image_data = image_data.encode('utf-8')  # Convert string to bytes

            # Encode the image data as base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            # Send the image and prompt to the Gemini API
            # logger.info("###############")
            # logger.info(base64_image)
            
            # response = self.client.generate_content(
            #     contents=[
            #         {
            #             "role": "user",
            #             "parts": [
            #                 {"text": prompt},
            #                 {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            #             ]
            #         }
            #     ]
            # )
            # return response.text
            standardized_question = f"""Analyze this technical drawing systematically, focusing on equipment and devices:
                1. Task: {prompt}

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
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {
                            "role": "system",
                            "content": Config.SYSTEM_PROMPT_FOR_IMAGE_ANALYSIS
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
            return response.choices[0]
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
