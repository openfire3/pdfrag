import google.generativeai as genai
import base64

from app.logger_config import logger

class ImageAnalyzer:
    def __init__(self):
        # Configure Google Generative AI
        genai.configure(api_key="AIzaSyA6VAI5qzNfsMnCr3c5X4z5X7LkQNji6-I")
        self.client = genai.GenerativeModel('gemini-2.0-flash')  # Use the updated model

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
            
            response = self.client.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                        ]
                    }
                ]
            )
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
